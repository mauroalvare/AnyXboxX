//! Synthetic end-to-end ray-tracing render-pass integration tests and GPU
//! timing benchmark (no game files).
//!
//! Exercises the pass chain from ray activation (`ActivateRays`) through
//! indirect dispatch clamping (`ExecuteIndirect`) and ray traversal against a
//! movable BVH box occluder to lighting resolve (`R9G9B9E5` `RWTexture2D<uint>`,
//! matching `CS_ScreenProbes_TemporalResolve` / `crate::ambient`).

use std::mem::ManuallyDrop;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
    IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, s};

const RT_PASSES_HLSL: &str = r#"
cbuffer Scene : register(b0) {
    uint width;
    uint height;
    uint enable_rt;
    uint pad0;
    float4 occluder_min; // xyz = box min, w = ray t_max
    float4 occluder_max; // xyz = box max, w = unused
    float4 light_dir;    // xyz = unit direction toward light, w = direct intensity
    float4 ambient_rgb;  // xyz = ambient floor, w = unused
};

struct RayRecord {
    float3 origin;
    float t_max;
    float3 dir;
    uint pixel;
};

#ifdef ACTIVATE
#define RS "RootConstants(num32BitConstants=20, b0), UAV(u0), UAV(u1)"
RWStructuredBuffer<RayRecord> rays : register(u0);
RWByteAddressBuffer indirect_args : register(u1);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void activate_rays(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    uint idx = id.y * width + id.x;
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    float2 world_xy = (uv * 2.0 - 1.0) * float2(8.0, 4.5);
    RayRecord r;
    r.origin = float3(world_xy.x, world_xy.y, 10.0);
    r.t_max = occluder_min.w;
    r.dir = light_dir.xyz;
    r.pixel = idx;
    rays[idx] = r;
    if (idx == 0) {
        uint total = width * height;
        indirect_args.Store3(0, uint3((total + 63u) / 64u, 1u, 1u));
    }
}
#endif

#ifdef CLAMP
#define RS "SRV(t0), UAV(u0)"
ByteAddressBuffer src_args : register(t0);
RWByteAddressBuffer dst_args : register(u0);

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void clamp_args() {
    uint3 g = min(src_args.Load3(0), uint3(65535u, 65535u, 65535u));
    dst_args.Store3(0, g);
}
#endif

#ifdef TRACE
#define RS "RootConstants(num32BitConstants=20, b0), SRV(t0), UAV(u0)"
StructuredBuffer<RayRecord> rays : register(t0);
RWStructuredBuffer<float> visibility : register(u0);

bool hit_box(float3 ro, float3 rd, float t_max, float3 bmin, float3 bmax) {
    float3 inv_d = 1.0 / (abs(rd) < 1e-6 ? float3(1e-6, 1e-6, 1e-6) : rd);
    float3 t0 = (bmin - ro) * inv_d;
    float3 t1 = (bmax - ro) * inv_d;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.001));
    float exit_t = min(min(tmax.x, tmax.y), min(tmax.z, t_max));
    return exit_t >= enter;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void trace_rays(uint3 id : SV_DispatchThreadID) {
    uint total = width * height;
    if (id.x >= total) return;
    RayRecord r = rays[id.x];
    bool occluded = hit_box(r.origin, r.dir, r.t_max, occluder_min.xyz, occluder_max.xyz);
    visibility[r.pixel] = occluded ? 0.0 : 1.0;
}
#endif

#ifdef RESOLVE
#define RS "RootConstants(num32BitConstants=20, b0), SRV(t0), DescriptorTable(UAV(u0))"
StructuredBuffer<float> visibility : register(t0);
RWTexture2D<uint> lighting : register(u0);

uint pack_r9g9b9e5(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void resolve_lighting(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    uint idx = id.y * width + id.x;
    float vis = (enable_rt != 0u) ? visibility[idx] : 1.0;
    float3 direct = float3(1.0, 0.95, 0.85) * light_dir.w * vis;
    float3 total = ambient_rgb.xyz + direct;
    lighting[id.xy] = pack_r9g9b9e5(total);
}
#endif
"#;

fn unpack_r9g9b9e5(v: u32) -> [f32; 3] {
    let scale = 2.0f32.powi((v >> 27) as i32 - 24);
    [
        (v & 511) as f32 * scale,
        ((v >> 9) & 511) as f32 * scale,
        ((v >> 18) & 511) as f32 * scale,
    ]
}

fn compile_cs(define: &str, entry: PCSTR) -> Vec<u8> {
    let src = format!("#define {define}\n{RT_PASSES_HLSL}");
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: valid HLSL source and out pointers.
    let hr = unsafe {
        D3DCompile(
            src.as_ptr().cast(),
            src.len(),
            s!("rt_passes"),
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = hr {
        let msg = errors
            .map(|b| unsafe {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .into_owned()
            })
            .unwrap_or_default();
        panic!("D3DCompile failed for {define}: {e}: {msg}");
    }
    let blob = code.unwrap();
    // SAFETY: valid compiled bytecode blob.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    }
}

fn make_compute(
    device: &ID3D12Device,
    define: &str,
    entry: PCSTR,
) -> (ID3D12RootSignature, ID3D12PipelineState) {
    let bytes = compile_cs(define, entry);
    // SAFETY: compiled DXBC with embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes).unwrap();
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let pso = device.CreateComputePipelineState(&desc).unwrap();
        (root, pso)
    }
}

fn create_device() -> Option<(ID3D12Device, String)> {
    // SAFETY: standard DXGI + D3D12 enumeration and device creation.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let mut chosen: Option<(IDXGIAdapter1, String)> = None;
        for i in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(i) else {
                break;
            };
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                && let Some(dev) = dev
            {
                return Some((dev, name));
            }
            chosen = Some((adapter, name));
        }
        let _ = chosen;
        let warp: IDXGIAdapter1 = factory.EnumWarpAdapter().ok()?;
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_12_0, &mut dev).ok()?;
        Some((dev?, "WARP".to_owned()))
    }
}

fn make_buffer(
    device: &ID3D12Device,
    bytes: u64,
    heap: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let props = D3D12_HEAP_PROPERTIES {
        Type: heap,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid committed buffer description.
    unsafe {
        device
            .CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res)
            .unwrap();
    }
    res.unwrap()
}

fn make_lighting_tex(device: &ID3D12Device, width: u32, height: u32) -> ID3D12Resource {
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_UINT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid committed UAV texture description.
    unsafe {
        device
            .CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
                &mut res,
            )
            .unwrap();
    }
    res.unwrap()
}

fn uav_barrier_of(res: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: borrowed COM pointer valid for the ResourceBarrier call.
                pResource: ManuallyDrop::new(res.map(|r| unsafe { std::mem::transmute_copy(r) })),
            }),
        },
    }
}

fn transition_of(
    res: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    crate::host::transition(res, before, after)
}

struct RtSceneRunner {
    width: u32,
    height: u32,
    adapter_name: String,
    _device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    fence_value: u64,
    activate: (ID3D12RootSignature, ID3D12PipelineState),
    clamp: (ID3D12RootSignature, ID3D12PipelineState),
    trace: (ID3D12RootSignature, ID3D12PipelineState),
    resolve: (ID3D12RootSignature, ID3D12PipelineState),
    dispatch_sig: ID3D12CommandSignature,
    uav_heap: ID3D12DescriptorHeap,
    rays: ID3D12Resource,
    raw_args: ID3D12Resource,
    clamped_args: ID3D12Resource,
    visibility: ID3D12Resource,
    lighting: ID3D12Resource,
    readback: ID3D12Resource,
    row_pitch: u32,
    zero_upload: ID3D12Resource,
    ts_heap: ID3D12QueryHeap,
    ts_readback: ID3D12Resource,
    ts_freq: u64,
}

impl RtSceneRunner {
    fn new(width: u32, height: u32) -> Option<Self> {
        let (device, adapter_name) = create_device()?;
        // SAFETY: standard D3D12 object creation on valid device.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .ok()?;
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let ts_freq = queue.GetTimestampFrequency().unwrap_or(1);

            let activate = make_compute(&device, "ACTIVATE", s!("activate_rays"));
            let clamp = make_compute(&device, "CLAMP", s!("clamp_args"));
            let trace = make_compute(&device, "TRACE", s!("trace_rays"));
            let resolve = make_compute(&device, "RESOLVE", s!("resolve_lighting"));

            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                ..Default::default()
            };
            let mut dispatch_sig: Option<ID3D12CommandSignature> = None;
            device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &arg,
                        NodeMask: 0,
                    },
                    None::<&ID3D12RootSignature>,
                    &mut dispatch_sig,
                )
                .ok()?;
            let dispatch_sig = dispatch_sig?;

            let uav_heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;

            let pixels = u64::from(width) * u64::from(height);
            let rays = make_buffer(
                &device,
                pixels * 32,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let raw_args = make_buffer(
                &device,
                256,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let clamped_args = make_buffer(
                &device,
                256,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let visibility = make_buffer(
                &device,
                pixels * 4,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let lighting = make_lighting_tex(&device, width, height);

            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV {
                        MipSlice: 0,
                        PlaneSlice: 0,
                    },
                },
            };
            device.CreateUnorderedAccessView(
                &lighting,
                None,
                Some(&uav_desc),
                uav_heap.GetCPUDescriptorHandleForHeapStart(),
            );

            let row_pitch = (width * 4).next_multiple_of(256);
            let readback = make_buffer(
                &device,
                u64::from(row_pitch) * u64::from(height),
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            // Upload buffer initialized to 0 (args) and 1.0f (visibility) for
            // clean frame resets between test cases.
            let zero_bytes = 256 + pixels * 4;
            let zero_upload = make_buffer(
                &device,
                zero_bytes,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            zero_upload.Map(0, None, Some(&mut mapped)).ok()?;
            std::ptr::write_bytes(mapped.cast::<u8>(), 0, 256);
            let vis_slice = std::slice::from_raw_parts_mut(
                mapped.cast::<u8>().add(256).cast::<f32>(),
                pixels as usize,
            );
            vis_slice.fill(1.0);
            zero_upload.Unmap(0, None);

            let mut ts_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 16,
                        NodeMask: 0,
                    },
                    &mut ts_heap,
                )
                .ok()?;
            let ts_heap = ts_heap?;
            let ts_readback = make_buffer(
                &device,
                16 * 8,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            Some(Self {
                width,
                height,
                adapter_name,
                _device: device,
                queue,
                allocator,
                list,
                fence,
                event,
                fence_value: 0,
                activate,
                clamp,
                trace,
                resolve,
                dispatch_sig,
                uav_heap,
                rays,
                raw_args,
                clamped_args,
                visibility,
                lighting,
                readback,
                row_pitch,
                zero_upload,
                ts_heap,
                ts_readback,
                ts_freq,
            })
        }
    }

    fn wait(&mut self) {
        self.fence_value += 1;
        // SAFETY: valid queue, fence, and Win32 event handle.
        unsafe {
            self.queue.Signal(&self.fence, self.fence_value).unwrap();
            self.fence
                .SetEventOnCompletion(self.fence_value, self.event)
                .unwrap();
            WaitForSingleObject(self.event, INFINITE);
        }
    }

    /// Runs one synthetic frame and returns `(gpu_pass_ms: [activate, clamp, trace, resolve, total], pixels)`.
    /// When `read_pixels` is false, skips texture readback (for clean GPU timing).
    fn run_frame(
        &mut self,
        occluder_min: [f32; 3],
        occluder_max: [f32; 3],
        enable_rt: bool,
        run_activate: bool,
        read_pixels: bool,
    ) -> ([f64; 5], Vec<u32>) {
        let constants: [u32; 16] = [
            self.width,
            self.height,
            u32::from(enable_rt),
            0,
            occluder_min[0].to_bits(),
            occluder_min[1].to_bits(),
            occluder_min[2].to_bits(),
            20.0f32.to_bits(),
            occluder_max[0].to_bits(),
            occluder_max[1].to_bits(),
            occluder_max[2].to_bits(),
            0,
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            (-1.0f32).to_bits(),
            2.0f32.to_bits(),
        ];
        // Note: ambient_rgb is constants[12..16] — pass [0.1, 0.1, 0.1] by overwriting
        // through a second 16-dword array matching Scene cbuffer (16 dwords total):
        // cbuffer Scene has 4 uints + 4 float4s = 20 dwords! Let's pack all 20 cleanly.
        let _ = constants;
        let scene: [u32; 20] = [
            self.width,
            self.height,
            u32::from(enable_rt),
            0,
            occluder_min[0].to_bits(),
            occluder_min[1].to_bits(),
            occluder_min[2].to_bits(),
            20.0f32.to_bits(),
            occluder_max[0].to_bits(),
            occluder_max[1].to_bits(),
            occluder_max[2].to_bits(),
            0,
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            (-1.0f32).to_bits(),
            2.0f32.to_bits(),
            0.10f32.to_bits(),
            0.10f32.to_bits(),
            0.10f32.to_bits(),
            0,
        ];
        let pixels = u64::from(self.width) * u64::from(self.height);
        // SAFETY: valid open command list and live D3D12 resources.
        unsafe {
            self.allocator.Reset().unwrap();
            self.list.Reset(&self.allocator, None).unwrap();

            // Reset args to 0 and visibility to 1.0 when testing skipped activation
            // so we prove visibility only changes when ActivateRays actually runs.
            if read_pixels {
                self.list
                    .CopyBufferRegion(&self.raw_args, 0, &self.zero_upload, 0, 12);
                self.list.CopyBufferRegion(
                    &self.visibility,
                    0,
                    &self.zero_upload,
                    256,
                    pixels * 4,
                );
                self.list.ResourceBarrier(&[
                    transition_of(
                        &self.raw_args,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                    transition_of(
                        &self.visibility,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
            }

            self.list
                .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);

            if enable_rt {
                // Pass 1: ActivateRays
                if run_activate {
                    self.list.SetComputeRootSignature(&self.activate.0);
                    self.list.SetPipelineState(&self.activate.1);
                    self.list
                        .SetComputeRoot32BitConstants(0, 20, scene.as_ptr().cast(), 0);
                    self.list
                        .SetComputeRootUnorderedAccessView(1, self.rays.GetGPUVirtualAddress());
                    self.list.SetComputeRootUnorderedAccessView(
                        2,
                        self.raw_args.GetGPUVirtualAddress(),
                    );
                    self.list
                        .Dispatch(self.width.div_ceil(8), self.height.div_ceil(8), 1);
                }
                self.list
                    .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);

                // Producer -> Consumer barrier: ActivateRays -> Indirect Clamp
                self.list.ResourceBarrier(&[
                    transition_of(
                        &self.raw_args,
                        if run_activate {
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                        } else {
                            D3D12_RESOURCE_STATE_COMMON
                        },
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    ),
                    transition_of(
                        &self.clamped_args,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ),
                ]);

                // Indirect argument clamp (mirrors victory-gpu indirect.rs)
                self.list.SetComputeRootSignature(&self.clamp.0);
                self.list.SetPipelineState(&self.clamp.1);
                self.list
                    .SetComputeRootShaderResourceView(0, self.raw_args.GetGPUVirtualAddress());
                self.list.SetComputeRootUnorderedAccessView(
                    1,
                    self.clamped_args.GetGPUVirtualAddress(),
                );
                self.list.Dispatch(1, 1, 1);

                self.list.ResourceBarrier(&[
                    transition_of(
                        &self.rays,
                        if run_activate {
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                        } else {
                            D3D12_RESOURCE_STATE_COMMON
                        },
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    ),
                    transition_of(
                        &self.clamped_args,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    ),
                ]);
                self.list
                    .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 2);

                // Pass 2: Indirect RayTrace against movable occluder
                self.list.SetComputeRootSignature(&self.trace.0);
                self.list.SetPipelineState(&self.trace.1);
                self.list
                    .SetComputeRoot32BitConstants(0, 20, scene.as_ptr().cast(), 0);
                self.list
                    .SetComputeRootShaderResourceView(1, self.rays.GetGPUVirtualAddress());
                self.list.SetComputeRootUnorderedAccessView(
                    2,
                    self.visibility.GetGPUVirtualAddress(),
                );
                self.list
                    .ExecuteIndirect(&self.dispatch_sig, 1, &self.clamped_args, 0, None, 0);

                // Return buffers to readable/COMMON states + sync visibility for Resolve
                self.list.ResourceBarrier(&[
                    transition_of(
                        &self.visibility,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    ),
                    transition_of(
                        &self.raw_args,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                    transition_of(
                        &self.rays,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                    transition_of(
                        &self.clamped_args,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
                self.list
                    .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 3);
            } else {
                self.list
                    .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                self.list
                    .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 2);
                self.list
                    .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 3);
            }

            // Pass 3: Resolve lighting into R9G9B9E5 (R32_UINT)
            let heaps = [Some(self.uav_heap.clone())];
            self.list.SetDescriptorHeaps(&heaps);
            self.list.SetComputeRootSignature(&self.resolve.0);
            self.list.SetPipelineState(&self.resolve.1);
            self.list
                .SetComputeRoot32BitConstants(0, 20, scene.as_ptr().cast(), 0);
            self.list
                .SetComputeRootShaderResourceView(1, self.visibility.GetGPUVirtualAddress());
            self.list.SetComputeRootDescriptorTable(
                2,
                self.uav_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            self.list
                .Dispatch(self.width.div_ceil(8), self.height.div_ceil(8), 1);
            if enable_rt {
                self.list.ResourceBarrier(&[
                    uav_barrier_of(Some(&self.lighting)),
                    transition_of(
                        &self.visibility,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
            } else {
                self.list.ResourceBarrier(&[uav_barrier_of(Some(&self.lighting))]);
            }
            self.list
                .EndQuery(&self.ts_heap, D3D12_QUERY_TYPE_TIMESTAMP, 4);

            self.list.ResolveQueryData(
                &self.ts_heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                5,
                &self.ts_readback,
                0,
            );

            if read_pixels {
                self.list.ResourceBarrier(&[transition_of(
                    &self.lighting,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(self.readback.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R32_UINT,
                                Width: self.width,
                                Height: self.height,
                                Depth: 1,
                                RowPitch: self.row_pitch,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: ManuallyDrop::new(Some(self.lighting.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                self.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                self.list.ResourceBarrier(&[transition_of(
                    &self.lighting,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }

            self.list.Close().unwrap();
            let cmd: ID3D12CommandList = self.list.cast().unwrap();
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
        }

        self.wait();

        let mut ts = [0u64; 5];
        // SAFETY: readback buffer is idle after fence wait.
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            self.ts_readback.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(ptr.cast::<u64>(), ts.as_mut_ptr(), 5);
            self.ts_readback.Unmap(0, None);
        }
        let to_ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * 1000.0 / self.ts_freq as f64;
        let timings = [
            to_ms(ts[0], ts[1]),
            to_ms(ts[1], ts[2]),
            to_ms(ts[2], ts[3]),
            to_ms(ts[3], ts[4]),
            to_ms(ts[0], ts[4]),
        ];

        let mut out = Vec::new();
        if read_pixels {
            out.resize((self.width * self.height) as usize, 0);
            // SAFETY: readback buffer is idle after fence wait.
            unsafe {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                self.readback.Map(0, None, Some(&mut ptr)).unwrap();
                for y in 0..self.height as usize {
                    let row = ptr.cast::<u8>().add(y * self.row_pitch as usize).cast::<u32>();
                    std::ptr::copy_nonoverlapping(
                        row,
                        out.as_mut_ptr().add(y * self.width as usize),
                        self.width as usize,
                    );
                }
                self.readback.Unmap(0, None);
            }
        }
        (timings, out)
    }
}

/// End-to-end synthetic RT pass verification: an occluder blocking the light
/// casts a shadow in the resolved `R9G9B9E5` lighting texture, and moving the
/// occluder shifts the shadow accurately. Also confirms that skipping
/// `ActivateRays` leaves rays unlaunched and loses occlusion.
#[test]
fn movable_occluder_changes_rt_lighting_output() {
    let Some(mut runner) = RtSceneRunner::new(64, 36) else {
        return;
    };
    let luma = |px: &[u32], x: u32, y: u32| -> f32 {
        let rgb = unpack_r9g9b9e5(px[(y * 64 + x) as usize]);
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    };

    // Position A: occluder on the left half (world x in [-6, -1], y in [-3, 3]).
    let (_, left_frame) = runner.run_frame([-6.0, -3.0, 4.0], [-1.0, 3.0, 5.0], true, true, true);
    let left_a = luma(&left_frame, 16, 18);  // world x ~= -4 (behind occluder A)
    let right_a = luma(&left_frame, 48, 18); // world x ~= +4 (unoccluded)
    assert!(
        (0.05..0.20).contains(&left_a) && right_a > 1.5,
        "occluder A should shadow left to ambient ({left_a}) and leave right lit ({right_a})"
    );

    // Position B: move occluder to the right half (world x in [1, 6], y in [-3, 3]).
    let (_, right_frame) = runner.run_frame([1.0, -3.0, 4.0], [6.0, 3.0, 5.0], true, true, true);
    let left_b = luma(&right_frame, 16, 18);  // now unoccluded
    let right_b = luma(&right_frame, 48, 18); // now behind occluder B
    assert!(
        left_b > 1.5 && (0.05..0.20).contains(&right_b),
        "moving occluder to B should light left ({left_b}) and shadow right ({right_b})"
    );
    assert_ne!(left_frame, right_frame);

    // Negative test: skipping ActivateRays (the old VICTORY_RT_RESOLVE=1 bug)
    // leaves indirect dispatch args at 0 so the occluder never shadows the frame.
    let (_, skipped_frame) = runner.run_frame([-6.0, -3.0, 4.0], [-1.0, 3.0, 5.0], true, false, true);
    let left_skipped = luma(&skipped_frame, 16, 18);
    assert!(
        left_skipped > 1.5,
        "skipping ActivateRays should fail to cast shadow (got luma {left_skipped})"
    );
}

/// GPU timing benchmark at 1920x1080 comparing RT OFF vs RT ON against the
/// 33.3 ms (30 FPS) frame budget. Run with:
/// `cargo test -p victory-gpu --release --lib rt_passes_tests::rt_passes_frame_cost_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn rt_passes_frame_cost_bench() {
    let mut runner = RtSceneRunner::new(1920, 1080).expect("D3D12 device");
    const WARMUP: usize = 10;
    const ITERS: usize = 50;
    const BUDGET_MS: f64 = 33.333;

    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };

    for _ in 0..WARMUP {
        runner.run_frame([-6.0, -3.0, 4.0], [-1.0, 3.0, 5.0], false, false, false);
    }
    let mut off_total = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let (t, _) = runner.run_frame([-6.0, -3.0, 4.0], [-1.0, 3.0, 5.0], false, false, false);
        off_total.push(t[4]);
    }

    for _ in 0..WARMUP {
        runner.run_frame([-6.0, -3.0, 4.0], [-1.0, 3.0, 5.0], true, true, false);
    }
    let mut on_passes = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for i in 0..ITERS {
        let x0 = if i % 2 == 0 { -6.0 } else { 1.0 };
        let (t, _) = runner.run_frame([x0, -3.0, 4.0], [x0 + 5.0, 3.0, 5.0], true, true, false);
        for k in 0..5 {
            on_passes[k].push(t[k]);
        }
    }

    let rt_off_ms = median(off_total);
    let act_ms = median(on_passes[0].clone());
    let clamp_ms = median(on_passes[1].clone());
    let trace_ms = median(on_passes[2].clone());
    let res_ms = median(on_passes[3].clone());
    let rt_on_ms = median(on_passes[4].clone());
    let rt_cost_ms = rt_on_ms - rt_off_ms;

    println!(
        "adapter={} res=1920x1080 rays=2073600 budget_30fps={BUDGET_MS:.3}ms",
        runner.adapter_name
    );
    println!(
        "RT_OFF median_total={rt_off_ms:.4}ms ({:.2}% of 33.3ms)",
        rt_off_ms / BUDGET_MS * 100.0
    );
    println!(
        "RT_ON  median_total={rt_on_ms:.4}ms ({:.2}% of 33.3ms) | activate={act_ms:.4}ms clamp={clamp_ms:.4}ms trace={trace_ms:.4}ms resolve={res_ms:.4}ms",
        rt_on_ms / BUDGET_MS * 100.0
    );
    println!(
        "RT_ENABLE_COST delta={rt_cost_ms:+.4}ms ({:.2}% of 33.3ms)",
        rt_cost_ms / BUDGET_MS * 100.0
    );
}
