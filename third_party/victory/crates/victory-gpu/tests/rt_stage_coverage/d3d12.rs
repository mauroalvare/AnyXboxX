//! Record the fixture's phases in one list and read their report after the fence.

use std::mem::{ManuallyDrop, size_of};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};

use anyhow::{Context, Result};
use victory_shader::emit::host::create_preferred_device;
use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
    IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, PCSTR, s};

use super::{CASES, Case, RAY_END, RAY_X, Report, Stage, assert_report, reference};

const SOURCE: &str = include_str!("phases.hlsl");
const REPORT_BYTES: u64 = size_of::<Report>() as u64;
const SCENE_BYTES: u64 = 6 * size_of::<u32>() as u64;
const FENCE_TIMEOUT_MS: u32 = 10_000;

fn device(warp: bool) -> Result<(ID3D12Device, String)> {
    // SAFETY: query the adapter description for the created device's LUID.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let device: ID3D12Device = if warp {
            let adapter: IDXGIAdapter1 = factory.EnumWarpAdapter()?;
            let mut device = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            device.context("no WARP device")?
        } else {
            create_preferred_device()?
        };
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid())?;
        let desc = adapter.GetDesc1()?;
        let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        anyhow::ensure!(software == warp, "unexpected adapter type for WARP={warp}");
        let name = String::from_utf16_lossy(&desc.Description);
        Ok((device, name.trim_end_matches('\0').to_owned()))
    }
}

fn compile(entry: PCSTR) -> ID3DBlob {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the checked-in source and entry point are live through the call.
    let result = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("rt_stage_coverage/phases.hlsl"),
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
    if let Err(error) = result {
        let diagnostics = errors.as_ref().map(blob_bytes).map(String::from_utf8_lossy);
        // SAFETY: callers pass static, nul-terminated entry names.
        let name = unsafe { entry.to_string() }.expect("entry name");
        panic!(
            "compiling {name}: {error}: {}",
            diagnostics.unwrap_or_default()
        );
    }
    code.expect("compiled compute shader")
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob owns this memory for its lifetime.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

fn pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    code: &ID3DBlob,
) -> ID3D12PipelineState {
    let bytes = blob_bytes(code);
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        },
        ..Default::default()
    };
    // SAFETY: the root signature and compiled shader stay live through creation.
    let pso = unsafe { device.CreateComputePipelineState(&desc) };
    drop(ManuallyDrop::into_inner(desc.pRootSignature));
    pso.expect("fixture compute pipeline")
}

fn buffer(device: &ID3D12Device, bytes: u64, readback: bool) -> ID3D12Resource {
    let (heap, flags, state) = if readback {
        (
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
    } else {
        (
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: heap,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut resource = None;
    // SAFETY: valid row-major buffer and heap description.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut resource,
        )
    }
    .expect("fixture buffer creation");
    resource.expect("fixture buffer")
}

struct Fixture {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    root: ID3D12RootSignature,
    reset: ID3D12PipelineState,
    build: ID3D12PipelineState,
    bind: ID3D12PipelineState,
    resolve: ID3D12PipelineState,
    output: ID3D12PipelineState,
    scene: ID3D12Resource,
    spare_scene: ID3D12Resource,
    report: ID3D12Resource,
    readback: ID3D12Resource,
    fence: ID3D12Fence,
    finished: OwnedHandle,
}

impl Fixture {
    fn new(device: ID3D12Device) -> Self {
        let reset = compile(s!("reset_report"));
        // SAFETY: create a root signature from the compiled shader and objects
        // on the same device. The direct list is closed before its first reset.
        unsafe {
            let root = device
                .CreateRootSignature(0, blob_bytes(&reset))
                .expect("root signature");
            let queue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
                .expect("direct command queue");
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("direct command allocator");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .expect("direct command list");
            list.Close().expect("close new list");
            let event = CreateEventW(None, false, false, None).expect("fence event");
            let finished = OwnedHandle::from_raw_handle(event.0);
            Self {
                queue,
                allocator,
                list,
                reset: pipeline(&device, &root, &reset),
                build: pipeline(&device, &root, &compile(s!("build_scene"))),
                bind: pipeline(&device, &root, &compile(s!("bind_scene"))),
                resolve: pipeline(&device, &root, &compile(s!("resolve_visibility"))),
                output: pipeline(&device, &root, &compile(s!("publish_lighting"))),
                scene: buffer(&device, SCENE_BYTES, false),
                spare_scene: buffer(&device, SCENE_BYTES, false),
                report: buffer(&device, REPORT_BYTES, false),
                readback: buffer(&device, REPORT_BYTES, true),
                root,
                fence: device
                    .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .expect("fixture fence"),
                finished,
                device,
            }
        }
    }

    fn dispatch(&self, pipeline: &ID3D12PipelineState, scene: &ID3D12Resource) {
        // SAFETY: the list is recording with the fixture root signature. Both
        // bound buffers remain live, and the UAV barrier orders the next phase.
        unsafe {
            self.list.SetPipelineState(pipeline);
            self.list
                .SetComputeRootUnorderedAccessView(1, scene.GetGPUVirtualAddress());
            self.list.Dispatch(1, 1, 1);
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER::default()),
                },
            }]);
        }
    }

    fn frame(&mut self, case: Case, generation: u32) -> Report {
        let [x0, x1] = case.occluder.map_or([0; 2], |o| o.x);
        let [z0, z1] = case.occluder.map_or([0; 2], |o| o.z);
        let inputs = [
            generation,
            u32::from(case.occluder.is_some()),
            x0 as u32,
            x1 as u32,
            z0 as u32,
            z1 as u32,
            RAY_X[0] as u32,
            RAY_X[1] as u32,
            RAY_END as u32,
            0,
        ];
        // SAFETY: frame() waits for the fence before returning, so the allocator
        // is idle. The arguments match the root signature in phases.hlsl.
        unsafe {
            self.allocator.Reset().expect("reset allocator");
            self.list.Reset(&self.allocator, None).expect("reset list");
            self.list.SetComputeRootSignature(&self.root);
            self.list.SetComputeRoot32BitConstants(
                0,
                inputs.len() as u32,
                inputs.as_ptr().cast(),
                0,
            );
            self.list
                .SetComputeRootUnorderedAccessView(2, self.report.GetGPUVirtualAddress());
        }
        self.dispatch(&self.reset, &self.spare_scene);
        if case.stage != Stage::MissingBuild {
            self.dispatch(&self.build, &self.scene);
        }
        let bound = if case.stage == Stage::RejectedBinding {
            &self.spare_scene
        } else {
            &self.scene
        };
        self.dispatch(&self.bind, bound);
        if case.stage != Stage::UnavailableResolve {
            self.dispatch(&self.resolve, bound);
        }
        self.dispatch(&self.output, bound);
        self.copy_report();
        self.submit_and_read(u64::from(generation))
    }

    fn copy_report(&self) {
        // The buffer promotes from COMMON on first UAV access and decays to
        // COMMON again when the submitted list finishes.
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(self.report.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        };
        // SAFETY: the list is recording, the transition arm is initialized,
        // and the copy uses the full size of the two live buffers.
        unsafe {
            self.list.ResourceBarrier(std::slice::from_ref(&barrier));
            let transition = ManuallyDrop::into_inner(barrier.Anonymous.Transition);
            drop(ManuallyDrop::into_inner(transition.pResource));
            self.list
                .CopyBufferRegion(&self.readback, 0, &self.report, 0, REPORT_BYTES);
        }
    }

    fn submit_and_read(&self, point: u64) -> Report {
        let event = HANDLE(self.finished.as_raw_handle());
        // SAFETY: all command resources and the event remain live through the
        // fence. The returned record consists of aligned, initialized dwords.
        unsafe {
            self.list.Close().expect("close fixture list");
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast().expect("command list"))]);
            self.queue
                .Signal(&self.fence, point)
                .expect("signal fixture fence");
            self.fence
                .SetEventOnCompletion(point, event)
                .expect("set fence event");
            assert_eq!(
                WaitForSingleObject(event, FENCE_TIMEOUT_MS),
                WAIT_OBJECT_0,
                "fixture fence wait"
            );
            self.device
                .GetDeviceRemovedReason()
                .expect("fixture device status");
            let read = D3D12_RANGE {
                Begin: 0,
                End: REPORT_BYTES as usize,
            };
            let mut ptr = std::ptr::null_mut();
            self.readback
                .Map(0, Some(&read), Some(&mut ptr))
                .expect("map frame report");
            let result = ptr.cast::<Report>().read();
            self.readback.Unmap(0, Some(&D3D12_RANGE::default()));
            result
        }
    }
}

fn check_adapter(warp: bool, created: Result<(ID3D12Device, String)>) {
    let (device, name) = match created {
        Ok(found) => found,
        Err(error) => {
            let hardware_requested =
                std::env::var_os("VICTORY_D3D12_ADAPTER").is_some_and(|v| !v.is_empty());
            let required = std::env::var_os("ANYXBOXX_REQUIRE_RUNTIME").is_some_and(|v| v == "1");
            if warp || required || hardware_requested {
                let backend = if warp { "WARP" } else { "hardware" };
                panic!("D3D12 {backend} fixture device unavailable ({error:#})");
            }
            eprintln!("SKIP D3D12 hardware fixture ({error:#})");
            return;
        }
    };
    eprintln!("D3D12 stage coverage on {name}");
    let mut fixture = Fixture::new(device);
    for (index, case) in CASES.into_iter().enumerate() {
        let generation = index as u32 + 1;
        let result = fixture.frame(case, generation);
        assert_report(&result, &reference(case, generation), case.name);
        eprintln!("{}: {result:?}", case.name);
    }
}

#[test]
fn d3d12_hardware_matches_cpu_phase_oracle() {
    check_adapter(false, device(false));
}

#[test]
fn d3d12_warp_matches_cpu_phase_oracle() {
    check_adapter(true, device(true));
}

#[test]
#[should_panic(expected = "D3D12 WARP fixture device unavailable (synthetic WARP setup error)")]
fn missing_warp_device_fails_the_fixture() {
    check_adapter(true, Err(anyhow::anyhow!("synthetic WARP setup error")));
}
