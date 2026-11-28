use super::*;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};

const PROBE: &str = r#"
    Texture2D<float4> tex : register(t0);
    SamplerState pointMip : register(s0);
    SamplerState linearMip : register(s1);
    RWByteAddressBuffer result : register(u0);
    [RootSignature("DescriptorTable(SRV(t0)), DescriptorTable(Sampler(s0, numDescriptors=2)), UAV(u0)")]
    [numthreads(1, 1, 1)]
    void main() {
        float2 uv = float2(0.5, 0.5);
        result.Store4(0, asuint(float4(
            tex.SampleLevel(pointMip, uv, 0.25).r,
            tex.SampleLevel(pointMip, uv, 0.75).r,
            tex.SampleLevel(linearMip, uv, 0.25).r,
            tex.SampleLevel(linearMip, uv, 0.75).r
        )));
    }
"#;

#[test]
fn anisotropic_heap_sampler_reads_the_requested_mip() {
    let Some(host) = host() else {
        eprintln!("no host D3D12 adapter; skipping anisotropic sampling readback");
        return;
    };
    let adapter = crate::host::adapter_summary()
        .expect("identified test adapter")
        .0;
    let device = &host.device;
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
    // SAFETY: correctly sized storage for the OPTIONS19 query.
    let options19 = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS19,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    }
    .is_ok();
    let point_supported = options19 && options.AnisoFilterWithPointMipSupported.as_bool();

    let list = host.new_list().unwrap();
    let result = host.uav_buffer(16).unwrap();
    let readback = host.read_back_buffer(16).unwrap();
    // SAFETY: live resources, descriptor heaps sized for the test, and
    // a single shader thread with bounded accesses.
    let resources = unsafe {
        let heap = |kind, count, flags| {
            device
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        };
        let sampler_heap = heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            2,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        );
        let srv_heap = heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            1,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        );
        let rtv_heap = heap(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            2,
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        );
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 2,
            Height: 2,
            DepthOrArraySize: 1,
            MipLevels: 2,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &texture_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut texture,
            )
            .unwrap();
        let texture = texture.unwrap();
        let rtv_cpu = rtv_heap.GetCPUDescriptorHandleForHeapStart();
        let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for mip in 0..2 {
            let rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_cpu.ptr + (mip * rtv_step) as usize,
            };
            device.CreateRenderTargetView(
                &texture,
                Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: texture_desc.Format,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_RTV {
                            MipSlice: mip,
                            PlaneSlice: 0,
                        },
                    },
                }),
                rtv,
            );
            list.list.ClearRenderTargetView(rtv, &[mip as f32; 4], None);
        }
        list.list.ResourceBarrier(&[crate::host::transition(
            &texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);
        device.CreateShaderResourceView(
            &texture,
            None,
            srv_heap.GetCPUDescriptorHandleForHeapStart(),
        );

        let sampler_cpu = sampler_heap.GetCPUDescriptorHandleForHeapStart();
        let sampler_step =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        let mut words = [0u32; 27];
        words[1] = 3; // anisotropic bilinear magnification
        words[2] = 3; // anisotropic bilinear minification
        words[3] = 1; // point mips
        words[6..9].fill(D3D12_TEXTURE_ADDRESS_MODE_CLAMP.0 as u32);
        words[10] = 1.0f32.to_bits();
        words[13] = 4; // maximum anisotropy
        assert!(make(&View::ConsoleSampler(words), sampler_cpu).is_some());
        // A native linear-mip sampler supplies the fallback oracle on the
        // same adapter, including any differences in the blend weights.
        device.CreateSampler(
            &D3D12_SAMPLER_DESC {
                Filter: D3D12_FILTER_ANISOTROPIC,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxAnisotropy: 4,
                ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
                MaxLOD: 1.0,
                ..Default::default()
            },
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: sampler_cpu.ptr + sampler_step as usize,
            },
        );

        let mut shader: Option<ID3DBlob> = None;
        D3DCompile(
            PROBE.as_ptr().cast(),
            PROBE.len(),
            windows::core::s!("synthetic anisotropic sampler"),
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            0,
            0,
            &mut shader,
            None,
        )
        .unwrap();
        let shader = shader.unwrap();
        let code = std::slice::from_raw_parts(
            shader.GetBufferPointer().cast::<u8>(),
            shader.GetBufferSize(),
        );
        let root: ID3D12RootSignature = device.CreateRootSignature(0, code).unwrap();
        let pso: ID3D12PipelineState = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        list.list
            .SetDescriptorHeaps(&[Some(srv_heap.clone()), Some(sampler_heap.clone())]);
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list
            .SetComputeRootDescriptorTable(0, srv_heap.GetGPUDescriptorHandleForHeapStart());
        list.list
            .SetComputeRootDescriptorTable(1, sampler_heap.GetGPUDescriptorHandleForHeapStart());
        list.list
            .SetComputeRootUnorderedAccessView(2, result.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
        list.list.ResourceBarrier(&[crate::host::transition(
            &result,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.list.CopyBufferRegion(&readback, 0, &result, 0, 16);
        (sampler_heap, srv_heap, rtv_heap, texture, root, pso)
    };
    list.close();
    host.execute(&[&list]);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy completed and readback contains four f32 values.
    let sampled = unsafe {
        device.GetDeviceRemovedReason().unwrap();
        readback.Map(0, None, Some(&mut data)).unwrap();
        let sampled = data.cast::<[f32; 4]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
        sampled
    };
    drop(resources);
    println!(
        "adapter={adapter}, OPTIONS19={options19}, aniso_point_mip={point_supported}, point={:?}, native_linear={:?}",
        &sampled[..2],
        &sampled[2..]
    );
    let linear = &sampled[2..];
    assert!(
        0.0 < linear[0] && linear[0] < linear[1] && linear[1] < 1.0,
        "native linear mip control did not blend: {linear:?}",
    );
    let expected: &[f32] = if point_supported { &[0.0, 1.0] } else { linear };
    assert_eq!(&sampled[..2], expected, "point mip readback");
}
