//! Synthetic GPU verification and sustained-load benchmarks for BVH
//! builder/traversal CPU preparation, batched `land`, TLAS upload scanning
//! (`named_in_upload`), and coalesced write-watch sweeps (`look_again`).
//! Uses purely synthetic RDNA 2 BVH nodes and the real translated
//! `image_bvh64_intersect_ray` shader over `SLOT_MEMORY` + `USER_MEMORY_BASE`
//! + `SLOT_FAULTS` on the host GPU (or WARP), with zero game files.

use std::sync::atomic::Ordering;
use std::time::Instant;

use victory_shader::bvh::{self, Ray};
use victory_shader::dxbc::Container;
use victory_shader::emit::{
    self, HostShader, ROOT_CBV_SPACE, SLOT_FAULTS, SLOT_MEMORY, USER_DATA_SPACE, USER_MEMORY_BASE,
};
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{io, ir, resource, structure};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::*;
use crate::descriptor::host::HostHeap;
use crate::host::{Host, host};

const THREADS: u32 = 32;
const RECORD_BYTES: usize = 64;

fn empty_dxbc() -> Vec<u8> {
    let mut b = vec![0u8; 32];
    b[0..4].copy_from_slice(b"DXBC");
    b[0x14..0x16].copy_from_slice(&1u16.to_le_bytes());
    b[0x18..0x1c].copy_from_slice(&32u32.to_le_bytes());
    b
}

fn trace_program_words() -> Vec<u32> {
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x3428_0086, // v_lshlrev_b32 v20, 6, v0
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    ];
    for k in 0..4u32 {
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k));
    }
    w.extend([
        0xBE98_03FF,
        0,           // s_mov_b32 s24, 0
        0xBE99_0B81, // s_brev_b32 s25, 1 (BOX_SORT = 1 << 31)
        0xBE9A_03C1, // s_mov_b32 s26, -1
        0xBE9B_03FF,
        0x8900_03FF, // s_mov_b32 s27, 0x890003ff
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
        0xF19C_9F07,
        0x0000_0D01 | (24 >> 2) << 16,
        0x0504_0302,
        0x0908_0706,
        0x000C_0B0A,
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0x342A_0084, // v_lshlrev_b32 v21, 4, v0
        0xE078_1000,
        0x8003_0D15, // buffer_store_dwordx4 v[13:16], v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

struct TracePipeline {
    shader: HostShader,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    user_param: u32,
    memory_param: u32,
    faults_param: u32,
}

impl TracePipeline {
    fn new(host: &Host) -> Self {
        crate::descriptor::host::probe_view_limit();
        let words = trace_program_words();
        let table = |kind, at| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: at,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 22,
            layout: vec![
                RootParam {
                    root: RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![table(ResourceType::Uav, 0), table(ResourceType::Uav, 8)],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [THREADS, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let dxbc = empty_dxbc();
        let c = Container::parse(&dxbc).expect("empty dxbc");
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        let io = io::raise(&p, &native).expect("io::raise");
        let shader = emit::emit(&c, &native, &p, &st, &io, &res).expect("emit::emit");

        let dwords = shader.user_data_dwords();
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: USER_DATA_SPACE,
                        Num32BitValues: dwords,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: u32::from(SLOT_MEMORY),
                        RegisterSpace: ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: u32::from(SLOT_FAULTS),
                        RegisterSpace: ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut err: Option<ID3DBlob> = None;
        // SAFETY: valid D3D12 root signature and compute PSO creation.
        let (root, pso) = unsafe {
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut err)).expect("serialize root sig");
            let blob = blob.expect("root sig blob");
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).expect("CreateRootSignature");
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: shader.dxil.as_ptr().cast(),
                        BytecodeLength: shader.dxil.len(),
                    },
                    ..Default::default()
                })
                .expect("CreateComputePipelineState");
            (root, pso)
        };
        Self {
            shader,
            root,
            pso,
            user_param: 0,
            memory_param: 1,
            faults_param: 2,
        }
    }

    fn dispatch(&self, list: &HostList, heap: &HostHeap, ready: &Ready, table_base_slot: u32) {
        let dwords = self.shader.user_data_dwords() as usize;
        let mut values = vec![0u32; dwords];
        for u in &self.shader.user_data {
            values[u.index as usize] = match u.root {
                RootType::Other(USER_MEMORY_BASE) => ready.base,
                RootType::Table => table_base_slot * 32,
                _ => 0,
            };
        }
        // SAFETY: open command list, live pipeline, root sig, heap and buffers.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&self.root);
            list.list.SetPipelineState(&self.pso);
            list.list.SetComputeRoot32BitConstants(
                self.user_param,
                values.len() as u32,
                values.as_ptr().cast(),
                0,
            );
            list.list.SetComputeRootShaderResourceView(self.memory_param, ready.directory);
            list.list.SetComputeRootUnorderedAccessView(self.faults_param, ready.faults);
            list.list.Dispatch(1, 1, 1);
        }
    }
}

fn make_box32(children: [u32; 4], boxes: [[f32; 6]; 4]) -> [u32; 28] {
    let mut w = [0u32; 28];
    w[0..4].copy_from_slice(&children);
    for (i, b) in boxes.iter().enumerate() {
        for k in 0..6 {
            w[4 + i * 6 + k] = b[k].to_bits();
        }
    }
    w
}

fn make_tri(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [u32; 28] {
    let mut w = [0u32; 28];
    for (vi, v) in [v0, v1, v2].iter().enumerate() {
        for c in 0..3 {
            w[vi * 3 + c] = v[c].to_bits();
        }
    }
    w[15] = 0x0000_0009;
    w
}

fn write_packed_bvh(base: usize, nodes: u32, box_node: &[u32; 28], tri_node: &[u32; 28]) {
    let header: u64 = 0x8102_0006u64 | (u64::from(nodes) << 32);
    // SAFETY: caller allocated writable title memory covering `(nodes + 1) * 64` bytes.
    unsafe {
        std::ptr::write_unaligned(base as *mut u64, header);
        std::ptr::copy_nonoverlapping(box_node.as_ptr().cast::<u8>(), (base + 64) as *mut u8, 28 * 4);
        std::ptr::copy_nonoverlapping(tri_node.as_ptr().cast::<u8>(), (base + 192) as *mut u8, 16 * 4);
    }
}

fn write_unpacked_bvh(base: usize, box_node: &[u32; 28], tri_node: &[u32; 28]) {
    let header: u64 = 0x8002_0006u64 | (1023u64 << 32);
    // SAFETY: caller allocated writable title memory covering 64 KiB.
    unsafe {
        std::ptr::write_unaligned(base as *mut u64, header);
        std::ptr::copy_nonoverlapping(box_node.as_ptr().cast::<u8>(), (base + 64) as *mut u8, 28 * 4);
        std::ptr::copy_nonoverlapping(tri_node.as_ptr().cast::<u8>(), (base + 192) as *mut u8, 16 * 4);
    }
}

fn write_ray_record(records_ptr: *mut u8, lane: usize, node_addr: u64, kind: u32, ray: &Ray) {
    let ptr64 = (node_addr >> 3) | u64::from(kind & 7);
    let rec: [u32; 16] = [
        ptr64 as u32,
        (ptr64 >> 32) as u32,
        ray.extent.to_bits(),
        ray.origin[0].to_bits(),
        ray.origin[1].to_bits(),
        ray.origin[2].to_bits(),
        ray.direction[0].to_bits(),
        ray.direction[1].to_bits(),
        ray.direction[2].to_bits(),
        ray.inverse[0].to_bits(),
        ray.inverse[1].to_bits(),
        ray.inverse[2].to_bits(),
        1,
        0,
        0,
        0,
    ];
    // SAFETY: `lane < THREADS`, buffer holds `THREADS * RECORD_BYTES`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            rec.as_ptr().cast::<u8>(),
            records_ptr.add(lane * RECORD_BYTES),
            RECORD_BYTES,
        );
    }
}

fn read_results(results_ptr: *const u8, count: usize) -> Vec<[u32; 4]> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut r = [0u32; 4];
        // SAFETY: mapped results buffer holds at least `count * 16` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(results_ptr.add(i * 16), r.as_mut_ptr().cast::<u8>(), 16);
        }
        out.push(r);
    }
    out
}

fn process_private_mib() -> f64 {
    let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    counters.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    // SAFETY: valid struct pointer and current process handle.
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut counters).cast(),
            size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
    };
    if ok != 0 {
        counters.PrivateUsage as f64 / (1024.0 * 1024.0)
    } else {
        0.0
    }
}

/// Nearest-rank percentile summary `(p50, p95, mean)` matching the independent
/// benchmarking methodology.
fn percentile_summary(mut v: Vec<u64>) -> (f64, u64, f64) {
    v.sort_unstable();
    let n = v.len();
    let p50 = if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) as f64 / 2.0
    } else {
        v[n / 2] as f64
    };
    let p95 = v[(95 * n).div_ceil(100) - 1];
    let mean = v.iter().sum::<u64>() as f64 / n as f64;
    (p50, p95, mean)
}

#[cfg(feature = "diagnostics")]
fn take_watch_asks() -> (u64, u64) {
    let a = crate::watch::asks();
    (a[0], a[11])
}

#[cfg(not(feature = "diagnostics"))]
fn take_watch_asks() -> (u64, u64) {
    (0, 0)
}

struct TitleRegion {
    base: usize,
}

impl TitleRegion {
    fn new(len: usize) -> Self {
        let base = crate::upload::reserve(len).expect("upload::reserve for synthetic title BVH memory");
        Self { base }
    }
}

impl Drop for TitleRegion {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // SAFETY: memory reserved by `VirtualAlloc2` in `upload::reserve`.
        unsafe {
            VirtualFree(self.base as *mut _, 0, MEM_RELEASE);
        }
    }
}

static PREP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Writes synthetic TLAS instance records (128 bytes each, BLAS 40-bit address
/// in the last 8 bytes, matching `Memory::named_in_upload`) into an `Upload`
/// block.
fn fill_tlas_upload(upload: &mut crate::upload::Upload, blas_addrs: &[u64]) -> u64 {
    let base = upload.start();
    // SAFETY: `upload` has at least `blas_addrs.len() * 128` bytes.
    unsafe {
        for (i, &addr) in blas_addrs.iter().enumerate() {
            std::ptr::write_unaligned(base.add(i * 128 + 120).cast::<u64>(), addr);
        }
    }
    base as u64
}

#[test]
fn synthetic_bvh_prep_fast_path_tlas_replace_and_sweep_on_gpu() {
    let _guard = PREP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = host() else { return };
    if !crate::shadow::cards() || !host.card_ready() {
        return;
    }
    eprintln!("adapter={} card_memory={}", host.adapter_label(), crate::shadow::cards());
    let heap = HostHeap::new(false, 64).expect("HostHeap");
    let pipe = TracePipeline::new(host);

    let (rec_buf, rec_ptr) = host.shared_buffer((THREADS as usize * RECORD_BYTES) as u64).expect("records");
    let (res_buf, res_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("results");
    heap.raw_uav_at(0, &rec_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(1, &res_buf, 0, (THREADS as usize * 16) as u64);

    // Two distinct VirtualAlloc reservations so coalescing across separate
    // allocations is also exercised.
    let region_a = TitleRegion::new(1 << 20);
    let region_b = TitleRegion::new(512 << 10);
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0);

    // 1. TLAS upload block (`named_in_upload`) discovers and brings BLAS pages
    //    not named directly in `addresses`.
    let blas0 = region_a.base;
    let blas1 = region_a.base + (64 << 10);
    let blas2 = region_b.base;
    let tri0 = make_tri([-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0]);
    let box0 = make_box32(
        [0x11, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 3.5, 1.0, 1.0, 4.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let tri1 = make_tri([-1.0, -1.0, 6.0], [1.0, -1.0, 6.0], [0.0, 1.0, 6.0]);
    let box1 = make_box32(
        [0x22, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 5.5, 1.0, 1.0, 6.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let tri2 = make_tri([-1.0, -1.0, 9.0], [1.0, -1.0, 9.0], [0.0, 1.0, 9.0]);
    let box2 = make_box32(
        [0x33, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 8.5, 1.0, 1.0, 9.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    write_unpacked_bvh(blas0, &box0, &tri0);
    write_unpacked_bvh(blas1, &box1, &tri1);
    write_unpacked_bvh(blas2, &box2, &tri2);

    let mut tlas_upload = crate::upload::Upload::new(64 << 10).expect("tlas upload");
    let tlas_addr = fill_tlas_upload(&mut tlas_upload, &[blas0 as u64, blas1 as u64, blas2 as u64]);

    write_ray_record(rec_ptr, 0, (blas0 + 64) as u64, 5, &ray);
    write_ray_record(rec_ptr, 1, (blas1 + 64) as u64, 5, &ray);
    write_ray_record(rec_ptr, 2, (blas2 + 64) as u64, 5, &ray);
    write_ray_record(rec_ptr, 3, (blas2 + 192) as u64, 0, &ray);

    crate::frame::test_advance();
    let list = host.new_list().expect("list tlas");
    let r = ready(&list, &heap, &[tlas_addr], &[], &[], Use::Builds { back: false }, None).expect("ready tlas");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    host.run_test_list_sync(&list);
    let out = read_results(res_ptr, 4);
    assert_eq!(out[0], bvh::intersect(&box0, 5, &ray, true), "TLAS blas0 mismatch");
    assert_eq!(out[1], bvh::intersect(&box1, 5, &ray, true), "TLAS blas1 mismatch");
    assert_eq!(out[2], bvh::intersect(&box2, 5, &ray, true), "TLAS blas2 box mismatch");
    assert_eq!(out[3], bvh::intersect(&tri2, 0, &ray, true), "TLAS blas2 tri mismatch");

    // 2. Dynamic move of blas1 referenced ONLY via TLAS upload block:
    //    moving box1 out of the ray must change GPU intersection to MISS.
    let box1_moved = make_box32(
        [0x22, bvh::NONE, bvh::NONE, bvh::NONE],
        [[5.0, 5.0, 5.5, 7.0, 7.0, 6.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    write_unpacked_bvh(blas1, &box1_moved, &tri1);
    crate::frame::test_advance();
    let list = host.new_list().expect("list tlas moved");
    let r = ready(&list, &heap, &[tlas_addr], &[], &[], Use::Builds { back: false }, None).expect("ready tlas moved");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    host.run_test_list_sync(&list);
    let out = read_results(res_ptr, 2);
    assert_eq!(out[0], bvh::intersect(&box0, 5, &ray, true), "static blas0 changed");
    assert_eq!(out[1], bvh::intersect(&box1_moved, 5, &ray, true), "moved blas1 must miss");

    // 2b. Sub-page (4 KiB piece > 0) dynamic update: place a box node in the
    //     second 4 KiB sub-page (+4096) of `blas0`, verify GPU sees initial
    //     hit and then sees the sub-page move (verifying `parts_written` mask).
    let subpage_box_hit = make_box32(
        [0x44, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 2.5, 1.0, 1.0, 3.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let subpage_box_miss = make_box32(
        [0x44, bvh::NONE, bvh::NONE, bvh::NONE],
        [[8.0, 8.0, 2.5, 9.0, 9.0, 3.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    // SAFETY: `blas0` has 64 KiB committed writable memory; offset 4096 is in piece 1.
    unsafe {
        std::ptr::copy_nonoverlapping(subpage_box_hit.as_ptr().cast::<u8>(), (blas0 + 4096) as *mut u8, 28 * 4);
    }
    write_ray_record(rec_ptr, 0, (blas0 + 4096) as u64, 5, &ray);
    crate::frame::test_advance();
    let list = host.new_list().expect("list subpage hit");
    let r = ready(&list, &heap, &[tlas_addr], &[], &[], Use::Builds { back: false }, None).expect("ready subpage hit");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    host.run_test_list_sync(&list);
    assert_eq!(read_results(res_ptr, 1)[0], bvh::intersect(&subpage_box_hit, 5, &ray, true), "sub-page piece 1 hit mismatch");

    // Now modify ONLY piece 1 (+4096) while leaving piece 0 untouched.
    unsafe {
        std::ptr::copy_nonoverlapping(subpage_box_miss.as_ptr().cast::<u8>(), (blas0 + 4096) as *mut u8, 28 * 4);
    }
    crate::frame::test_advance();
    let list = host.new_list().expect("list subpage miss");
    let r = ready(&list, &heap, &[tlas_addr], &[], &[], Use::Builds { back: false }, None).expect("ready subpage miss");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    host.run_test_list_sync(&list);
    assert_eq!(read_results(res_ptr, 1)[0], bvh::intersect(&subpage_box_miss, 5, &ray, true), "sub-page piece 1 move must miss");

    // 3. Packed BVH mid-flight header replacement between record and submit
    //    (within `AGAIN` frames): `land_batch` must detect the dirty header
    //    page and upload the new geometry on that exact submission!
    let packed_a = region_a.base + (256 << 10);
    let box_p1 = make_box32(
        [0x55, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 4.5, 1.0, 1.0, 5.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let tri_p1 = make_tri([-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]);
    write_packed_bvh(packed_a, 1023, &box_p1, &tri_p1); // 64 KiB
    write_ray_record(rec_ptr, 0, (packed_a + 64) as u64, 5, &ray);
    write_ray_record(rec_ptr, 1, (packed_a + 192) as u64, 0, &ray);

    crate::frame::test_advance();
    let list = host.new_list().expect("list p1");
    let r = ready(&list, &heap, &[packed_a as u64], &[], &[], Use::Builds { back: false }, None).expect("ready p1");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    host.run_test_list_sync(&list);
    let out = read_results(res_ptr, 2);
    assert_eq!(out[0], bvh::intersect(&box_p1, 5, &ray, true), "packed p1 box mismatch");
    assert_eq!(out[1], bvh::intersect(&tri_p1, 0, &ray, true), "packed p1 tri mismatch");

    // Next frame (within `AGAIN`): record builder list while old header is still
    // in place, THEN overwrite `packed_a` with a NEW header (2047 nodes = 128 KiB)
    // and different geometry BEFORE submitting!
    let box_p2 = make_box32(
        [0x66, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 7.5, 1.0, 1.0, 8.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let tri_p2 = make_tri([-1.0, -1.0, 8.0], [1.0, -1.0, 8.0], [0.0, 1.0, 8.0]);
    crate::frame::test_advance();
    let list = host.new_list().expect("list p2");
    let r = ready(&list, &heap, &[packed_a as u64], &[], &[], Use::Builds { back: false }, None).expect("ready p2");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    // Replace header & geometry AFTER record, BEFORE submit!
    write_packed_bvh(packed_a, 2047, &box_p2, &tri_p2);
    host.run_test_list_sync(&list);
    let out = read_results(res_ptr, 2);
    assert_eq!(out[0], bvh::intersect(&box_p2, 5, &ray, true), "mid-flight replaced packed box must update");
    assert_eq!(out[1], bvh::intersect(&tri_p2, 0, &ray, true), "mid-flight replaced packed tri must update");

    // 4. Geometry removal detected automatically by `look_again()` sweep:
    //    zero `packed_a`'s header and advance frames until `look_again()` sweeps
    //    it and calls `gone()`, clearing the directory so GPU trace misses.
    // SAFETY: clearing packed header in synthetic title memory.
    unsafe { std::ptr::write_unaligned(packed_a as *mut u64, 0) };
    let mut removed = false;
    for _ in 0..=SWEEP {
        crate::frame::test_advance();
        let list = host.new_list().expect("list sweep");
        let _ = ready(&list, &heap, &[], &[], &[], Use::Reads, None).expect("ready sweep");
        list.close();
        let still_tracked = MEMORY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|m| m.packed.contains_key(&(packed_a as u64)));
        if !still_tracked {
            removed = true;
            break;
        }
    }
    assert!(removed, "look_again() sweep must detect zeroed packed BVH header and reclaim it");

    crate::frame::test_advance();
    let list = host.new_list().expect("list after sweep");
    let r = ready(&list, &heap, &[], &[], &[], Use::Reads, None).expect("ready after sweep");
    pipe.dispatch(&list, &heap, &r, 0);
    list.close();
    host.run_test_list_sync(&list);
    let out = read_results(res_ptr, 2);
    assert_eq!(out[0], [bvh::NONE; 4], "swept-removed packed box must miss on GPU");
    assert_eq!(out[1], bvh::MISS.map(f32::to_bits), "swept-removed packed tri must miss on GPU");

    // 4b. Decommit-without-write test: `VirtualFree(..., MEM_DECOMMIT)` on a
    //     write-watched `VirtualAlloc2` region keeps the reservation intact
    //     without dirtying the page in `GetWriteWatch`. Verify `look_again()`
    //     still calls `there()` (`committed()`) and reclaims the decommitted
    //     packed BVH so GPU tracing misses cleanly.
    let decommit_addr = region_a.base + (512 << 10);
    write_packed_bvh(decommit_addr, 1023, &box_p1, &tri_p1);
    write_ray_record(rec_ptr, 0, (decommit_addr + 64) as u64, 5, &ray);
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("list decommit prime");
        let r = ready(&list, &heap, &[decommit_addr as u64], &[], &[], Use::Builds { back: false }, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0);
        list.close();
        host.run_test_list_sync(&list);
        assert_eq!(read_results(res_ptr, 1)[0], bvh::intersect(&box_p1, 5, &ray, true), "decommit prime hit mismatch");
    }
    // Advance one clean frame so `Packed::checked` is populated, then decommit
    // the page WITHOUT writing to it!
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("list decommit steady");
        let _ = ready(&list, &heap, &[decommit_addr as u64], &[], &[], Use::Builds { back: false }, None).unwrap();
        list.close();
        host.run_test_list_sync(&list);
    }
    {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        // SAFETY: `decommit_addr` is page-aligned inside `region_a` (`VirtualAlloc2`).
        let ok = unsafe { VirtualFree(decommit_addr as *mut _, PAGE as usize, MEM_DECOMMIT) };
        assert_ne!(ok, 0, "VirtualFree(MEM_DECOMMIT) must succeed");
    }
    let mut decommitted_reclaimed = false;
    for _ in 0..=SWEEP {
        crate::frame::test_advance();
        let list = host.new_list().expect("list decommit sweep");
        let _ = ready(&list, &heap, &[], &[], &[], Use::Reads, None).unwrap();
        list.close();
        let still_tracked = MEMORY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|m| m.packed.contains_key(&(decommit_addr as u64)));
        if !still_tracked {
            decommitted_reclaimed = true;
            break;
        }
    }
    assert!(
        decommitted_reclaimed,
        "look_again() must detect a packed BVH page decommitted without a CPU write"
    );
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("list after decommit");
        let r = ready(&list, &heap, &[], &[], &[], Use::Reads, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0);
        list.close();
        host.run_test_list_sync(&list);
        assert_eq!(read_results(res_ptr, 1)[0], [bvh::NONE; 4], "decommitted packed BVH must miss on GPU");
    }

    // 5. Regression checks for batched `defer_host` and coalesced write-watch:
    //    a multi-address builder dispatch must emit ONE batched `Step::Host`
    //    instead of one per address, and a sustained frame must coalesce
    //    `GetWriteWatch` calls.
    let batch_region = TitleRegion::new(16 * (128 << 10));
    let mut batch_addrs = Vec::with_capacity(16);
    for i in 0..16 {
        let addr = batch_region.base + i * (128 << 10);
        write_packed_bvh(addr, 2047, &box_p1, &tri_p1);
        batch_addrs.push(addr as u64);
    }
    // Prime batch once so `Packed::checked` is populated.
    crate::frame::test_advance();
    let list_prime = host.new_list().expect("list batch prime");
    let _ = ready(&list_prime, &heap, &batch_addrs, &[], &[], Use::Builds { back: false }, None).expect("ready batch prime");
    assert_eq!(
        list_prime.deferred_host_count(),
        1,
        "16 builder addresses must coalesce into a single batched Step::Host"
    );
    list_prime.close();
    host.run_test_list_sync(&list_prime);

    // Steady-state frame: verify batched `Step::Host` and coalesced `watch::writes`.
    crate::frame::test_advance();
    let _ = take_watch_asks();
    let list_steady = host.new_list().expect("list batch steady");
    let _ = ready(&list_steady, &heap, &batch_addrs, &[], &[], Use::Builds { back: false }, None).expect("ready batch steady");
    assert_eq!(list_steady.deferred_host_count(), 1, "steady-state builder batch must emit 1 Step::Host");
    list_steady.close();
    host.run_test_list_sync(&list_steady);
    #[cfg(feature = "diagnostics")]
    {
        let (watch_calls, _) = take_watch_asks();
        assert!(
            watch_calls <= 16,
            "coalesced steady-state frame should issue <= 16 GetWriteWatch calls, got {watch_calls}"
        );
    }
}

/// Sustained-load GPU benchmark measuring BVH CPU prep time, `GetWriteWatch`
/// kernel calls, deferred `Step::Host` closure count, total frame time, bytes
/// copied, and peak memory on separate static and dynamic scenes (including
/// TLAS upload-block scanning and `look_again` sweeps), verifying EVERY
/// measured frame bit-for-bit against `victory_shader::bvh::intersect`.
#[test]
#[ignore = "GPU benchmark (run with --release --features diagnostics -- --exact --ignored --nocapture --test-threads=1)"]
fn bench_bvh_prep_and_tlas_sustained() {
    let _guard = PREP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = host() else {
        panic!("no host GPU available");
    };
    assert!(crate::shadow::cards() && host.card_ready(), "card memory required");
    eprintln!("adapter={} card_memory={}", host.adapter_label(), crate::shadow::cards());

    let heap = HostHeap::new(false, 64).expect("HostHeap");
    let pipe = TracePipeline::new(host);
    let (rec_buf, rec_ptr) = host.shared_buffer((THREADS as usize * RECORD_BYTES) as u64).expect("records");
    let (res_buf, res_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("results");
    heap.raw_uav_at(0, &rec_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(1, &res_buf, 0, (THREADS as usize * 16) as u64);

    const STATIC_MESHES: usize = 128;
    const MESH_BYTES: usize = 128 << 10; // 128 KiB = 2 pages each (256 pages = 16 MiB)
    const BUILDER_ADDRS: usize = 64;
    const DYNAMIC_MESHES: usize = 64;
    const MOVING_PER_FRAME: usize = 16;
    const WARMUP: usize = 20;
    const FRAMES: usize = 120;

    let mem_before_mib = process_private_mib();
    let static_region = TitleRegion::new(STATIC_MESHES * MESH_BYTES);
    let dynamic_region = TitleRegion::new(DYNAMIC_MESHES * (64 << 10));
    let mut static_tlas_upload = crate::upload::Upload::new(64 << 10).expect("static tlas upload");
    let mut dyn_tlas_upload = crate::upload::Upload::new(64 << 10).expect("dyn tlas upload");

    let mut static_addrs = Vec::with_capacity(STATIC_MESHES);
    let mut static_boxes = Vec::with_capacity(STATIC_MESHES);
    let mut static_tris = Vec::with_capacity(STATIC_MESHES);
    for i in 0..STATIC_MESHES {
        let addr = static_region.base + i * MESH_BYTES;
        let z = 3.0 + (i as f32) * 0.25;
        let tri = make_tri([-1.0, -1.0, z], [1.0, -1.0, z], [0.0, 1.0, z]);
        let bx = make_box32(
            [0x100 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
            [[-1.0, -1.0, z - 0.1, 1.0, 1.0, z + 0.1], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
        );
        write_packed_bvh(addr, (MESH_BYTES as u32 / 64) - 1, &bx, &tri);
        static_addrs.push(addr as u64);
        static_boxes.push(bx);
        static_tris.push(tri);
    }
    // 256 TLAS instances referencing static BLAS addresses (2 instances per mesh).
    let mut static_tlas_refs = Vec::with_capacity(256);
    for i in 0..256 {
        static_tlas_refs.push(static_addrs[i % STATIC_MESHES]);
    }
    let static_tlas_addr = fill_tlas_upload(&mut static_tlas_upload, &static_tlas_refs);

    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0);
    for lane in 0..16 {
        let idx = lane * 8;
        write_ray_record(rec_ptr, lane * 2, static_addrs[idx] + 64, 5, &ray);
        write_ray_record(rec_ptr, lane * 2 + 1, static_addrs[idx] + 192, 0, &ray);
    }

    // Prime all 128 static packed BVHs + static TLAS upload.
    crate::frame::test_advance();
    {
        let mut prime = static_addrs.clone();
        prime.push(static_tlas_addr);
        let list = host.new_list().expect("prime static");
        let r = ready(&list, &heap, &prime, &[], &[], Use::Builds { back: false }, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0);
        list.close();
        host.run_test_list_sync(&list);
        let out = read_results(res_ptr, 32);
        for lane in 0..16 {
            let idx = lane * 8;
            assert_eq!(out[lane * 2], bvh::intersect(&static_boxes[idx], 5, &ray, true));
            assert_eq!(out[lane * 2 + 1], bvh::intersect(&static_tris[idx], 0, &ray, true));
        }
    }

    // --- Scene A: Static World Sustained (64 static packed builder addrs + TLAS upload + 4 RT lists) ---
    let mut static_prep_us = Vec::with_capacity(FRAMES);
    let mut static_frame_us = Vec::with_capacity(FRAMES);
    let mut static_host_steps = 0usize;
    let mut static_watch_calls = 0u64;
    let mut static_watch_ns = 0u64;
    let mut static_card_bytes = 0u64;
    let mut builder_batch = static_addrs[..BUILDER_ADDRS].to_vec();
    builder_batch.push(static_tlas_addr);

    let _ = take_watch_asks();
    for f in 0..(WARMUP + FRAMES) {
        crate::frame::test_advance();
        let cb0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
        let _ = take_watch_asks();
        let ft0 = Instant::now();
        let mut prep_us = 0u64;
        let mut host_steps = 0usize;

        let list_b = host.new_list().expect("static builder");
        let p0 = Instant::now();
        let _ = ready(&list_b, &heap, &builder_batch, &[], &[], Use::Builds { back: false }, None).unwrap();
        prep_us += p0.elapsed().as_micros() as u64;
        host_steps += list_b.deferred_host_count();
        list_b.close();
        host.run_test_list_sync(&list_b);

        for pass in 0..4 {
            let list_r = host.new_list().expect("static trace");
            let p0 = Instant::now();
            let r = ready(&list_r, &heap, &static_addrs[pass * 16..(pass + 1) * 16], &[], &[], Use::Reads, None).unwrap();
            prep_us += p0.elapsed().as_micros() as u64;
            host_steps += list_r.deferred_host_count();
            pipe.dispatch(&list_r, &heap, &r, 0);
            list_r.close();
            host.run_test_list_sync(&list_r);
        }
        let frame_us = ft0.elapsed().as_micros() as u64;
        let (w_calls, w_ns) = take_watch_asks();

        if f >= WARMUP {
            static_prep_us.push(prep_us);
            static_frame_us.push(frame_us);
            static_host_steps += host_steps;
            static_watch_calls += w_calls;
            static_watch_ns += w_ns;
            static_card_bytes += crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - cb0;
            // Verify EVERY measured frame against CPU oracle.
            let out = read_results(res_ptr, 32);
            for lane in 0..16 {
                let idx = lane * 8;
                assert_eq!(out[lane * 2], bvh::intersect(&static_boxes[idx], 5, &ray, true));
                assert_eq!(out[lane * 2 + 1], bvh::intersect(&static_tris[idx], 0, &ray, true));
            }
        }
    }

    // --- Scene B: Dynamic World + TLAS Refit Sustained (64 dynamic BLASes via TLAS upload, 16 moving/frame) ---
    let mut dyn_addrs = Vec::with_capacity(DYNAMIC_MESHES);
    for i in 0..DYNAMIC_MESHES {
        let addr = dynamic_region.base + i * (64 << 10);
        let z = 4.0 + (i as f32) * 0.5;
        let tri = make_tri([-1.0, -1.0, z], [1.0, -1.0, z], [0.0, 1.0, z]);
        let bx = make_box32(
            [0x200 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
            [[-1.0, -1.0, z - 0.1, 1.0, 1.0, z + 0.1], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
        );
        write_unpacked_bvh(addr, &bx, &tri);
        dyn_addrs.push(addr as u64);
    }
    let mut dyn_tlas_refs = Vec::with_capacity(256);
    for i in 0..256 {
        dyn_tlas_refs.push(dyn_addrs[i % DYNAMIC_MESHES]);
    }
    let dyn_tlas_addr = fill_tlas_upload(&mut dyn_tlas_upload, &dyn_tlas_refs);

    // Prime dynamic BLASes via TLAS upload.
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("prime dyn");
        let _ = ready(&list, &heap, &[dyn_tlas_addr], &[], &[], Use::Builds { back: false }, None).unwrap();
        list.close();
        host.run_test_list_sync(&list);
    }

    let mut dyn_prep_us = Vec::with_capacity(FRAMES);
    let mut dyn_frame_us = Vec::with_capacity(FRAMES);
    let mut dyn_host_steps = 0usize;
    let mut dyn_watch_calls = 0u64;
    let mut dyn_watch_ns = 0u64;
    let mut dyn_card_bytes = 0u64;

    for f in 0..(WARMUP + FRAMES) {
        crate::frame::test_advance();
        let shift = ((f % 32) as f32) * 0.02;
        let mut expected_boxes = Vec::with_capacity(MOVING_PER_FRAME);
        let mut expected_tris = Vec::with_capacity(MOVING_PER_FRAME);
        for i in 0..MOVING_PER_FRAME {
            let addr = dyn_addrs[i] as usize;
            let z = 4.0 + (i as f32) * 0.5;
            let tri = make_tri([-1.0 + shift, -1.0, z], [1.0 + shift, -1.0, z], [shift, 1.0, z]);
            let bx = make_box32(
                [0x200 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
                [
                    [-1.0 + shift, -1.0, z - 0.1, 1.0 + shift, 1.0, z + 0.1],
                    [f32::NAN; 6],
                    [f32::NAN; 6],
                    [f32::NAN; 6],
                ],
            );
            write_unpacked_bvh(addr, &bx, &tri);
            write_ray_record(rec_ptr, i * 2, (addr + 64) as u64, 5, &ray);
            write_ray_record(rec_ptr, i * 2 + 1, (addr + 192) as u64, 0, &ray);
            expected_boxes.push(bx);
            expected_tris.push(tri);
        }

        let cb0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
        let _ = take_watch_asks();
        let ft0 = Instant::now();

        let list = host.new_list().expect("dyn list");
        let p0 = Instant::now();
        // TLAS build scans upload block (256 instances -> 64 unique dynamic BLAS pages)
        let _ = ready(&list, &heap, &[dyn_tlas_addr], &[], &[], Use::Builds { back: false }, None).unwrap();
        let r = ready(&list, &heap, &dyn_addrs[..MOVING_PER_FRAME], &[], &[], Use::Reads, None).unwrap();
        let prep_us = p0.elapsed().as_micros() as u64;
        let host_steps = list.deferred_host_count();
        pipe.dispatch(&list, &heap, &r, 0);
        list.close();
        host.run_test_list_sync(&list);
        let frame_us = ft0.elapsed().as_micros() as u64;
        let (w_calls, w_ns) = take_watch_asks();

        if f >= WARMUP {
            dyn_prep_us.push(prep_us);
            dyn_frame_us.push(frame_us);
            dyn_host_steps += host_steps;
            dyn_watch_calls += w_calls;
            dyn_watch_ns += w_ns;
            dyn_card_bytes += crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - cb0;
            // Verify EVERY measured frame against CPU oracle.
            let out = read_results(res_ptr, 32);
            for i in 0..MOVING_PER_FRAME {
                assert_eq!(out[i * 2], bvh::intersect(&expected_boxes[i], 5, &ray, true));
                assert_eq!(out[i * 2 + 1], bvh::intersect(&expected_tris[i], 0, &ray, true));
            }
        }
    }

    let mem_after_mib = process_private_mib();
    let (sp50, sp95, spm) = percentile_summary(static_prep_us);
    let (sf50, sf95, sfm) = percentile_summary(static_frame_us);
    let (dp50, dp95, dpm) = percentile_summary(dyn_prep_us);
    let (df50, df95, dfm) = percentile_summary(dyn_frame_us);

    eprintln!("=== BVH PREP & TLAS BENCHMARK RESULTS ===");
    eprintln!(
        "static_sustained ({FRAMES}f): prep_us(p50/p95/mean)={sp50:.1}/{sp95}/{spm:.1} frame_us(p50/p95/mean)={sf50:.1}/{sf95}/{sfm:.1} host_steps/f={:.1} watch_calls/f={:.1} watch_us/f={:.1} card_bytes/f={:.1}",
        static_host_steps as f64 / FRAMES as f64,
        static_watch_calls as f64 / FRAMES as f64,
        static_watch_ns as f64 / 1000.0 / FRAMES as f64,
        static_card_bytes as f64 / FRAMES as f64,
    );
    eprintln!(
        "dynamic_sustained ({FRAMES}f): prep_us(p50/p95/mean)={dp50:.1}/{dp95}/{dpm:.1} frame_us(p50/p95/mean)={df50:.1}/{df95}/{dfm:.1} host_steps/f={:.1} watch_calls/f={:.1} watch_us/f={:.1} card_bytes/f={:.1}",
        dyn_host_steps as f64 / FRAMES as f64,
        dyn_watch_calls as f64 / FRAMES as f64,
        dyn_watch_ns as f64 / 1000.0 / FRAMES as f64,
        dyn_card_bytes as f64 / FRAMES as f64,
    );
    eprintln!(
        "memory: private_before_mib={mem_before_mib:.1} private_after_mib={mem_after_mib:.1} delta_mib={:.1}",
        mem_after_mib - mem_before_mib,
    );
}
