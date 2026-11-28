//! Synthetic GPU verification and sustained-load benchmarks for BVH
//! construction/update memory management (`memory.rs`), its buffers, and
//! residency. Uses purely synthetic RDNA 2 BVH nodes and the real translated
//! `image_bvh64_intersect_ray` shader over `SLOT_MEMORY` + `USER_MEMORY_BASE`
//! + `SLOT_FAULTS` on the host GPU (or WARP) without any game files.

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

/// Threads per compute group for our synthetic ray-tracing test shader.
const THREADS: u32 = 32;
/// Bytes per ray record: node ptr (2 dwords), extent, origin(3), dir(3), inv(3), active, pad(3).
const RECORD_BYTES: usize = 64;

/// Minimal valid empty DXBC container so `victory_shader::emit::emit` can lower
/// our synthetic `NativeProgram` without any external files.
fn empty_dxbc() -> Vec<u8> {
    let mut b = vec![0u8; 32];
    b[0..4].copy_from_slice(b"DXBC");
    b[0x14..0x16].copy_from_slice(&1u16.to_le_bytes());
    b[0x18..0x1c].copy_from_slice(&32u32.to_le_bytes());
    b
}

/// RDNA 2 compute program running `image_bvh64_intersect_ray` once per thread
/// using a raw-address descriptor (`Descriptor::Numbers(0, bvh::BOX_SORT)`),
/// reading ray records from table slot 0 (raw UAV) and writing `[u32; 4]`
/// intersection outputs to table slot 8 (raw UAV).
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
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k)); // v_mov_b32 v13+k, 1+k
    }
    // Raw-address BVH descriptor in s[24:27]: base=0, BOX_SORT, size=-1, RETURN_BARYCENTRICS|TYPE=8
    w.extend([
        0xBE98_03FF,
        0,           // s_mov_b32 s24, 0
        0xBE99_0B81, // s_brev_b32 s25, 1 (BOX_SORT = 1 << 31)
        0xBE9A_03C1, // s_mov_b32 s26, -1
        0xBE9B_03FF,
        0x8900_03FF, // s_mov_b32 s27, 0x890003ff
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
        // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[24:27]
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
        assert!(shader.root_srvs.contains(&SLOT_MEMORY), "shader must read SLOT_MEMORY");
        assert!(shader.root_uavs.contains(&SLOT_FAULTS), "shader must write SLOT_FAULTS");

        let dwords = shader.user_data_dwords();
        let mut params = Vec::new();
        let user_param = params.len() as u32;
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: USER_DATA_SPACE,
                    Num32BitValues: dwords,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        let memory_param = params.len() as u32;
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: u32::from(SLOT_MEMORY),
                    RegisterSpace: ROOT_CBV_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        let faults_param = params.len() as u32;
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: u32::from(SLOT_FAULTS),
                    RegisterSpace: ROOT_CBV_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });

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
            user_param,
            memory_param,
            faults_param,
        }
    }

    fn dispatch(
        &self,
        list: &HostList,
        heap: &HostHeap,
        ready: &Ready,
        table_base_slot: u32,
    ) {
        let dwords = self.shader.user_data_dwords() as usize;
        let mut values = vec![0u32; dwords];
        for u in &self.shader.user_data {
            values[u.index as usize] = match u.root {
                RootType::Other(USER_MEMORY_BASE) => ready.base,
                // Console descriptor table root values are byte offsets (32 B per descriptor).
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

/// A synthetic 128-byte RDNA 2 `box32` node (`kind == 5`).
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

/// A synthetic 64-byte RDNA 2 triangle node (`kind == 0`).
fn make_tri(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [u32; 28] {
    let mut w = [0u32; 28];
    for (vi, v) in [v0, v1, v2].iter().enumerate() {
        for c in 0..3 {
            w[vi * 3 + c] = v[c].to_bits();
        }
    }
    // Triangle 0 uses vertices [0, 1, 2]; id byte 0x09 => I=vertex 1, J=vertex 2, first=vertex 0.
    w[15] = 0x0000_0009;
    w
}

/// Writes a packed BVH header (`0x8102_0006`, `nodes`) and its root `box32` +
/// `tri` nodes into `base` (must be 256-byte aligned in reserved title memory).
fn write_packed_bvh(base: usize, nodes: u32, box_node: &[u32; 28], tri_node: &[u32; 28]) {
    let header: u64 = 0x8102_0006u64 | (u64::from(nodes) << 32);
    // SAFETY: caller allocated writable title memory covering `(nodes + 1) * 64` bytes.
    unsafe {
        std::ptr::write_unaligned(base as *mut u64, header);
        std::ptr::copy_nonoverlapping(box_node.as_ptr().cast::<u8>(), (base + 64) as *mut u8, 28 * 4);
        std::ptr::copy_nonoverlapping(tri_node.as_ptr().cast::<u8>(), (base + 192) as *mut u8, 16 * 4);
    }
}

/// Writes one thread's 64-byte ray record into `records_ptr`.
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
        1, // active
        0,
        0,
        0,
    ];
    // SAFETY: `lane < THREADS`, buffer has `THREADS * RECORD_BYTES`.
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
        // SAFETY: mapped results buffer with at least `count * 16` bytes.
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

/// Allocates write-watched title-like memory in `[4 GiB, 2^37)` for tests,
/// freed cleanly on drop.
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

/// Serializes GPU memory tests in one process since `MEMORY` is global.
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Lets the test change scene state while an earlier GPU submission is pending.
struct QueueBlock(ID3D12Fence);

impl QueueBlock {
    fn new(host: &Host) -> Self {
        // SAFETY: the new fence belongs to the queue's device; Drop signals it.
        unsafe {
            let block = Self(host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("test fence"));
            host.queue().Wait(&block.0, 1).expect("queue test fence");
            block
        }
    }
}

impl Drop for QueueBlock {
    fn drop(&mut self) {
        // SAFETY: the queue only waits on this live CPU-signaled fence.
        unsafe {
            let _ = self.0.Signal(1);
        }
    }
}

fn run_trace(
    host: &Host,
    heap: &HostHeap,
    pipe: &TracePipeline,
    table_base_slot: u32,
    results_ptr: *const u8,
    builder_addrs: &[u64],
    trace_addrs: &[u64],
    count: usize,
) -> (Vec<[u32; 4]>, usize, u64) {
    let list = host.new_list().expect("new_list");
    let t0 = Instant::now();
    if !builder_addrs.is_empty() {
        let _ = ready(
            &list,
            heap,
            builder_addrs,
            &[],
            &[],
            Use::Builds { back: false },
            None,
        )
        .expect("memory::ready builds");
    }
    let r = ready(&list, heap, trace_addrs, &[], &[], Use::Reads, None).expect("memory::ready reads");
    let prepare_us = t0.elapsed().as_micros() as u64;
    let refreshes = list.deferred_refresh_count();
    pipe.dispatch(&list, heap, &r, table_base_slot);
    list.close();
    host.run_test_list_sync(&list);
    (read_results(results_ptr, count), refreshes, prepare_us)
}

/// Set `VICTORY_MEMORY_CARD_MB=0` to exercise the shared-memory fallback.
#[test]
fn synthetic_bvh_static_dynamic_move_remove_and_lifetime_on_gpu() {
    let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = host() else { return };
    let card_memory = crate::shadow::cards() && host.card_ready();
    let adapter = victory_shader::emit::host::Host { device: host.device.clone() }.adapter_name();
    println!("adapter={},card_memory={card_memory}", adapter.as_deref().unwrap_or("unknown"));
    if !card_memory {
        return;
    }
    let heap = HostHeap::new(false, 64).expect("HostHeap");
    let pipe = TracePipeline::new(host);

    let (rec_buf, rec_ptr) = host.shared_buffer((THREADS as usize * RECORD_BYTES) as u64).expect("records");
    let (res_buf, res_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("results");
    heap.raw_uav_at(0, &rec_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(1, &res_buf, 0, (THREADS as usize * 16) as u64);

    // Allocate 2 MiB of write-watched title memory for synthetic BVHs.
    let region = TitleRegion::new(2 << 20);
    let bvh_a = region.base;
    let bvh_b = region.base + (256 << 10);
    let bvh_c = region.base + (512 << 10);

    // 1. Static packed BVH A: box32 + triangle at z = 5.0
    let tri_a = make_tri([-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]);
    let box_a = make_box32(
        [0x10, bvh::NONE, bvh::NONE, bvh::NONE],
        [
            [-1.0, -1.0, 4.5, 1.0, 1.0, 5.5],
            [f32::NAN; 6],
            [f32::NAN; 6],
            [f32::NAN; 6],
        ],
    );
    write_packed_bvh(bvh_a, 1023, &box_a, &tri_a); // 64 KiB packed BVH

    let ray_hit_a = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0);
    let ray_miss_a = Ray::new([5.0, 5.0, 0.0], [0.0, 0.0, 1.0], 20.0);

    write_ray_record(rec_ptr, 0, (bvh_a + 64) as u64, 5, &ray_hit_a);
    write_ray_record(rec_ptr, 1, (bvh_a + 192) as u64, 0, &ray_hit_a);
    write_ray_record(rec_ptr, 2, (bvh_a + 192) as u64, 0, &ray_miss_a);

    crate::frame::test_advance();
    let (out, _, _) = run_trace(
        host,
        &heap,
        &pipe,
        0,
        res_ptr,
        &[bvh_a as u64],
        &[bvh_a as u64],
        3,
    );
    assert_eq!(out[0], bvh::intersect(&box_a, 5, &ray_hit_a, true), "static box32 hit mismatch");
    assert_eq!(out[1], bvh::intersect(&tri_a, 0, &ray_hit_a, true), "static tri hit mismatch");
    assert_eq!(out[2], bvh::intersect(&tri_a, 0, &ray_miss_a, true), "static tri miss mismatch");

    // 2. Dynamic geometry move (partial 128-byte edit inside the same 64 KiB page):
    // Move triangle A from x in [-1,1] to x in [4,6], updating both box32 and triangle!
    let tri_a_moved = make_tri([4.0, -1.0, 5.0], [6.0, -1.0, 5.0], [5.0, 1.0, 5.0]);
    let box_a_moved = make_box32(
        [0x20, bvh::NONE, bvh::NONE, bvh::NONE],
        [
            [4.0, -1.0, 4.5, 6.0, 1.0, 5.5],
            [f32::NAN; 6],
            [f32::NAN; 6],
            [f32::NAN; 6],
        ],
    );
    // SAFETY: writing updated dynamic nodes in-place inside title memory.
    unsafe {
        std::ptr::copy_nonoverlapping(box_a_moved.as_ptr().cast::<u8>(), (bvh_a + 64) as *mut u8, 28 * 4);
        std::ptr::copy_nonoverlapping(tri_a_moved.as_ptr().cast::<u8>(), (bvh_a + 192) as *mut u8, 16 * 4);
    }
    let ray_hit_moved = Ray::new([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], 20.0);
    write_ray_record(rec_ptr, 0, (bvh_a + 64) as u64, 5, &ray_hit_a);
    write_ray_record(rec_ptr, 1, (bvh_a + 192) as u64, 0, &ray_hit_a);
    write_ray_record(rec_ptr, 2, (bvh_a + 64) as u64, 5, &ray_hit_moved);
    write_ray_record(rec_ptr, 3, (bvh_a + 192) as u64, 0, &ray_hit_moved);

    crate::frame::test_advance();
    let (out, _, _) = run_trace(host, &heap, &pipe, 0, res_ptr, &[], &[bvh_a as u64], 4);
    assert_eq!(out[0], bvh::intersect(&box_a_moved, 5, &ray_hit_a, true), "old ray must now miss moved box");
    assert_eq!(out[1], bvh::intersect(&tri_a_moved, 0, &ray_hit_a, true), "old ray must now miss moved tri");
    assert_eq!(out[2], bvh::intersect(&box_a_moved, 5, &ray_hit_moved, true), "new ray must hit moved box");
    assert_eq!(out[3], bvh::intersect(&tri_a_moved, 0, &ray_hit_moved, true), "new ray must hit moved tri");

    // 3. Late-streamed packed BVH B: header/body arrives BETWEEN list record and submission!
    // Record list BEFORE writing packed BVH B, then write BVH B, then submit.
    let tri_b = make_tri([-2.0, -2.0, 3.0], [2.0, -2.0, 3.0], [0.0, 2.0, 3.0]);
    let box_b = make_box32(
        [0x30, bvh::NONE, bvh::NONE, bvh::NONE],
        [
            [-2.0, -2.0, 2.5, 2.0, 2.0, 3.5],
            [f32::NAN; 6],
            [f32::NAN; 6],
            [f32::NAN; 6],
        ],
    );
    write_ray_record(rec_ptr, 0, (bvh_b + 64) as u64, 5, &ray_hit_a);
    write_ray_record(rec_ptr, 1, (bvh_b + 192) as u64, 0, &ray_hit_a);
    crate::frame::test_advance();
    let list_b = host.new_list().expect("new_list b");
    let _ = ready(&list_b, &heap, &[bvh_b as u64], &[], &[], Use::Builds { back: false }, None).expect("ready b builds");
    let rb = ready(&list_b, &heap, &[bvh_b as u64], &[], &[], Use::Reads, None).expect("ready b reads");
    pipe.dispatch(&list_b, &heap, &rb, 0);
    list_b.close();
    // Simulate async streaming landing right before submission!
    write_packed_bvh(bvh_b, 1023, &box_b, &tri_b);
    host.run_test_list_sync(&list_b);
    let out_b = read_results(res_ptr, 2);
    assert_eq!(out_b[0], bvh::intersect(&box_b, 5, &ray_hit_a, true), "late-streamed box32 mismatch");
    assert_eq!(out_b[1], bvh::intersect(&tri_b, 0, &ray_hit_a, true), "late-streamed tri mismatch");

    // 4. Geometry removal: zero BVH B's header so `gone()` reclaims it, then trace BVH B -> must MISS!
    // SAFETY: clearing packed header in synthetic title memory.
    unsafe { std::ptr::write_unaligned(bvh_b as *mut u64, 0) };
    {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = guard.as_mut() {
            m.gone(bvh_b as u64);
        }
    }
    crate::frame::test_advance();
    let (out_removed, _, _) = run_trace(host, &heap, &pipe, 0, res_ptr, &[], &[], 2);
    assert_eq!(out_removed[0], [bvh::NONE; 4], "removed box32 must miss (directory zeroed)");
    assert_eq!(out_removed[1], bvh::MISS.map(f32::to_bits), "removed tri must miss (directory zeroed)");

    // 5. Pipelined in-flight GPU lifetime test:
    // Record & submit BVH C on Flight 1 (do NOT settle yet!), immediately remove BVH C and stream
    // a different geometry at BVH B, submit Flight 2, then settle and verify BOTH flights got their
    // own exact intersections without corrupting each other.
    let tri_c = make_tri([-1.0, -1.0, 8.0], [1.0, -1.0, 8.0], [0.0, 1.0, 8.0]);
    let box_c = make_box32(
        [0x40, bvh::NONE, bvh::NONE, bvh::NONE],
        [
            [-1.0, -1.0, 7.5, 1.0, 1.0, 8.5],
            [f32::NAN; 6],
            [f32::NAN; 6],
            [f32::NAN; 6],
        ],
    );
    write_packed_bvh(bvh_c, 1023, &box_c, &tri_c);
    let (rec2_buf, rec2_ptr) = host.shared_buffer((THREADS as usize * RECORD_BYTES) as u64).expect("records2");
    let (res2_buf, res2_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("results2");
    heap.raw_uav_at(2, &rec2_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(3, &res2_buf, 0, (THREADS as usize * 16) as u64);

    // Prime BVH C first so it's resident.
    write_ray_record(rec_ptr, 0, (bvh_c + 64) as u64, 5, &ray_hit_a);
    write_ray_record(rec_ptr, 1, (bvh_c + 192) as u64, 0, &ray_hit_a);
    crate::frame::test_advance();
    let (out_c0, _, _) = run_trace(host, &heap, &pipe, 0, res_ptr, &[bvh_c as u64], &[bvh_c as u64], 2);
    assert_eq!(out_c0[0], bvh::intersect(&box_c, 5, &ray_hit_a, true));

    // Hold Flight 1 on the GPU queue until the CPU has removed BVH C.
    unsafe { std::ptr::write_bytes(res_ptr, 0xAA, 32) };
    crate::frame::test_advance();
    let list_f1 = host.new_list().expect("list_f1");
    let rf1 = ready(&list_f1, &heap, &[bvh_c as u64], &[], &[], Use::Reads, None).expect("ready f1");
    pipe.dispatch(&list_f1, &heap, &rf1, 0);
    list_f1.close();
    let queue_block = QueueBlock::new(host);
    let (_, fence_f1) = host.run_test_list_async(&list_f1);

    // Retire C while Flight 1 still has to read its old directory entry.
    unsafe { std::ptr::write_unaligned(bvh_c as *mut u64, 0) };
    let retired_while_in_flight = {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let m = guard.as_mut().unwrap();
        let before_ret = m.retiring.len();
        m.gone(bvh_c as u64);
        m.retiring.len().saturating_sub(before_ret)
    };
    assert!(
        retired_while_in_flight >= 1,
        "freed arena page must stay in fence-gated retiring queue while Flight 1 is in flight"
    );
    drop(queue_block);
    write_packed_bvh(bvh_b, 1023, &box_b, &tri_b);
    write_ray_record(rec2_ptr, 0, (bvh_b + 64) as u64, 5, &ray_hit_a);
    write_ray_record(rec2_ptr, 1, (bvh_b + 192) as u64, 0, &ray_hit_a);
    crate::frame::test_advance();
    let list_f2 = host.new_list().expect("list_f2");
    let _ = ready(&list_f2, &heap, &[bvh_b as u64], &[], &[], Use::Builds { back: false }, None).expect("ready f2 b");
    let rf2 = ready(&list_f2, &heap, &[bvh_b as u64], &[], &[], Use::Reads, None).expect("ready f2 r");
    pipe.dispatch(&list_f2, &heap, &rf2, 2);
    list_f2.close();
    let (_, fence_f2) = host.run_test_list_async(&list_f2);
    host.wait_test_fence(fence_f1.max(fence_f2));

    let out_f1 = read_results(res_ptr, 2);
    let out_f2 = read_results(res2_ptr, 2);
    assert_eq!(out_f1[0], bvh::intersect(&box_c, 5, &ray_hit_a, true), "in-flight Flight 1 box32 corrupted");
    assert_eq!(out_f1[1], bvh::intersect(&tri_c, 0, &ray_hit_a, true), "in-flight Flight 1 tri corrupted");
    assert_eq!(out_f2[0], bvh::intersect(&box_b, 5, &ray_hit_a, true), "Flight 2 box32 mismatch");
    assert_eq!(out_f2[1], bvh::intersect(&tri_b, 0, &ray_hit_a, true), "Flight 2 tri mismatch");

    // Once both flights have completed, `reclaim_retired` must drain `retiring` back into `free`.
    {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let m = guard.as_mut().unwrap();
        let free_before = m.free.len();
        m.reclaim_retired();
        assert!(m.retiring.is_empty(), "completed flights must drain retiring queue");
        assert!(m.free.len() >= free_before + retired_while_in_flight, "retired pages must return to free list");
    }

    // Clean up packed entries so we leave global MEMORY clean.
    unsafe {
        std::ptr::write_unaligned(bvh_a as *mut u64, 0);
        std::ptr::write_unaligned(bvh_b as *mut u64, 0);
    }
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = guard.as_mut() {
        m.gone(bvh_a as u64);
        m.gone(bvh_b as u64);
    }
}

fn stats_summary(mut samples: Vec<u64>) -> (u64, u64, f64) {
    samples.sort_unstable();
    let len = samples.len();
    let median = samples[(len - 1) / 2].midpoint(samples[len / 2]);
    let p95 = samples[(len * 95).div_ceil(100) - 1];
    let mean = samples.iter().sum::<u64>() as f64 / len as f64;
    (median, p95, mean)
}

#[test]
fn bvh_benchmark_summary_uses_median_and_nearest_rank_p95() {
    assert_eq!(stats_summary((1..=120).rev().collect()), (60, 114, 60.5));
    assert_eq!(stats_summary(vec![23]), (23, 23, 23.0));
}

/// Sustained-load GPU benchmark comparing static and dynamic BVH scenes
/// (build/update time, bytes copied, Step::Refresh count, peak memory) with
/// bit-for-bit ray intersection verification on every sampled frame.
#[test]
#[ignore = "GPU benchmark (run with --release -- --ignored --nocapture --test-threads=1)"]
fn bench_bvh_static_and_dynamic_sustained() {
    let _guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = host() else {
        panic!("no host GPU available");
    };
    assert!(crate::shadow::cards() && host.card_ready(), "card memory required");
    let adapter = victory_shader::emit::host::Host { device: host.device.clone() }
        .adapter_name().expect("DXGI adapter");
    println!("adapter={adapter},card_memory=true,p95=nearest_rank");
    let heap = HostHeap::new(false, 64).expect("HostHeap");
    let pipe = TracePipeline::new(host);

    let (rec_buf, rec_ptr) = host.shared_buffer((THREADS as usize * RECORD_BYTES) as u64).expect("records");
    let (res_buf, res_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("results");
    heap.raw_uav_at(0, &rec_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(1, &res_buf, 0, (THREADS as usize * 16) as u64);

    // 16 MiB synthetic static world (128 packed BVH meshes x 128 KiB each = 256 pages)
    // + 16 MiB dynamic world (256 pages, 16 dynamic meshes moving every frame)
    // + 4 MiB churn pool (spawn/remove packed BVH every 8 frames).
    const STATIC_MESHES: usize = 128;
    const MESH_BYTES: usize = 128 << 10; // 128 KiB = 2 pages each
    const DYNAMIC_MESHES: usize = 16;
    const WARMUP: usize = 20;
    const FRAMES: usize = 120;

    let mem_before_mib = process_private_mib();
    let static_region = TitleRegion::new(STATIC_MESHES * MESH_BYTES);
    let dynamic_region = TitleRegion::new(DYNAMIC_MESHES * (64 << 10));
    let churn_region = TitleRegion::new(4 << 20);

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

    let ray_center = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0);
    let mut static_hits = Vec::with_capacity(32);
    for lane in 0..16 {
        let idx = lane * 8;
        let addr = static_addrs[idx];
        write_ray_record(rec_ptr, lane * 2, addr + 64, 5, &ray_center);
        write_ray_record(rec_ptr, lane * 2 + 1, addr + 192, 0, &ray_center);
        static_hits.push(bvh::intersect(&static_boxes[idx], 5, &ray_center, true));
        static_hits.push(bvh::intersect(&static_tris[idx], 0, &ray_center, true));
    }

    // --- Initial Static Stream / Build ---
    let card_bytes_0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
    let card_runs_0 = crate::shadow::CARD_STATS[1].load(Ordering::Relaxed);
    let t_stream = Instant::now();
    crate::frame::test_advance();
    let (init_out, init_refreshes, _) = run_trace(
        host,
        &heap,
        &pipe,
        0,
        res_ptr,
        &static_addrs,
        &static_addrs[..16],
        32,
    );
    let stream_us = t_stream.elapsed().as_micros() as u64;
    let stream_card_bytes = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - card_bytes_0;
    let stream_card_runs = crate::shadow::CARD_STATS[1].load(Ordering::Relaxed) - card_runs_0;
    assert_eq!(init_out, static_hits, "initial static stream");

    // --- Scene A: Static Sustained Load (120 frames, 4 RT lists + 1 builder list per frame) ---
    let mut static_prep_us = Vec::with_capacity(FRAMES);
    let mut static_frame_us = Vec::with_capacity(FRAMES);
    let mut static_refreshes_total = 0usize;
    let mut static_card_bytes = 0u64;
    let mut static_card_runs = 0u64;

    for f in 0..(WARMUP + FRAMES) {
        crate::frame::test_advance();
        let cb0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
        let cr0 = crate::shadow::CARD_STATS[1].load(Ordering::Relaxed);
        let ft0 = Instant::now();

        let mut frame_prep = 0u64;
        let mut frame_ref = 0usize;
        // 1 builder list touching 32 static addresses + 4 RT traversal lists (as in a real frame)
        let list_b = host.new_list().expect("static builder list");
        let p0 = Instant::now();
        let _ = ready(&list_b, &heap, &static_addrs[..32], &[], &[], Use::Builds { back: false }, None).unwrap();
        frame_prep += p0.elapsed().as_micros() as u64;
        frame_ref += list_b.deferred_refresh_count();
        list_b.close();
        host.run_test_list_sync(&list_b);

        for pass in 0..4 {
            let list_r = host.new_list().expect("static trace list");
            let p0 = Instant::now();
            let r = ready(&list_r, &heap, &static_addrs[pass * 8..(pass + 1) * 8], &[], &[], Use::Reads, None).unwrap();
            frame_prep += p0.elapsed().as_micros() as u64;
            frame_ref += list_r.deferred_refresh_count();
            pipe.dispatch(&list_r, &heap, &r, 0);
            list_r.close();
            host.run_test_list_sync(&list_r);
        }
        let frame_us = ft0.elapsed().as_micros() as u64;

        if f >= WARMUP {
            static_prep_us.push(frame_prep);
            static_frame_us.push(frame_us);
            static_refreshes_total += frame_ref;
            static_card_bytes += crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - cb0;
            static_card_runs += crate::shadow::CARD_STATS[1].load(Ordering::Relaxed) - cr0;
            assert_eq!(read_results(res_ptr, 32), static_hits, "static frame {f}");
        }
    }

    // --- Scene B: Dynamic Sustained Load (16 moving meshes x 128 B each + packed churn every 8 frames) ---
    let mut dyn_addrs = Vec::with_capacity(DYNAMIC_MESHES);
    for i in 0..DYNAMIC_MESHES {
        let addr = dynamic_region.base + i * (64 << 10);
        let tri = make_tri([-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0]);
        let bx = make_box32(
            [0x200 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
            [[-1.0, -1.0, 4.5, 1.0, 1.0, 5.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
        );
        write_packed_bvh(addr, 1023, &bx, &tri);
        dyn_addrs.push(addr as u64);
    }
    // Prime dynamic meshes
    crate::frame::test_advance();
    let _ = run_trace(host, &heap, &pipe, 0, res_ptr, &dyn_addrs, &dyn_addrs, 1);

    let mut dyn_prep_us = Vec::with_capacity(FRAMES);
    let mut dyn_frame_us = Vec::with_capacity(FRAMES);
    let mut move_frame_us = Vec::new();
    let mut stream_frame_us = Vec::new();
    let mut dyn_refreshes_total = 0usize;
    let mut dyn_card_bytes = 0u64;
    let mut dyn_card_runs = 0u64;
    let mut active_churn: Option<usize> = None;

    for f in 0..(WARMUP + FRAMES) {
        crate::frame::test_advance();
        let shift = ((f % 32) as f32) * 0.02;
        let mut expected_boxes = Vec::with_capacity(DYNAMIC_MESHES);
        let mut expected_tris = Vec::with_capacity(DYNAMIC_MESHES);
        // Move all 16 dynamic meshes by `shift` in X (small partial update per 64 KiB page)
        for i in 0..DYNAMIC_MESHES {
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
            unsafe {
                std::ptr::copy_nonoverlapping(bx.as_ptr().cast::<u8>(), (addr + 64) as *mut u8, 28 * 4);
                std::ptr::copy_nonoverlapping(tri.as_ptr().cast::<u8>(), (addr + 192) as *mut u8, 16 * 4);
            }
            write_ray_record(rec_ptr, i * 2, (addr + 64) as u64, 5, &ray_center);
            write_ray_record(rec_ptr, i * 2 + 1, (addr + 192) as u64, 0, &ray_center);
            expected_boxes.push(bx);
            expected_tris.push(tri);
        }

        // Every 8 frames, retire old churn packed BVH and stream a new 256 KiB packed BVH.
        let mut build_addrs = dyn_addrs.clone();
        let stream = f % 8 == 0;
        if stream {
            if let Some(old) = active_churn.take() {
                unsafe { std::ptr::write_unaligned(old as *mut u64, 0) };
                if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                    m.gone(old as u64);
                }
            }
            let slot = (f / 8) % 16;
            let caddr = churn_region.base + slot * (256 << 10);
            write_packed_bvh(caddr, (256 * 1024 / 64) - 1, &expected_boxes[0], &expected_tris[0]);
            active_churn = Some(caddr);
            build_addrs.push(caddr as u64);
        }

        let cb0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
        let cr0 = crate::shadow::CARD_STATS[1].load(Ordering::Relaxed);
        let ft0 = Instant::now();

        let list = host.new_list().expect("dyn list");
        let p0 = Instant::now();
        let _ = ready(&list, &heap, &build_addrs, &[], &[], Use::Builds { back: false }, None).unwrap();
        let r = ready(&list, &heap, &dyn_addrs, &[], &[], Use::Reads, None).unwrap();
        let prep = p0.elapsed().as_micros() as u64;
        let refs = list.deferred_refresh_count();
        pipe.dispatch(&list, &heap, &r, 0);
        list.close();
        host.run_test_list_sync(&list);
        let frame_us = ft0.elapsed().as_micros() as u64;

        if f >= WARMUP {
            dyn_prep_us.push(prep);
            dyn_frame_us.push(frame_us);
            if stream {
                stream_frame_us.push(frame_us);
            } else {
                move_frame_us.push(frame_us);
            }
            dyn_refreshes_total += refs;
            dyn_card_bytes += crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - cb0;
            dyn_card_runs += crate::shadow::CARD_STATS[1].load(Ordering::Relaxed) - cr0;
            let out = read_results(res_ptr, 32);
            for i in 0..DYNAMIC_MESHES {
                assert_eq!(
                    out[i * 2],
                    bvh::intersect(&expected_boxes[i], 5, &ray_center, true),
                    "dynamic frame {f}, box {i}"
                );
                assert_eq!(
                    out[i * 2 + 1],
                    bvh::intersect(&expected_tris[i], 0, &ray_center, true),
                    "dynamic frame {f}, triangle {i}"
                );
            }
        }
    }

    let mem_after_mib = process_private_mib();
    let [_, vram_bvh_bytes] = crate::shadow::vram_breakdown();
    let (arenas_count, pages_held, free_pages) = {
        let g = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let m = g.as_ref().unwrap();
        (m.arenas.len(), m.pages.len(), m.free.len())
    };

    let (sp_med, sp_p95, sp_mean) = stats_summary(static_prep_us);
    let (sf_med, sf_p95, sf_mean) = stats_summary(static_frame_us);
    let (dp_med, dp_p95, dp_mean) = stats_summary(dyn_prep_us);
    let (df_med, df_p95, df_mean) = stats_summary(dyn_frame_us);

    eprintln!("=== BVH BENCHMARK RESULTS ===");
    eprintln!(
        "initial_static_stream: time_us={stream_us} card_upload_bytes={stream_card_bytes} card_runs={stream_card_runs} refreshes={init_refreshes}"
    );
    eprintln!(
        "static_sustained ({FRAMES}f): prep_us(med/p95/mean)={sp_med}/{sp_p95}/{sp_mean:.1} frame_us(med/p95/mean)={sf_med}/{sf_p95}/{sf_mean:.1} refreshes/f={:.1} card_bytes/f={:.1} card_runs/f={:.1}",
        static_refreshes_total as f64 / FRAMES as f64,
        static_card_bytes as f64 / FRAMES as f64,
        static_card_runs as f64 / FRAMES as f64,
    );
    eprintln!(
        "dynamic_sustained ({FRAMES}f): prep_us(med/p95/mean)={dp_med}/{dp_p95}/{dp_mean:.1} frame_us(med/p95/mean)={df_med}/{df_p95}/{df_mean:.1} refreshes/f={:.1} card_bytes/f={:.1} card_runs/f={:.1}",
        dyn_refreshes_total as f64 / FRAMES as f64,
        dyn_card_bytes as f64 / FRAMES as f64,
        dyn_card_runs as f64 / FRAMES as f64,
    );
    let (move_frames, stream_frames) = (move_frame_us.len(), stream_frame_us.len());
    let (move_median, move_p95, _) = stats_summary(move_frame_us);
    let (stream_median, stream_p95, _) = stats_summary(stream_frame_us);
    eprintln!(
        "dynamic_frame_classes: move_frames={move_frames} move_us(med/p95)={move_median}/{move_p95} stream_frames={stream_frames} replace_us(med/p95)={stream_median}/{stream_p95}"
    );
    eprintln!(
        "memory: private_before_mib={mem_before_mib:.1} private_after_mib={mem_after_mib:.1} delta_mib={:.1} vram_bvh_mib={:.1} arenas={arenas_count} pages_held={pages_held} free_pages={free_pages}",
        mem_after_mib - mem_before_mib,
        vram_bvh_bytes as f64 / (1024.0 * 1024.0),
    );
}
