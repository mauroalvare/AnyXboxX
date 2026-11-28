//! Synthetic GPU verification and sustained-load benchmarks for BVH arena
//! locality, contiguous multi-page placement, and lowest-arena-first packing
//! after streaming churn.
//!
//! Uses purely synthetic RDNA 2 BVH nodes and the real translated
//! `image_bvh64_intersect_ray` compute shader over `SLOT_MEMORY` +
//! `USER_MEMORY_BASE` + `SLOT_FAULTS` with hardware D3D12 timestamp queries,
//! verifying every measured frame bit-for-bit against `victory_shader::bvh::intersect`.

use std::collections::HashSet;
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

/// Builds an RDNA 2 compute shader that runs `steps` iterations of
/// `image_bvh64_intersect_ray` per thread over `records[(gid + iter * stride) & mask]`,
/// reading BVH nodes through the real `SLOT_MEMORY` directory and `USER_MEMORY_BASE`.
///
/// When `steps == 1`, stores exact `v[13:16]` for `gid & mask`.
/// When `steps > 1`, accumulates `v19 += v13` across every step so all `steps`
/// iterations remain live and stores `[v13, v14, v15, v19]` for `gid & mask`.
fn trace_program_words(steps: u32, stride: u32, record_mask: u32) -> Vec<u32> {
    let mut w = vec![
        0xBE86_0305, // s_mov_b32 s6, s5 (save group_id_x)
        0x8F06_8506, // s_lshl_b32 s6, s6, 5 (group_id_x * 32)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0  (records UAV)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (results UAV)
        // v22 = (group_id_x * 32 + v0) & record_mask
        (0x25 << 25) | (22 << 17) | (0 << 9) | 6, // v_add_nc_u32 v22, s6, v0
        (0x1B << 25) | (22 << 17) | (22 << 9) | 0xFF,
        record_mask, // v_and_b32 v22, record_mask, v22
        // Load constant ray fields (extent, origin, dir, inv_dir, active) once from records[v22].
        (0x1A << 25) | (20 << 17) | (22 << 9) | 0x86, // v_lshlrev_b32 v20, 6, v22
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
        0x7E26_0280, // v_mov_b32 v19, 0 (checksum accumulator)
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
    ]);

    if steps == 1 {
        w.extend([
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
            0xF19C_9F07,
            0x0000_0D01 | (24 >> 2) << 16,
            0x0504_0302,
            0x0908_0706,
            0x000C_0B0A,
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
        ]);
    } else {
        w.extend([
            0xBE9C_0380, // s_mov_b32 s28, 0 (iter)
            0xBE9D_0380, // s_mov_b32 s29, 0 (offset = iter * stride)
        ]);
        let loop_pc = w.len();
        w.extend([
            // v23 = (v22 + s29) & record_mask
            (0x25 << 25) | (23 << 17) | (22 << 9) | 29, // v_add_nc_u32 v23, s29, v22
            (0x1B << 25) | (23 << 17) | (23 << 9) | 0xFF,
            record_mask, // v_and_b32 v23, record_mask, v23
            (0x1A << 25) | (20 << 17) | (23 << 9) | 0x86, // v_lshlrev_b32 v20, 6, v23
            0xE034_1000,
            0x8002_0114, // buffer_load_dwordx2 v[1:2], v20, s[8:11], 0 offen
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
            0xF19C_9F07,
            0x0000_0D01 | (24 >> 2) << 16,
            0x0504_0302,
            0x0908_0706,
            0x000C_0B0A,
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            // v19 = v19 + v13 (keeps every iteration live & oracle-checkable)
            (0x25 << 25) | (19 << 17) | (19 << 9) | (256 + 13), // v_add_nc_u32 v19, v13, v19
            // s29 += stride; s28 += 1; if s28 < steps goto loop_pc
            (0b10 << 30) | (29 << 16) | (0xFF << 8) | 29,
            stride,      // s_add_u32 s29, s29, stride
            0x801C_811C, // s_add_u32 s28, s28, 1
            0xBF0A_FF1C,
            steps, // s_cmp_lt_u32 s28, steps
        ]);
        let branch_pc = w.len();
        let rel = (loop_pc as isize) - ((branch_pc + 1) as isize);
        w.push(0xBF85_0000 | (rel as i16 as u16 as u32)); // s_cbranch_scc1 loop_pc
        w.push((0x7E00_0200) | (16 << 17) | (256 + 19)); // v_mov_b32 v16, v19
    }

    w.extend([
        (0x1A << 25) | (21 << 17) | (22 << 9) | 0x84, // v_lshlrev_b32 v21, 4, v22
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
    fn new(host: &Host, steps: u32, stride: u32, record_mask: u32) -> Self {
        crate::descriptor::host::probe_view_limit();
        let words = trace_program_words(steps, stride, record_mask);
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
            vgprs: 24,
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
                // 5 user SGPRs (s0..s4) + TGID_X_EN (bit 7 -> group_id_x in s5)
                rsrc2: (5 << 1) | (1 << 7),
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

    fn dispatch(
        &self,
        list: &HostList,
        heap: &HostHeap,
        ready: &Ready,
        table_base_slot: u32,
        groups: u32,
        timer: Option<&GpuTimer>,
    ) {
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
            if let Some(t) = timer {
                list.list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            }
            list.list.Dispatch(groups, 1, 1);
            if let Some(t) = timer {
                list.list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list.list.ResolveQueryData(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &t.readback, 0);
            }
        }
    }
}

struct GpuTimer {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ptr: *const u64,
    freq: u64,
}

impl GpuTimer {
    fn new(host: &Host) -> Self {
        let desc = D3D12_QUERY_HEAP_DESC {
            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            Count: 2,
            NodeMask: 0,
        };
        let mut heap: Option<ID3D12QueryHeap> = None;
        // SAFETY: valid D3D12 timestamp query heap creation.
        unsafe {
            host.device.CreateQueryHeap(&desc, &mut heap).expect("CreateQueryHeap");
        }
        let (readback, ptr) = host.shared_buffer(16).expect("timer readback");
        Self {
            heap: heap.expect("query heap"),
            readback,
            ptr: ptr.cast::<u64>(),
            freq: host.timestamp_frequency().max(1),
        }
    }

    fn elapsed_us(&self) -> f64 {
        // SAFETY: 16-byte mapped readback buffer written by ResolveQueryData before sync wait.
        let (t0, t1) = unsafe { (self.ptr.read_volatile(), self.ptr.add(1).read_volatile()) };
        t1.saturating_sub(t0) as f64 * 1_000_000.0 / self.freq as f64
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

/// Writes a 2-page (128 KiB) packed BVH with a box32 node on page 0 (+64)
/// and a second box32 (+65536) + triangle (+65728) on page 1.
fn write_packed_bvh_2page(
    base: usize,
    nodes: u32,
    box_p0: &[u32; 28],
    box_p1: &[u32; 28],
    tri_p1: &[u32; 28],
) {
    let header: u64 = 0x8102_0006u64 | (u64::from(nodes) << 32);
    // SAFETY: caller allocated writable memory covering `(nodes + 1) * 64` bytes (>= 128 KiB).
    unsafe {
        std::ptr::write_unaligned(base as *mut u64, header);
        std::ptr::copy_nonoverlapping(box_p0.as_ptr().cast::<u8>(), (base + 64) as *mut u8, 28 * 4);
        std::ptr::copy_nonoverlapping(box_p1.as_ptr().cast::<u8>(), (base + 65536) as *mut u8, 28 * 4);
        std::ptr::copy_nonoverlapping(tri_p1.as_ptr().cast::<u8>(), (base + 65536 + 192) as *mut u8, 16 * 4);
    }
}

fn write_packed_header_only(base: usize, nodes: u32) {
    let header: u64 = 0x8102_0006u64 | (u64::from(nodes) << 32);
    // SAFETY: caller allocated writable memory at `base`.
    unsafe {
        std::ptr::write_unaligned(base as *mut u64, header);
    }
}

fn write_unpacked_bvh(base: usize, box_node: &[u32; 28], tri_node: &[u32; 28]) {
    let header: u64 = 0x8002_0006u64 | (1023u64 << 32);
    // SAFETY: caller allocated writable memory covering 64 KiB.
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
    // SAFETY: caller sized records buffer for `lane + 1` records.
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

fn percentile_summary_f64(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let p50 = if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    };
    let p95 = v[(95 * n).div_ceil(100) - 1];
    let mean = v.iter().sum::<f64>() / n as f64;
    (p50, p95, mean)
}

/// Inspects the live BVH directory for a set of title page numbers:
/// returns `(distinct_arenas, non_contiguous_pairs, total_arenas)` where
/// `non_contiguous_pairs` counts how many consecutive `(p, p + 1)` BVH page
/// pairs in `bvh_starts` either live in different arenas or are not adjacent
/// in their arena.
fn inspect_placement(bvh_starts: &[u64], pages_per_bvh: u64) -> (usize, usize, usize) {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_ref() else {
        return (0, 0, 0);
    };
    let mut arenas = HashSet::new();
    let mut split_bvhs = 0usize;
    for &start_addr in bvh_starts {
        let p0 = start_addr / PAGE;
        let mut prev: Option<(usize, usize)> = None;
        let mut bvh_split = false;
        for k in 0..pages_per_bvh {
            let p = p0 + k;
            if let Some(held) = m.pages.get(&p) {
                arenas.insert(held.copy);
                if let Some((prev_copy, prev_at)) = prev {
                    if held.copy != prev_copy || held.offset != prev_at + PAGE as usize {
                        bvh_split = true;
                    }
                }
                prev = Some((held.copy, held.offset));
            } else {
                bvh_split = true;
            }
        }
        if bvh_split {
            split_bvhs += 1;
        }
    }
    (arenas.len(), split_bvhs, m.arenas.len())
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

static LOCALITY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn arena_bitmap_contiguous_run_and_reclaim() {
    let base = 0x2_0000_0000usize;
    let mut arena = Arena::new(0, base);
    assert_eq!(arena.free_count, ARENA_PAGES);

    // Cross 64-bit word boundary (slots 0..60 taken, then take_run(10) spans words 0 and 1).
    let a = arena.take_run(60).expect("first 60");
    assert_eq!(a, base);
    let b = arena.take_run(10).expect("cross-word 10");
    assert_eq!(b, base + 60 * PAGE as usize);
    assert_eq!(arena.free_count, ARENA_PAGES - 70);

    // Punch a 3-slot hole inside word 0 (slots 10..13) and a 6-slot hole across words 0..1 (slots 62..68).
    for s in 10..13 {
        assert!(arena.give_back(base + s * PAGE as usize));
    }
    assert!(!arena.give_back(base + 10 * PAGE as usize), "double give_back must be a no-op");
    for s in 62..68 {
        assert!(arena.give_back(base + s * PAGE as usize));
    }
    assert_eq!(arena.free_count, ARENA_PAGES - 61);

    // Request of 6 skips the 3-slot hole at 10..13 and lands contiguously at 62..68.
    let c = arena.take_run(6).expect("reuse 6-slot cross-word hole");
    assert_eq!(c, base + 62 * PAGE as usize);
    // Request of 3 lands in the lowest 3-slot hole at 10..13.
    let d = arena.take_run(3).expect("reuse 3-slot hole");
    assert_eq!(d, base + 10 * PAGE as usize);
    // Remaining contiguous tail (70..1024 = 954 pages) fits exactly.
    let tail = arena.take_run(ARENA_PAGES - 70).expect("exact tail");
    assert_eq!(tail, base + 70 * PAGE as usize);
    assert_eq!(arena.free_count, 0);
    assert!(arena.take_one().is_none());
    assert!(arena.take_run(1).is_none());
}

#[test]
fn synthetic_bvh_arena_locality_packing_move_remove_and_lifetime_on_gpu() {
    let _guard = LOCALITY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = host() else { return };
    if !crate::shadow::cards() || !host.card_ready() {
        return;
    }
    eprintln!("adapter={} card_memory={}", host.adapter_label(), crate::shadow::cards());
    let heap = HostHeap::new(false, 64).expect("HostHeap");
    let pipe = TracePipeline::new(host, 1, 1, 31);
    let (rec_buf, rec_ptr) = host.shared_buffer((THREADS as usize * RECORD_BYTES) as u64).expect("records");
    let (res_buf, res_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("results");
    heap.raw_uav_at(0, &rec_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(1, &res_buf, 0, (THREADS as usize * 16) as u64);

    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0);

    // 1. Stranded Arena 0 regression test:
    //    Allocate 1 page in a builder dispatch (dropping free from 1024 -> 1023),
    //    then allocate 16 two-page packed BVHs (32 pages) on the next frame.
    //    On `origin/main` (LIFO `free`), Arena 1 is pushed on top of Arena 0,
    //    so page 0 stays in Arena 0 while the 16 BVHs land in Arena 1
    //    (`distinct_arenas == 2`). Lowest-arena-first keeps all 33 pages in 1 arena
    //    and every 2-page BVH contiguous (`split_bvhs == 0`).
    let region = TitleRegion::new(8 << 20);
    let seed_addr = region.base;
    let seed_box = make_box32(
        [0x11, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 3.5, 1.0, 1.0, 4.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let seed_tri = make_tri([-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0]);
    write_unpacked_bvh(seed_addr, &seed_box, &seed_tri);

    crate::frame::test_advance();
    {
        let list = host.new_list().expect("seed list");
        let _ = ready(&list, &heap, &[seed_addr as u64], &[], &[], Use::Reads, None).unwrap();
        list.close();
        host.run_test_list_sync(&list);
    }

    let mut bvh_addrs = Vec::with_capacity(16);
    let mut bvh_boxes_p0 = Vec::with_capacity(16);
    let mut bvh_boxes_p1 = Vec::with_capacity(16);
    let mut bvh_tris_p1 = Vec::with_capacity(16);
    for i in 0..16 {
        let addr = region.base + (1 << 20) + i * (128 << 10);
        let z = 5.0 + (i as f32) * 0.5;
        let bp0 = make_box32(
            [0x100 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
            [[-1.0, -1.0, z - 0.1, 1.0, 1.0, z + 0.1], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
        );
        let bp1 = make_box32(
            [0x200 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
            [[-1.0, -1.0, z + 0.2, 1.0, 1.0, z + 0.4], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
        );
        let tp1 = make_tri([-1.0, -1.0, z + 0.3], [1.0, -1.0, z + 0.3], [0.0, 1.0, z + 0.3]);
        write_packed_bvh_2page(addr, 2047, &bp0, &bp1, &tp1);
        bvh_addrs.push(addr as u64);
        bvh_boxes_p0.push(bp0);
        bvh_boxes_p1.push(bp1);
        bvh_tris_p1.push(tp1);
        write_ray_record(rec_ptr, i * 2, (addr + 64) as u64, 5, &ray);
        write_ray_record(rec_ptr, i * 2 + 1, (addr + 65536 + 192) as u64, 0, &ray);
    }

    crate::frame::test_advance();
    {
        let list = host.new_list().expect("packed batch");
        let r = ready(&list, &heap, &bvh_addrs, &[], &[], Use::Builds { back: false }, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0, 1, None);
        list.close();
        host.run_test_list_sync(&list);
    }
    let out = read_results(res_ptr, 32);
    for i in 0..16 {
        assert_eq!(out[i * 2], bvh::intersect(&bvh_boxes_p0[i], 5, &ray, true), "bvh {i} p0 box mismatch");
        assert_eq!(out[i * 2 + 1], bvh::intersect(&bvh_tris_p1[i], 0, &ray, true), "bvh {i} p1 tri mismatch");
    }

    let mut all_starts = vec![seed_addr as u64];
    all_starts.extend_from_slice(&bvh_addrs);
    let (distinct_all, _, _) = inspect_placement(&all_starts, 1);
    let (distinct_packed, split_packed, _) = inspect_placement(&bvh_addrs, 2);
    assert_eq!(
        distinct_all, 1,
        "seed page + 16 packed BVHs must pack into 1 arena without stranding Arena 0 (got {distinct_all})"
    );
    assert_eq!(distinct_packed, 1, "all 16 packed BVHs must live in 1 arena (got {distinct_packed})");
    assert_eq!(split_packed, 0, "every 2-page packed BVH must be contiguous in 1 arena (got {split_packed} split)");

    // 2. Fragmentation + contiguous multi-page allocation:
    //    Remove alternate BVHs (even indices) via `look_again()` sweep, then
    //    stream a new 4-page (256 KiB) packed BVH and verify it lands
    //    contiguously in 1 arena and intersects bit-for-bit on GPU.
    for i in (0..16).step_by(2) {
        // SAFETY: zeroing header of synthetic packed BVH.
        unsafe { std::ptr::write_unaligned(bvh_addrs[i] as *mut u64, 0) };
    }
    for _ in 0..=SWEEP {
        crate::frame::test_advance();
        let list = host.new_list().expect("sweep");
        let _ = ready(&list, &heap, &[], &[], &[], Use::Reads, None).unwrap();
        list.close();
    }

    // Verify removed BVH 0 now misses on GPU while surviving BVH 1 still hits!
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("after remove");
        let r = ready(&list, &heap, &[], &[], &[], Use::Reads, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0, 1, None);
        list.close();
        host.run_test_list_sync(&list);
    }
    let out = read_results(res_ptr, 4);
    assert_eq!(out[0], [bvh::NONE; 4], "removed bvh 0 p0 must miss");
    assert_eq!(out[1], bvh::MISS.map(f32::to_bits), "removed bvh 0 p1 tri must miss");
    assert_eq!(out[2], bvh::intersect(&bvh_boxes_p0[1], 5, &ray, true), "surviving bvh 1 p0 must still hit");
    assert_eq!(out[3], bvh::intersect(&bvh_tris_p1[1], 0, &ray, true), "surviving bvh 1 p1 tri must still hit");

    // Stream a 4-page (256 KiB) packed BVH into the fragmented arena.
    let big_addr = region.base + (4 << 20);
    let big_box_p0 = make_box32(
        [0x555, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 14.5, 1.0, 1.0, 15.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let big_box_p1 = make_box32(
        [0x666, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 16.5, 1.0, 1.0, 17.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    let big_tri_p1 = make_tri([-1.0, -1.0, 17.0], [1.0, -1.0, 17.0], [0.0, 1.0, 17.0]);
    write_packed_bvh_2page(big_addr, 4095, &big_box_p0, &big_box_p1, &big_tri_p1); // 4 pages = 256 KiB
    write_ray_record(rec_ptr, 0, (big_addr + 64) as u64, 5, &ray);
    write_ray_record(rec_ptr, 1, (big_addr + 65536 + 192) as u64, 0, &ray);

    crate::frame::test_advance();
    {
        let list = host.new_list().expect("big packed");
        let r = ready(&list, &heap, &[big_addr as u64], &[], &[], Use::Builds { back: false }, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0, 1, None);
        list.close();
        host.run_test_list_sync(&list);
    }
    let out = read_results(res_ptr, 2);
    assert_eq!(out[0], bvh::intersect(&big_box_p0, 5, &ray, true), "4-page packed p0 mismatch");
    assert_eq!(out[1], bvh::intersect(&big_tri_p1, 0, &ray, true), "4-page packed p1 tri mismatch");
    let (big_arenas, big_split, _) = inspect_placement(&[big_addr as u64], 4);
    assert_eq!(big_arenas, 1, "4-page packed BVH must live in 1 arena");
    assert_eq!(big_split, 0, "4-page packed BVH must be contiguous in 1 arena");

    // 3. Dynamic BLAS movement: moving `seed_box` out of the ray must change
    //    GPU intersection from HIT to MISS.
    write_ray_record(rec_ptr, 0, (seed_addr + 64) as u64, 5, &ray);
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("dyn hit");
        let r = ready(&list, &heap, &[seed_addr as u64], &[], &[], Use::Reads, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0, 1, None);
        list.close();
        host.run_test_list_sync(&list);
    }
    assert_eq!(read_results(res_ptr, 1)[0], bvh::intersect(&seed_box, 5, &ray, true), "dyn initial hit mismatch");

    let seed_box_moved = make_box32(
        [0x11, bvh::NONE, bvh::NONE, bvh::NONE],
        [[9.0, 9.0, 3.5, 10.0, 10.0, 4.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    write_unpacked_bvh(seed_addr, &seed_box_moved, &seed_tri);
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("dyn miss");
        let r = ready(&list, &heap, &[seed_addr as u64], &[], &[], Use::Reads, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0, 1, None);
        list.close();
        host.run_test_list_sync(&list);
    }
    assert_eq!(read_results(res_ptr, 1)[0], bvh::intersect(&seed_box_moved, 5, &ray, true), "dyn moved box must miss");

    // 4. Fence-gated retirement under lowest-arena-first reuse:
    //    Submit Flight 1 reading `big_addr` asynchronously (`run_test_list_async`),
    //    then immediately evict `big_addr` via `gone()` and allocate a brand-new
    //    4-page packed BVH `repl_addr` BEFORE waiting for Flight 1's fence.
    //    Retired slots MUST stay out of Arena 0's bitmap until Flight 1 finishes,
    //    so Flight 1 still sees `big_box_p0` intact and `repl_addr` gets fresh
    //    slots without overwriting Flight 1's pages!
    let (res_f1_buf, res_f1_ptr) = host.shared_buffer((THREADS as usize * 16) as u64).expect("res f1");
    heap.raw_uav_at(2, &rec_buf, 0, (THREADS as usize * RECORD_BYTES) as u64);
    heap.raw_uav_at(3, &res_f1_buf, 0, (THREADS as usize * 16) as u64);
    write_ray_record(rec_ptr, 0, (big_addr + 64) as u64, 5, &ray);
    crate::frame::test_advance();
    let list_f1 = host.new_list().expect("flight 1");
    let r_f1 = ready(&list_f1, &heap, &[], &[], &[], Use::Reads, None).unwrap();
    pipe.dispatch(&list_f1, &heap, &r_f1, 2, 1, None);
    list_f1.close();
    let (sub_f1, fence_f1) = host.run_test_list_async(&list_f1);

    let big_slot_before = MEMORY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|m| m.pages.get(&((big_addr as u64) / PAGE)).map(|h| (h.copy, h.offset)))
        .expect("big_addr page must be held");

    // Evict `big_addr` while Flight 1 may still be in flight.
    // SAFETY: zeroing header of synthetic packed BVH.
    unsafe { std::ptr::write_unaligned(big_addr as *mut u64, 0) };
    {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_mut().unwrap().gone(big_addr as u64);
    }

    // Stream a replacement 4-page packed BVH with different geometry right away.
    let repl_addr = region.base + (5 << 20);
    let repl_box_p0 = make_box32(
        [0x777, bvh::NONE, bvh::NONE, bvh::NONE],
        [[-1.0, -1.0, 20.5, 1.0, 1.0, 21.5], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
    );
    write_packed_bvh_2page(repl_addr, 4095, &repl_box_p0, &big_box_p1, &big_tri_p1);
    crate::frame::test_advance();
    let list_f2 = host.new_list().expect("flight 2");
    let r_f2 = ready(&list_f2, &heap, &[repl_addr as u64], &[], &[], Use::Builds { back: false }, None).unwrap();
    if sub_f1 != 0 {
        let repl_slot = MEMORY
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|m| m.pages.get(&((repl_addr as u64) / PAGE)).map(|h| (h.copy, h.offset)))
            .expect("repl_addr page must be held");
        assert_ne!(
            repl_slot, big_slot_before,
            "retired slot must not be handed back to Arena 0 bitmap while Flight 1 is still in flight"
        );
    }
    list_f2.close();
    host.wait_test_fence(fence_f1);
    assert_eq!(
        read_results(res_f1_ptr, 1)[0],
        bvh::intersect(&big_box_p0, 5, &ray, true),
        "in-flight Flight 1 must still read pre-removal geometry"
    );

    // Now submit Flight 2 and verify `repl_addr` renders cleanly AND retired
    // pages return to the arena bitmap once Flight 1 has completed.
    write_ray_record(rec_ptr, 0, (repl_addr + 64) as u64, 5, &ray);
    let list_f2_trace = host.new_list().expect("flight 2 trace");
    pipe.dispatch(&list_f2_trace, &heap, &r_f2, 0, 1, None);
    list_f2_trace.close();
    host.run_test_list_sync(&list_f2);
    host.run_test_list_sync(&list_f2_trace);
    assert_eq!(
        read_results(res_ptr, 1)[0],
        bvh::intersect(&repl_box_p0, 5, &ray, true),
        "replacement packed BVH must intersect cleanly"
    );
    let retiring_left = {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let m = guard.as_mut().unwrap();
        m.reclaim_retired();
        m.retiring.len()
    };
    assert_eq!(retiring_left, 0, "all retired pages must return to arena bitmap after GPU fence completes");
}

/// Sustained-load GPU benchmark measuring arena locality, split multi-page BVHs,
/// hardware D3D12 BVH traversal time (32,768 rays x 64 steps = 2,097,152 steps/frame),
/// frame wall time, bytes uploaded, and peak private memory after realistic
/// 8-arena streaming churn — separately for static and dynamic scenes, verifying
/// EVERY measured frame bit-for-bit against `victory_shader::bvh::intersect`.
#[test]
#[ignore = "GPU benchmark (run with --release --features diagnostics -- --exact --ignored --nocapture --test-threads=1)"]
fn bench_bvh_arena_locality_sustained() {
    let _guard = LOCALITY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(host) = host() else {
        panic!("no host GPU available");
    };
    assert!(crate::shadow::cards() && host.card_ready(), "card memory required");
    eprintln!("adapter={} card_memory={}", host.adapter_label(), crate::shadow::cards());

    const RECORD_COUNT: usize = 4096;
    const RECORD_MASK: u32 = (RECORD_COUNT as u32) - 1;
    const GROUPS: u32 = 1024; // 1024 * 32 = 32,768 threads
    const STEPS: u32 = 64;    // 32,768 * 64 = 2,097,152 image_bvh64_intersect_ray calls/frame
    const STRIDE: u32 = 131;  // coprime with 4096
    const WARMUP: usize = 10;
    const FRAMES: usize = 60;

    let heap = HostHeap::new(false, 64).expect("HostHeap");
    let pipe = TracePipeline::new(host, STEPS, STRIDE, RECORD_MASK);
    let timer = GpuTimer::new(host);

    let (rec_buf, rec_ptr) = host.shared_buffer((RECORD_COUNT * RECORD_BYTES) as u64).expect("records");
    let (res_buf, res_ptr) = host.shared_buffer((RECORD_COUNT * 16) as u64).expect("results");
    heap.raw_uav_at(0, &rec_buf, 0, (RECORD_COUNT * RECORD_BYTES) as u64);
    heap.raw_uav_at(1, &res_buf, 0, (RECORD_COUNT * 16) as u64);

    let mem_before_mib = process_private_mib();

    // --- Phase 0: Realistic 8-arena streaming churn ---
    // Stream 8 sectors of 64 MiB (each 128 packed BVHs x 512 KiB = 8 pages = 64 MiB),
    // then unload 75% of them in interleaved sector order (0, 1, ..., 7, 0, 1, ...)
    // via `look_again()` sweep, leaving persistent landmarks plus holes across all 8 arenas.
    const CHURN_SECTORS: usize = 8;
    const BVHS_PER_SECTOR: usize = 128;
    const CHURN_BVH_BYTES: usize = 512 << 10; // 8 pages
    let mut churn_regions = Vec::with_capacity(CHURN_SECTORS);
    let mut churn_addrs = vec![Vec::with_capacity(BVHS_PER_SECTOR); CHURN_SECTORS];
    for s in 0..CHURN_SECTORS {
        let reg = TitleRegion::new(BVHS_PER_SECTOR * CHURN_BVH_BYTES);
        for i in 0..BVHS_PER_SECTOR {
            let addr = reg.base + i * CHURN_BVH_BYTES;
            write_packed_header_only(addr, (CHURN_BVH_BYTES as u32 / 64) - 1);
            churn_addrs[s].push(addr as u64);
        }
        crate::frame::test_advance();
        let list = host.new_list().expect("churn sector");
        let _ = ready(&list, &heap, &churn_addrs[s], &[], &[], Use::Builds { back: false }, None).unwrap();
        list.close();
        host.run_test_list_sync(&list);
        churn_regions.push(reg);
    }

    // Unload sector 0 completely (oldest world sector behind the camera -> 1024 free
    // pages in Arena 0) and partially unload sectors 1..7 (8 BVHs = 64 pages each).
    // Because `look_again()` sweeps `packed_order` in order, Arena 0's 1024 free pages
    // are pushed underneath the 7 x 64 scattered pages of Arenas 1..7 on `origin/main`'s
    // LIFO free stack.
    for i in 0..BVHS_PER_SECTOR {
        // SAFETY: zeroing header to trigger streaming eviction in `look_again`.
        unsafe { std::ptr::write_unaligned(churn_addrs[0][i] as *mut u64, 0) };
    }
    for s in 1..CHURN_SECTORS {
        for i in 0..8 {
            // SAFETY: zeroing header to trigger streaming eviction in `look_again`.
            unsafe { std::ptr::write_unaligned(churn_addrs[s][i] as *mut u64, 0) };
        }
    }
    for _ in 0..=SWEEP {
        crate::frame::test_advance();
        let list = host.new_list().expect("churn sweep");
        let _ = ready(&list, &heap, &[], &[], &[], Use::Reads, None).unwrap();
        list.close();
    }

    // --- Scene A: Static Active Neighborhood (256 packed BVHs x 128 KiB = 512 pages = 32 MiB) ---
    // Fits completely inside 1 single 64 MiB arena if packed lowest-arena-first!
    const ACTIVE_BVHS: usize = 256;
    const ACTIVE_BYTES: usize = 128 << 10; // 2 pages each
    let active_region = TitleRegion::new(ACTIVE_BVHS * ACTIVE_BYTES);
    let mut active_addrs = Vec::with_capacity(ACTIVE_BVHS);
    let mut active_bp0 = Vec::with_capacity(ACTIVE_BVHS);
    let mut active_bp1 = Vec::with_capacity(ACTIVE_BVHS);
    let mut active_tp1 = Vec::with_capacity(ACTIVE_BVHS);
    for i in 0..ACTIVE_BVHS {
        let addr = active_region.base + i * ACTIVE_BYTES;
        let z = 3.0 + (i as f32) * 0.25;
        let bp0 = make_box32(
            [0x1000 + i as u32, 0x2000 + i as u32, bvh::NONE, bvh::NONE],
            [
                [-1.0, -1.0, z - 0.1, 1.0, 1.0, z + 0.1],
                [-0.5, -0.5, z + 0.2, 0.5, 0.5, z + 0.4],
                [f32::NAN; 6],
                [f32::NAN; 6],
            ],
        );
        let bp1 = make_box32(
            [0x3000 + i as u32, 0x4000 + i as u32, bvh::NONE, bvh::NONE],
            [
                [-1.0, -1.0, z + 0.5, 1.0, 1.0, z + 0.7],
                [-0.8, -0.8, z + 0.8, 0.8, 0.8, z + 1.0],
                [f32::NAN; 6],
                [f32::NAN; 6],
            ],
        );
        let tp1 = make_tri([-1.0, -1.0, z + 0.6], [1.0, -1.0, z + 0.6], [0.0, 1.0, z + 0.6]);
        write_packed_bvh_2page(addr, (ACTIVE_BYTES as u32 / 64) - 1, &bp0, &bp1, &tp1);
        active_addrs.push(addr as u64);
        active_bp0.push(bp0);
        active_bp1.push(bp1);
        active_tp1.push(tp1);
    }

    // Stream the 256 active BVHs in.
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("stream active");
        let _ = ready(&list, &heap, &active_addrs, &[], &[], Use::Builds { back: false }, None).unwrap();
        list.close();
        host.run_test_list_sync(&list);
    }
    let (static_arenas, static_split, total_arenas_static) = inspect_placement(&active_addrs, 2);

    // Populate 4,096 ray records cycling through page 0 box32, page 1 box32, and page 1 tri of all 256 BVHs.
    let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0);
    let mut step_oracle = Vec::with_capacity(RECORD_COUNT);
    for lane in 0..RECORD_COUNT {
        let bvh_idx = (lane * 7 + (lane >> 5)) % ACTIVE_BVHS;
        let base = active_addrs[bvh_idx];
        let (addr, kind, expected) = match lane % 3 {
            0 => (base + 64, 5, bvh::intersect(&active_bp0[bvh_idx], 5, &ray, true)),
            1 => (base + 65536, 5, bvh::intersect(&active_bp1[bvh_idx], 5, &ray, true)),
            _ => (base + 65536 + 192, 0, bvh::intersect(&active_tp1[bvh_idx], 0, &ray, true)),
        };
        write_ray_record(rec_ptr, lane, addr, kind, &ray);
        step_oracle.push(expected);
    }
    // Precompute exact expected final [v13, v14, v15, sum(v13)] for each of the 4,096 lanes.
    let mut static_expected = Vec::with_capacity(RECORD_COUNT);
    for lane in 0..RECORD_COUNT {
        let mut acc = 0u32;
        let mut last = [0u32; 4];
        for s in 0..STEPS {
            let idx = (lane + (s * STRIDE) as usize) & (RECORD_MASK as usize);
            last = step_oracle[idx];
            acc = acc.wrapping_add(last[0]);
        }
        static_expected.push([last[0], last[1], last[2], acc]);
    }

    let mut static_gpu_us = Vec::with_capacity(FRAMES);
    let mut static_frame_us = Vec::with_capacity(FRAMES);
    let mut static_card_bytes = 0u64;

    for f in 0..(WARMUP + FRAMES) {
        crate::frame::test_advance();
        let cb0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
        let ft0 = Instant::now();
        let list = host.new_list().expect("static trace");
        let r = ready(&list, &heap, &active_addrs[..16], &[], &[], Use::Reads, None).unwrap();
        pipe.dispatch(&list, &heap, &r, 0, GROUPS, Some(&timer));
        list.close();
        host.run_test_list_sync(&list);
        let frame_us = ft0.elapsed().as_micros() as f64;
        let gpu_us = timer.elapsed_us();

        if f >= WARMUP {
            static_gpu_us.push(gpu_us);
            static_frame_us.push(frame_us);
            static_card_bytes += crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - cb0;
            let out = read_results(res_ptr, 64);
            for i in 0..64 {
                assert_eq!(out[i], static_expected[i], "static frame {f} lane {i} oracle mismatch");
            }
        }
    }

    // --- Scene B: Dynamic BLASes Streamed into Churned Heap (64 dynamic BLASes, 16 moving/frame) ---
    // Simulate 1 more BVH (8 pages) streaming out of each of sectors 1..7 before
    // the 64 dynamic BLASes arrive (Arena 0 still has 512 free pages!).
    for s in 1..CHURN_SECTORS {
        // SAFETY: zeroing header to trigger streaming eviction in `look_again`.
        unsafe { std::ptr::write_unaligned(churn_addrs[s][8] as *mut u64, 0) };
    }
    for _ in 0..=SWEEP {
        crate::frame::test_advance();
        let list = host.new_list().expect("churn sweep dyn");
        let _ = ready(&list, &heap, &[], &[], &[], Use::Reads, None).unwrap();
        list.close();
    }

    const DYN_BVHS: usize = 64;
    const MOVING: usize = 16;
    let dyn_region = TitleRegion::new(DYN_BVHS * (64 << 10));
    let mut dyn_addrs = Vec::with_capacity(DYN_BVHS);
    let mut dyn_boxes = Vec::with_capacity(DYN_BVHS);
    let mut dyn_tris = Vec::with_capacity(DYN_BVHS);
    for i in 0..DYN_BVHS {
        let addr = dyn_region.base + i * (64 << 10);
        let z = 4.0 + (i as f32) * 0.5;
        let bx = make_box32(
            [0x7000 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
            [[-1.0, -1.0, z - 0.1, 1.0, 1.0, z + 0.1], [f32::NAN; 6], [f32::NAN; 6], [f32::NAN; 6]],
        );
        let tr = make_tri([-1.0, -1.0, z], [1.0, -1.0, z], [0.0, 1.0, z]);
        write_unpacked_bvh(addr, &bx, &tr);
        dyn_addrs.push(addr as u64);
        dyn_boxes.push(bx);
        dyn_tris.push(tr);
    }
    crate::frame::test_advance();
    {
        let list = host.new_list().expect("prime dyn");
        let _ = ready(&list, &heap, &dyn_addrs, &[], &[], Use::Reads, None).unwrap();
        list.close();
        host.run_test_list_sync(&list);
    }
    let (dyn_arenas, _, total_arenas_dyn) = inspect_placement(&dyn_addrs, 1);

    let mut dyn_gpu_us = Vec::with_capacity(FRAMES);
    let mut dyn_prep_us = Vec::with_capacity(FRAMES);
    let mut dyn_frame_us = Vec::with_capacity(FRAMES);
    let mut dyn_card_bytes = 0u64;

    for f in 0..(WARMUP + FRAMES) {
        crate::frame::test_advance();
        let shift = ((f % 32) as f32) * 0.02;
        for i in 0..MOVING {
            let addr = dyn_addrs[i] as usize;
            let z = 4.0 + (i as f32) * 0.5;
            let bx = make_box32(
                [0x7000 + i as u32, bvh::NONE, bvh::NONE, bvh::NONE],
                [
                    [-1.0 + shift, -1.0, z - 0.1, 1.0 + shift, 1.0, z + 0.1],
                    [f32::NAN; 6],
                    [f32::NAN; 6],
                    [f32::NAN; 6],
                ],
            );
            let tr = make_tri([-1.0 + shift, -1.0, z], [1.0 + shift, -1.0, z], [shift, 1.0, z]);
            write_unpacked_bvh(addr, &bx, &tr);
            dyn_boxes[i] = bx;
            dyn_tris[i] = tr;
        }
        let mut dyn_step_oracle = Vec::with_capacity(RECORD_COUNT);
        for lane in 0..RECORD_COUNT {
            let idx = (lane * 5 + (lane >> 4)) % DYN_BVHS;
            let base = dyn_addrs[idx];
            let (addr, kind, expected) = if lane % 2 == 0 {
                (base + 64, 5, bvh::intersect(&dyn_boxes[idx], 5, &ray, true))
            } else {
                (base + 192, 0, bvh::intersect(&dyn_tris[idx], 0, &ray, true))
            };
            write_ray_record(rec_ptr, lane, addr, kind, &ray);
            dyn_step_oracle.push(expected);
        }

        let cb0 = crate::shadow::CARD_STATS[0].load(Ordering::Relaxed);
        let ft0 = Instant::now();
        let list = host.new_list().expect("dyn trace");
        let p0 = Instant::now();
        let r = ready(&list, &heap, &dyn_addrs[..MOVING], &[], &[], Use::Reads, None).unwrap();
        let prep_us = p0.elapsed().as_micros() as f64;
        pipe.dispatch(&list, &heap, &r, 0, GROUPS, Some(&timer));
        list.close();
        host.run_test_list_sync(&list);
        let frame_us = ft0.elapsed().as_micros() as f64;
        let gpu_us = timer.elapsed_us();

        if f >= WARMUP {
            dyn_gpu_us.push(gpu_us);
            dyn_prep_us.push(prep_us);
            dyn_frame_us.push(frame_us);
            dyn_card_bytes += crate::shadow::CARD_STATS[0].load(Ordering::Relaxed) - cb0;
            let out = read_results(res_ptr, 64);
            for lane in 0..64 {
                let mut acc = 0u32;
                let mut last = [0u32; 4];
                for s in 0..STEPS {
                    let idx = (lane + (s * STRIDE) as usize) & (RECORD_MASK as usize);
                    last = dyn_step_oracle[idx];
                    acc = acc.wrapping_add(last[0]);
                }
                assert_eq!(out[lane], [last[0], last[1], last[2], acc], "dyn frame {f} lane {lane} oracle mismatch");
            }
        }
    }

    let mem_after_mib = process_private_mib();
    let (sg50, sg95, sgm) = percentile_summary_f64(static_gpu_us);
    let (sf50, sf95, sfm) = percentile_summary_f64(static_frame_us);
    let (dg50, dg95, dgm) = percentile_summary_f64(dyn_gpu_us);
    let (dp50, dp95, dpm) = percentile_summary_f64(dyn_prep_us);
    let (df50, df95, dfm) = percentile_summary_f64(dyn_frame_us);

    eprintln!("=== BVH ARENA LOCALITY BENCHMARK RESULTS ===");
    eprintln!(
        "static_sustained ({FRAMES}f, 2M steps/f): active_arenas={static_arenas} split_bvhs={static_split}/{ACTIVE_BVHS} total_arenas={total_arenas_static} gpu_us(p50/p95/mean)={sg50:.1}/{sg95:.1}/{sgm:.1} frame_us(p50/p95/mean)={sf50:.1}/{sf95:.1}/{sfm:.1} card_bytes/f={:.1}",
        static_card_bytes as f64 / FRAMES as f64,
    );
    eprintln!(
        "dynamic_sustained ({FRAMES}f, 2M steps/f): dyn_arenas={dyn_arenas} total_arenas={total_arenas_dyn} prep_us(p50/p95/mean)={dp50:.1}/{dp95:.1}/{dpm:.1} gpu_us(p50/p95/mean)={dg50:.1}/{dg95:.1}/{dgm:.1} frame_us(p50/p95/mean)={df50:.1}/{df95:.1}/{dfm:.1} card_bytes/f={:.1}",
        dyn_card_bytes as f64 / FRAMES as f64,
    );
    eprintln!(
        "memory: private_before_mib={mem_before_mib:.1} private_after_mib={mem_after_mib:.1} delta_mib={:.1}",
        mem_after_mib - mem_before_mib,
    );
    drop(churn_regions);
}
