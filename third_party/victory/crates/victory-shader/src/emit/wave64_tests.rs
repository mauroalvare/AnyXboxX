//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

#[allow(dead_code)]
#[path = "wave_regression_helpers.rs"]
mod wave_regression_helpers;
use wave_regression_helpers::*;

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn wave64_dpp_row_mask_uses_console_rows_in_split_32_lane_waves() {
    for with_exchange in [false, true] {
        let mut w = vop3(0x169, 1, v(0), k(3)).to_vec(); // v_mul_lo_u32 v1, v0, 3
        w.extend([
            vop2(0x25, 1, k(1), 1), // v_add_nc_u32 v1, 1, v1
            vop1(0x01, 4, LIT),     // v_mov_b32 v4, 1000
            1000,
            vop1(0x01, 4, 0xFA), // v_mov_b32 v4, v1 row_shr:1 row_mask:0x6 bank_mask:0xf
            0x6 << 28 | 0xF << 24 | 0x111 << 8 | 1,
        ]);
        if with_exchange {
            w.extend([
                vopc(0xC1, LIT, 4), // v_cmp_lt_u32 vcc, 500, v4
                500,
                sop1(0x10, 16, VCC), // s_bcnt1_i32_b64 s16, vcc
                vop1(0x01, 5, 16),   // v_mov_b32 v5, s16
            ]);
        } else {
            w.push(vop1(0x01, 5, k(0))); // v_mov_b32 v5, 0
        }
        w.extend(stores([4, 5, 1, 0]));
        let native = native(&w);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let shader = lowered(&native, &p, &HashSet::new()).expect("split wave64");
        let x = |n: u32| 3 * n + 1;
        let shifted = |n: u32| if matches!(n >> 4, 1 | 2) && n & 15 != 0 { x(n - 1) } else { 1000 };
        let count = if with_exchange {
            (0..64u32).filter(|&n| shifted(n) > 500).count() as u32
        } else {
            0
        };
        let want: Vec<u32> = (0..64u32)
            .flat_map(|n| [shifted(n), count, x(n), n])
            .collect();
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Some(got) = run_lanes(&shader, [0; 4], 256) else { return };
        same_as_console(&got, &want);
        eprintln!("ran on the hardware device (with_exchange={with_exchange}): DPP row_mask matched console rows 0..3");
    }
}

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
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
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    lowered_counted(native, p, wide, host_wave, false)
}

fn lowered_counted(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
    counted: bool,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower_counted(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    }, counted);
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// "Some lane is left" of a mask operation as a number: an upscaler's programs wait for
/// other groups in a loop that ends when every lane saw its flag, or after enough rounds:
/// both tests become -1 or 0 and SCC of `timeout | ~waiting` ends it.
#[test]
fn some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on() {
    let native = native(&[
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7D84_0200, // v_cmp_eq_u32 vcc, s0, v1
        0x8792_7E6A, // s_and_b64 s[18:19], vcc, exec
        0x8592_80C1, // s_cselect_b64 s[18:19], -1, 0
        0xBF09_8501, // s_cmp_ge_u32 s1, 5
        0x8594_80C1, // s_cselect_b64 s[20:21], -1, 0
        0x8B92_1214, // s_orn2_b64 s[18:19], s[20:21], s[18:19]
        0x8506_8081, // s_cselect_b32 s6, 1, 0
        0x7E04_0206, // v_mov_b32 v2, s6
        0xE070_0000,
        0x8002_0200, // buffer_store_dword v2, off, s[8:11], 0
    ]);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = on_64_lanes(&native, &p);
    // (lane s0 waits, if there is one; s1 rounds)
    for (wait, rounds, done) in [(0, 0, 0), (37, 4, 0), (63, 4, 0), (64, 0, 1), (37, 5, 1), (64, 9, 1)] {
        let Some(out) = dispatched(&shader, &[wait, rounds]) else {
            return;
        };
        assert_eq!(out, done, "lane {wait} after {rounds} rounds");
    }
}

/// `native` for a host of 64-lane waves.
fn on_64_lanes(native: &NativeProgram, p: &ir::Program) -> HostShader {
    let st = structure::structure(p).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(64));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered.expect("lower");
    assert_eq!(lowered.wave_size, Some(64));
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Dword 0 of the UAV after `shader` ran with root constants `constants`; none without a
/// device that takes it.
fn dispatched(shader: &HostShader, constants: &[u32]) -> Option<u32> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 64?): {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if u.root == RootType::Constants {
            ud[u.index as usize] = constants.get(u.dword as usize).copied().unwrap_or(0);
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// The same sum inside `if (t >= s0) { ... }` on a host of 64-lane waves: the lanes below s0
/// are elsewhere, and count as +0 in the console's tree.
#[test]
fn a_sum_of_halves_where_not_every_lane_runs_on_a_wave_of_64() {
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
        0x7D86_0000, // v_cmp_le_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | 23, // s_cbranch_execz to the restore of EXEC
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = on_64_lanes(&native, &p);
    let wide = |h: u16| match h {
        0 => 0.0,
        h => f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13),
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    for from in [0u32, 1, 5, 31, 32, 37, 63] {
        let mut tree: Vec<u16> = (0..64u16).map(|t| if u32::from(t) < from { 0 } else { 0x3C00 + 37 * t }).collect();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| narrow(wide(c[0]) + wide(c[1]))).collect();
        }
        let Some(out) = dispatched(&shader, &[from]) else {
            return;
        };
        assert_eq!(out, u32::from(tree[0]), "lanes from {from}");
    }
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around_with_loop_exits(&st, &needs, true), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    whole_uav(shader, constants, 1).map(|out| out[0])
}

/// As [`on_the_card`], and gives the first `dwords` dwords of the UAV.
fn whole_uav(shader: &HostShader, constants: [u32; 2], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

#[path = "wave64_after_loop.rs"]
mod after_loop;

/// A dispatch counted in threads: of the last group only the lanes below the
/// count store (after DexNom/VIctory bc3cd4b0).
#[test]
fn the_lanes_past_a_count_of_threads_do_not_run() {
    // Each lane that runs adds 1 to counter[local x].
    let native = native(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7e04_0281, // v_mov_b32 v2, 1
        0xe0c8_1000, 0x8002_0201, // buffer_atomic_add v2, v1, s[8:11], 0 offen
    ]);
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let shader = lowered_counted(&native, &p, &HashSet::new(), 64, true).unwrap();
    let limit = shader
        .user_data
        .iter()
        .find(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS_X))
        .unwrap();
    assert_eq!((limit.slot, limit.dword), (0, 64));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else { return };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else { return };
    for count in [0u32, 1, 7, 63, 64, 65, 74, 77, 127, 128] {
        let mut constants = vec![0; shader.user_data_dwords() as usize];
        constants[limit.index as usize] = count;
        let result = host
            .dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(constants))],
                views: &[View::RawUav(68)],
                samplers: &[],
                groups: [2, 1, 1],
            })
            .unwrap();
        let active = if count == 0 { 128 } else { count };
        for (i, &word) in result[0].iter().enumerate().take(64) {
            let i = i as u32;
            assert_eq!(word, u32::from(i < active) + u32::from(i + 64 < active), "count {count}, word {i}");
        }
    }
}

/// The prefix of wave totals the console's structure builder computes: each console
/// wave of a 256-thread group writes its total (10 + its index) to LDS, then, under
/// `if (id >> 6 != 0)`, loops over the totals of the waves before it with lanes leaving at their
/// own wave's index, reading the running sum's first lane (`v_readfirstlane`) every iteration.
/// (The total is read from the wave's last lane first, as the builder's scan does: a read
/// across the two host waves, which makes them exchange.)
/// The condition and the sum are the same in every lane of a console wave, so each host wave
/// takes its own first lane; before, the read was refused ("a first-lane read where not every
/// lane runs"), which kept the builder off on 32-lane GPUs. Every thread stores the sum of the
/// waves before its own: 0, 10, 10 + 11, 10 + 11 + 12.
fn wave_prefix_program(lane_dependent: bool) -> NativeProgram {
    let lds_address = if lane_dependent {
        0x3410_0082 // v_lshlrev_b32 v8, 2, v0
    } else {
        0x7E10_0210 // v_mov_b32 v8, s16
    };
    let words = [
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0             (the console wave's index)
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
        0x4A12_0E8A, // v_add_nc_u32 v9, 10, v7
        0xD760_0013,
        0x0201_7F09, // v_readlane_b32 s19, v9, 63   (the wave's last lane: the other host wave)
        0x7E12_0213, // v_mov_b32 v9, s19
        0xD834_0000,
        0x0000_0908, // ds_write_b32 v8, v9
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 0, v7
        0xBF88_0011, // s_cbranch_execz END
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0002, // s_branch BODY
        0x87FE_0614, // LOOPTOP: s_and_b64 exec, s[20:21], s[6:7]
        0x7E22_0508, // v_readfirstlane_b32 s17, v8
        0x8F10_8212, // BODY: s_lshl_b32 s16, s18, 2
        0x8112_8112, // s_add_i32 s18, s18, 1
        lds_address, // LDS address for the next subtotal
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A10_1011, // v_add_nc_u32 v8, s17, v8
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7DA4_0E12, // v_cmpx_eq_u32 s18, v7
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF3, // s_cbranch_scc1 LOOPTOP
        0xBEFE_0400, // END: s_mov_b64 exec, s[0:1]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0801, // buffer_store_dword v8, v1, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, true, 256);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 2 << 15; // 1 KiB covers the lane-varying address as well.
    }
    native
}

#[test]
fn a_first_lane_read_of_a_wave_alike_sum_under_a_wave_alike_loop() {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };

    let native = wave_prefix_program(false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_readfirstlane_b32")
    );
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
        eprintln!("DXIL validator accepted the wave32 prefix shader");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the output has the layout and size required by OPTIONS1.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut waves).cast(),
            size_of_val(&waves) as u32,
        )
    }.expect("query wave capabilities");
    if !(waves.WaveLaneCountMin..=waves.WaveLaneCountMax).contains(&32) {
        eprintln!("the device does not support wave32; hardware dispatch skipped");
        return;
    }
    let (root, pso) = host.compute_pipeline(&shader).expect("wave32 prefix pipeline");
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(256)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    let expected = |t: u32| (0..(t >> 6)).map(|w| 10 + w).sum::<u32>();
    let got: Vec<u32> = (0..256).map(|t| out[0][t]).collect();
    for t in 0..256u32 {
        assert_eq!(got[t as usize], expected(t), "thread {t}: {got:?}");
    }
    eprintln!("ran on the hardware device: every console wave got the sum of the waves before it");
}

#[test]
fn a_lane_dependent_prefix_cannot_take_its_own_first_lane() {
    let native = wave_prefix_program(true);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let Err(error) = lowered(&native, &p, &HashSet::new()) else {
        panic!("the two host waves may read different subtotals");
    };
    assert!(error.to_string().contains("a first-lane read where not every lane runs"), "{error}");
}

/// A wave64 program of 128 threads (two console waves) that leaves early when no lane of its
/// wave is below s0, as the console's compiler does when the end of the program is too far for
/// a branch: `s_cbranch_execnz` over a far jump to the end. `extra` are words that go into the
/// far jump's block before it, and `work_at_end` puts an instruction between the jump's target
/// and the end of the program. Then the gate: every thread stores 1, or 2 when some lane of its
/// console wave that is still running has the thread number s1 (the lanes that fail a test).
fn far_early_out(extra: &[u32], work_at_end: bool) -> NativeProgram {
    let gate = [
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE90_106A, // s_bcnt1_i32_b64 s16, vcc
        0xBF06_8010, // s_cmp_eq_u32 s16, 0
        0x8511_8281, // s_cselect_b32 s17, 1, 2
        0x7E02_0211, // v_mov_b32 v1, s17
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen
    ];
    let jump = [
        0xBE8E_1F00, // s_getpc_b64 s[14:15]
        0x800E_FF0E,
        4 * (4 + gate.len() as u32), // s_add_u32 s14, s14, the way to the end
        0x820F_800F, // s_addc_u32 s15, s15, 0
        0xBE80_200E, // s_setpc_b64 s[14:15]
    ];
    let mut words = vec![
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF89_0000 | (extra.len() + jump.len()) as u32, // s_cbranch_execnz over the jump
    ];
    words.extend_from_slice(extra);
    words.extend_from_slice(&jump);
    words.extend_from_slice(&gate);
    if work_at_end {
        words.push(0x7E06_0281); // v_mov_b32 v3, 1
    }
    native_of(&words, true, 128)
}

fn branches_named(p: &ir::Program, name: &str) -> usize {
    p.blocks.iter().filter(|b| b.insts.last().is_some_and(|i| i.name() == name)).count()
}

/// What the gate of [`far_early_out`] stores in each of 128 threads.
fn gate_expected(below: u32, failing: u32) -> Vec<u32> {
    (0..128u32)
        .map(|t| {
            let some_fail = failing < below && failing / 64 == t / 64;
            match (t < below, some_fail) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 2,
            }
        })
        .collect()
}

#[test]
fn a_far_early_out_is_left_out_and_the_gate_counts_the_lanes_of_both_halves() {
    let native = far_early_out(&[], false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1);
    lower::take_needs();
    let kept = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&kept, Err(ShaderError::Unsupported { what, .. }) if what.ends_with("where not every lane runs")),
        "{:?}",
        kept.err()
    );
    for (name, form) in [("early-outs", ir::without_early_outs(&p)), ("skips", ir::without_skips(&p, false))] {
        let form = form.unwrap_or_else(|| panic!("without {name}: the far early-out stays"));
        assert_eq!(branches_named(&form, "s_cbranch_execnz"), 0, "without {name}");
    }
    let form = ir::without_early_outs(&p).expect("left out");
    let shader = lowered(&native, &form, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (threads below s0 are in bounds; the one thread s1 fails the test) A failing thread of the
    // upper half gives 2 to the lower half's threads as well, and a failing thread out of
    // bounds counts for nothing
    for (below, failing) in [(128u32, 40u32), (128, 100), (128, 63), (40, 50), (40, 10), (33, 32), (1, 0), (0, 5)] {
        let Some(got) = whole_uav(&shader, [below, failing], 128) else { return };
        assert_eq!(got, gate_expected(below, failing), "{below} threads in bounds, thread {failing} fails");
    }
    eprintln!("ran on the hardware device: the gate's count over both halves of each console wave");
}

#[test]
fn a_far_jump_that_does_work_is_not_an_early_out() {
    // work in the jump's block, and work between the jump's target and the end
    for (name, native) in [
        ("work in the jump", far_early_out(&[0xBE94_0385], false)), // s_mov_b32 s20, 5
        ("work at the target", far_early_out(&[], true)),
    ] {
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        assert_eq!(branches_named(&p, "s_cbranch_execnz"), 1, "{name}");
        assert!(ir::without_early_outs(&p).is_none(), "{name}");
        assert!(ir::without_skips(&p, false).is_none(), "{name}");
        assert!(ir::without_skips(&p, true).is_none(), "{name}");
    }
}

/// `if (t < s0) { park s[20:21]; if (s1 == 0) {..} else {..}; take it back }` and a read of
/// s[20:21] after the join, as the console's compiler leaves a parked scalar across an inner
/// branch. `then` is the inner branch's first arm; the second one leaves the parked scalar alone.
fn parked_across_an_inner_branch(then: u32) -> ir::Program {
    let body = [
        0xBE96_0414, // s_mov_b64 s[22:23], s[20:21]
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0002, // s_cbranch_scc1 to the second arm
        then,
        0xBF82_0001, // s_branch to the join
        0xBE98_0382, // s_mov_b32 s24, 2
        0xBE94_0416, // s_mov_b64 s[20:21], s[22:23]
    ];
    let mut words = vec![
        0xBE94_0406, // s_mov_b64 s[20:21], s[6:7]
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz past the body
    ];
    words.extend_from_slice(&body);
    words.extend_from_slice(&[
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_0214, // v_mov_b32 v1, s20
    ]);
    let native = native(&words);
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

#[test]
fn a_skip_whose_parked_scalars_pass_an_inner_branch_unchanged_is_left_out() {
    let p = parked_across_an_inner_branch(0xBE96_0314); // s_mov_b32 s22, s20
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    // (s22 comes back through a phi of the inner join whose two arguments are copies of one
    // value: only `deep` follows it)
    assert!(ir::without_skips(&p, false).is_none());
    let deep = ir::without_skips(&p, true).expect("left out");
    assert_eq!(branches_named(&deep, "s_cbranch_execz"), 0);
}

#[test]
fn a_skip_whose_parked_scalar_an_inner_branch_changes_is_kept() {
    let p = parked_across_an_inner_branch(0xBE96_0387); // s_mov_b32 s22, 7
    assert_eq!(branches_named(&p, "s_cbranch_execz"), 1);
    assert!(ir::without_skips(&p, false).is_none());
    assert!(ir::without_skips(&p, true).is_none());
}

#[test]
fn writelane_ignores_exec_and_masks_lane_index() {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());

    // 1. Wave32 compute: narrow EXEC to even lanes (guarded, no branch), write to odd
    // (inactive) lane 5 via unmasked index s1 = 37 (37 & 31 == 5), restore EXEC, read back
    // with v_readlane_b32 at index s1 = 37.
    let mut w32 = vec![
        vop1(0x01, 6, v(0)),    // v_mov_b32 v6, v0
        vop2(0x1B, 3, k(1), 0), // v_and_b32 v3, 1, v0
        sop1(0x03, 12, EXEC),   // s_mov_b32 s12, exec_lo
        vopc(0xD2, k(0), 3),    // v_cmpx_eq_u32 exec_lo, 0, v3 (even lanes active)
    ];
    w32.extend(vop3(0x361, 6, 0, 1)); // v_writelane_b32 v6, s0, s1
    w32.push(sop1(0x03, EXEC, 12));   // s_mov_b32 exec_lo, s12
    w32.extend(vop3(0x360, 14, v(6), 1)); // v_readlane_b32 s14, v6, s1
    w32.push(vop1(0x01, 7, 14)); // v_mov_b32 v7, s14
    w32.extend(stores([6, 7, 0, 0]));
    let native32 = native_of(&w32, false, 32);
    let p32 = ir::build(&native32.code, native32.wave64, native32.vgprs).expect("ir::build");
    let shader32 = lowered(&native32, &p32, &HashSet::new()).expect("lower wave32");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader32.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let val32 = 0xDEAD_BEEF;
    let want32: Vec<u32> = (0..32u32)
        .flat_map(|t| [if t == 5 { val32 } else { t }, val32, t, t])
        .collect();
    let Some(got32) = run_lanes(&shader32, [val32, 37, 0, 0], 128) else { return };
    same_as_console(&got32, &want32);

    // 2. Split wave64 exchange on a 32-lane host: v_readlane_b32 and v_writelane_b32 with
    // unmasked indices s0 = 104 (lane 40) and s1 = 69 (lane 5).
    let mut wex = vop3(0x169, 6, v(0), k(3)).to_vec(); // v_mul_lo_u32 v6, v0, 3
    wex.push(vop2(0x25, 6, k(1), 6));                  // v_add_nc_u32 v6, 1, v6
    wex.extend(vop3(0x360, 14, v(6), 0));              // v_readlane_b32 s14, v6, s0 (lane 40)
    wex.extend(vop3(0x360, 15, v(6), 1));              // v_readlane_b32 s15, v6, s1 (lane 5)
    wex.extend(vop3(0x361, 6, 2, 0));                  // v_writelane_b32 v6, s2, s0 (lane 40)
    wex.push(vop1(0x01, 7, 14));                       // v_mov_b32 v7, s14
    wex.push(vop1(0x01, 8, 15));                       // v_mov_b32 v8, s15
    wex.extend(stores([6, 7, 8, 0]));
    let native_ex = native(&wex);
    let pex = ir::build(&native_ex.code, native_ex.wave64, native_ex.vgprs).expect("ir::build");
    let shader_ex = lowered(&native_ex, &pex, &HashSet::new()).expect("split wave64");
    let val_ex = 0x1234_5678;
    let want_ex: Vec<u32> = (0..64u32)
        .flat_map(|t| [if t == 40 { val_ex } else { 3 * t + 1 }, 3 * 40 + 1, 3 * 5 + 1, t])
        .collect();
    let Some(got_ex) = run_lanes(&shader_ex, [104, 69, val_ex, 0], 256) else { return };
    same_as_console(&got_ex, &want_ex);
    eprintln!("ran on the hardware device: writelane ignored EXEC and readlane/writelane masked lane indices in wave32 and split wave64");
}

/// `v_writelane_b32 vN, sX, K`.
fn writelane(n: u32, x: u32, lane: u32) -> Vec<u32> {
    vec![0xD761_0000 | n, (128 + lane) << 9 | x]
}

/// `v_readlane_b32 sX, vN, K`.
fn readlane(x: u32, n: u32, lane: u32) -> Vec<u32> {
    vec![0xD760_0000 | x, (128 + lane) << 9 | (256 + n)]
}

fn parked(words: &[Vec<u32>]) -> ir::Program {
    let native = native(&words.concat());
    ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build")
}

fn instructions<'a>(p: &'a ir::Program, name: &str) -> Vec<&'a ir::Inst> {
    p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == name).collect()
}

fn regs<T>(of: &[(ir::Reg, T)]) -> Vec<ir::Reg> {
    of.iter().map(|r| r.0).collect()
}

#[test]
fn a_pair_parked_in_lanes_that_hold_other_scalars_elsewhere_moves_as_a_pair() {
    // s[20:21] goes into lanes 4 and 5 of v10 and comes back into s[22:23]; later lane 4
    // holds s30 for a while: no lane of the two is the pair's alone
    let p = parked(&[
        writelane(10, 20, 4),
        writelane(10, 21, 5),
        vec![0x7E04_0281], // v_mov_b32 v2, 1
        readlane(22, 10, 4),
        readlane(23, 10, 5),
        writelane(10, 30, 4),
        readlane(31, 10, 4),
    ]);
    let wide = instructions(&p, "s_mov_b64");
    let [save, restore] = wide[..] else { panic!("one pair save and one pair restore: {} moves", wide.len()) };
    assert_eq!(regs(&save.reads), [ir::Reg::S(20), ir::Reg::S(21)]);
    assert_eq!(regs(&restore.writes), [ir::Reg::S(22), ir::Reg::S(23)]);
    assert_eq!(regs(&restore.reads), regs(&save.writes));
    // the single scalar has its own lane register, not the pair's
    let narrow = instructions(&p, "s_mov_b32");
    let lone_save = narrow.iter().find(|i| regs(&i.reads) == [ir::Reg::S(30)]).expect("the save of s30");
    let lone_restore = narrow.iter().find(|i| regs(&i.writes) == [ir::Reg::S(31)]).expect("the restore into s31");
    assert_eq!(regs(&lone_restore.reads), regs(&lone_save.writes));
    assert!(regs(&lone_save.writes).iter().all(|r| !regs(&save.writes).contains(r)));
}

#[test]
fn a_parked_pair_another_save_or_restore_shares_stays_two_scalars() {
    let cases = [
        (
            "a lone restore of the lane",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
                readlane(24, 10, 4),
            ],
        ),
        (
            "another save of the lane on one path to the restore",
            vec![
                writelane(10, 20, 4),
                writelane(10, 21, 5),
                vec![0xBF06_8001, 0xBF85_0002], // s_cmp_eq_u32 s1, 0 and s_cbranch_scc1 over the save
                writelane(10, 30, 4),
                readlane(22, 10, 4),
                readlane(23, 10, 5),
            ],
        ),
    ];
    for (name, words) in cases {
        let p = parked(&words);
        assert!(instructions(&p, "s_mov_b64").is_empty(), "{name}");
        assert!(!instructions(&p, "s_mov_b32").is_empty(), "{name}");
    }
}

/// A spilled 64-thread wave64 compute program in a partial final group
/// (`partial`, `USER_DISPATCH_THREADS_X`), as the structure builder's Box
/// rehydration kernels are: it reads `s_bcnt1_i32_b64(exec)` on entry,
/// takes a whole-wave OR of `1 << (lane >> 1)` (which crosses the 32-lane
/// halves), and stores `(count, or)` at `uav[lane * 2..+2]` for every lane.
///
/// On a host of 32-lane waves it must run as two host waves (waves of 32,
/// so the pipeline is made: NVIDIA refuses `WaveSize(64)`), and both must
/// stay alive for the barriers of the halves' exchange while the lanes past
/// the limit count for nothing and store nothing (adapted from
/// fxceless/victory-fork#646's test).
#[test]
fn partial_dispatch_with_a_cross_half_wave_op_runs_in_32_lane_waves() {
    let mut words = vec![
        // Read descriptor-table offset (user_data 16) from spilled table s[0:1] + 0.
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0x3D << 26 | 4 << 6 | 1,
        125 << 25, // s_load_dword s4, s[2:3], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x3402_0083, // v_lshlrev_b32 v1, 3, v0 (lane * 8 bytes)
        0x7E04_0206, // v_mov_b32 v2, s6 (popcnt(exec))
        0x7E06_0207, // v_mov_b32 v3, s7 (wave OR)
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 2 << 8 | 1, // buffer_store_dwordx2 v[2:3], v1, s[8:11], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 16,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [64, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::spill::build(&native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered.unwrap_or_else(|e| panic!("lower: {e}"));
    // NVIDIA (32 lanes) refuses WaveSize(64): the program must not keep the console's.
    assert_eq!(lowered.wave_size, Some(32), "the partial program kept 64-lane waves");
    let limit = *lowered
        .user_data
        .iter()
        .find(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS_X))
        .expect("USER_DISPATCH_THREADS_X");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap_or_else(|e| panic!("DXIL validation: {e}"));
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    let (root, pso) = host.compute_pipeline(&shader).expect("the host makes the 32-lane pipeline");
    for (count, active, expected_or) in [(45u32, 45usize, 0x007F_FFFFu32), (1, 1, 1), (33, 33, 0x1_FFFF), (0, 64, u32::MAX)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        ud[limit.index as usize] = count;
        let out = host
            .dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(128)],
                samplers: &[],
                groups: [1, 1, 1],
            })
            .unwrap_or_else(|e| panic!("dispatch (count {count}): {e}"));
        for lane in 0..64 {
            let got = (out[0][lane * 2], out[0][lane * 2 + 1]);
            let want = if lane < active { (active as u32, expected_or) } else { (0, 0) };
            assert_eq!(got, want, "count {count}, lane {lane}");
        }
    }
    eprintln!("ran on the hardware device: partial counts 45, 1, 33, 0 exact in 32-lane waves");
}

/// Cross-lane instructions in a spilled 64-thread wave64 compute program with a
/// partial final group (`USER_DISPATCH_THREADS_X`, `partial_active`) on a 32-lane
/// host (NVIDIA): unlaunched tail lanes stay alive in the host wave for group
/// barriers, so `v_permlane16_b32`, `v_permlanex16_b32`, `ds_bpermute_b32`,
/// `v_readfirstlane_b32`, DPP16 (`bound_ctrl:0` and `bound_ctrl:1`),
/// `ds_swizzle_b32`, and DPP8 (`fi:0`) must treat them as inactive (`EXEC == 0`),
/// matching the console and 64-lane hosts where tail lanes exit at entry.
#[test]
fn partial_dispatch_cross_lane_ops_exclude_unlaunched_lanes() {
    let vop3 = |op: u32, opsel: u32, dst: u32, s0: u32, s1: u32, s2: u32| -> [u32; 2] {
        [
            0xD400_0000 | op << 16 | opsel << 11 | dst,
            s0 | s1 << 9 | s2 << 18,
        ]
    };
    let store_off = |data: u32, addr: u32, off: u32| -> [u32; 2] {
        [0xE070_1000 | off, 0x8002_0000 | data << 8 | addr]
    };
    let mut words = vec![
        // Read descriptor-table offset (user_data 16) from spilled table s[0:1] + 0.
        0x8703_FF01,
        0x0000_00FF, // s_and_b32 s3, s1, 0xff
        0xBE82_0300, // s_mov_b32 s2, s0
        0x3D << 26 | 4 << 6 | 1,
        125 << 25,   // s_load_dword s4, s[2:3], 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25,   // s_load_dwordx4 s[8:11], s[4:5], 0
        // Force wave64-on-wave32 exchange so partial_active keeps tail threads alive.
        0xBE86_107E, // s_bcnt1_i32_b64 s6, exec
        // v1 = v0 + 100 (distinct non-zero value per lane: 100 + lane)
        0x4A02_00FF,
        100,         // v_add_nc_u32 v1, 100, v0
        // v2 = 0xAAAA sentinel; s12 = 0xFFFF_FFFF (select lane 15 of row)
        0x7E04_02FF,
        0xAAAA,
        0xBE8C_03C1, // s_mov_b32 s12, -1
    ];
    // 1. v_permlane16_b32 v2, v1, s12, s12 (fi:0, bound_ctrl:0 -> keep v2 if lane row+15 unlaunched)
    words.extend_from_slice(&vop3(0x377, 0b00, 2, 257, 12, 12));
    // 2. v_permlanex16_b32 v3, v1, 0, 0 op_sel:[0,1] (fi:0, bound_ctrl:1 -> 0 if other_row+0 unlaunched)
    words.extend_from_slice(&[0x7E06_02FF, 0xBBBB]); // v3 sentinel
    words.extend_from_slice(&vop3(0x378, 0b10, 3, 257, 128, 128));
    // 3. ds_bpermute_b32 v5, v4, v1 with v4 = 15 * 4 = 60 (reads lane 15 / 47 of half -> 0 if unlaunched)
    words.extend_from_slice(&[0x7E08_02BC]); // v_mov_b32 v4, 60 (inline const 128+60 = 188 = 0xBC)
    words.extend_from_slice(&[0xD800_0000 | 0xB3 << 18, 5 << 24 | 1 << 8 | 4]);
    // 4. v_readfirstlane_b32 s7, v1 under guard (v0 >= 20): when count == 10, no launched lane is in EXEC -> lane 0 (100)
    words.extend_from_slice(&[
        0x7D86_0094, // v_cmp_le_u32 vcc, 20, v0 (20 <= v0 <=> v0 >= 20; inline 20 = 148 = 0x94)
        0xBE8E_246A, // s_and_saveexec_b64 s[14:15], vcc
        0x7E0E_0501, // v_readfirstlane_b32 s7, v1 (VOP1 0x02)
        0xBEFE_040E, // s_mov_b64 exec, s[14:15]
        0x7E0C_0207, // v_mov_b32 v6, s7
    ]);
    // 5. DPP16 quad_perm:[3,3,3,3] bound_ctrl:0 fi:0 -> keep v7 (0xCCCC) if quad lane 3 unlaunched
    words.extend_from_slice(&[
        0x7E0E_02FF,
        0xCCCC,                                         // v_mov_b32 v7, 0xCCCC
        0x7E00_0000 | 7 << 17 | 1 << 9 | 0xFA,          // v_mov_b32_dpp v7, v1 ...
        0xFF00_FF01,                                    // quad_perm:[3,3,3,3] row:0xf bank:0xf bc:0 fi:0
    ]);
    // 6. DPP16 quad_perm:[3,3,3,3] bound_ctrl:1 fi:0 -> 0 if quad lane 3 unlaunched
    words.extend_from_slice(&[
        0x7E10_02FF,
        0xDDDD,                                         // v_mov_b32 v8, 0xDDDD
        0x7E00_0000 | 8 << 17 | 1 << 9 | 0xFA,          // v_mov_b32_dpp v8, v1 ...
        0xFF08_FF01,                                    // quad_perm:[3,3,3,3] row:0xf bank:0xf bc:1 fi:0
    ]);
    // 7. ds_swizzle_b32 v9, v1 offset:0x80FF (quad_perm:[3,3,3,3] -> 0 if quad lane 3 unlaunched)
    words.extend_from_slice(&[0xD800_0000 | 0x35 << 18 | 0x80FF, 9 << 24 | 1]);
    // 8. DPP8 dpp8:[7,7,7,7,7,7,7,7] fi:0 -> keep v10 (0xEEEE) if 8-lane group lane 7 unlaunched
    words.extend_from_slice(&[
        0x7E14_02FF,
        0xEEEE,                                         // v_mov_b32 v10, 0xEEEE
        0x7E00_0000 | 10 << 17 | 1 << 9 | 0xE9,         // v_mov_b32_dpp v10, v1 dpp8:[7..7] fi:0
        0xFFFF_FF01,
    ]);
    // Store 8 dwords per lane (stride 32 bytes: v15 = v0 << 5).
    words.push(0x341E_0085); // v_lshlrev_b32 v15, 5, v0
    words.extend_from_slice(&store_off(2, 15, 0));
    words.extend_from_slice(&store_off(3, 15, 4));
    words.extend_from_slice(&store_off(5, 15, 8));
    words.extend_from_slice(&store_off(6, 15, 12));
    words.extend_from_slice(&store_off(7, 15, 16));
    words.extend_from_slice(&store_off(8, 15, 20));
    words.extend_from_slice(&store_off(9, 15, 24));
    words.extend_from_slice(&store_off(10, 15, 28));
    words.push(0xBF81_0000); // s_endpgm

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 16,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 2 << 1, threads: [64, 1, 1] }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::spill::build(&native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered.unwrap_or_else(|e| panic!("lower: {e}"));
    assert_eq!(lowered.wave_size, Some(32));
    let limit = *lowered
        .user_data
        .iter()
        .find(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS_X))
        .expect("USER_DISPATCH_THREADS_X");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).unwrap_or_else(|e| panic!("DXIL validation: {e}"));
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else { return };
    let (root, pso) = host.compute_pipeline(&shader).expect("32-lane pipeline");
    for (count, active) in [(10u32, 10usize), (45, 45), (0, 64)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        ud[limit.index as usize] = count;
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(512)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch (count {count}): {e}"));
        for lane in 0..64usize {
            let got = [
                out[0][lane * 8],
                out[0][lane * 8 + 1],
                out[0][lane * 8 + 2],
                out[0][lane * 8 + 3],
                out[0][lane * 8 + 4],
                out[0][lane * 8 + 5],
                out[0][lane * 8 + 6],
                out[0][lane * 8 + 7],
            ];
            let want = if lane < active {
                let row15 = (lane & !15) + 15;
                let w_perm16 = if row15 < active { 100 + row15 as u32 } else { 0xAAAA };
                let other0 = (lane & !31) | ((lane & 16) ^ 16);
                let w_permx16 = if other0 < active { 100 + other0 as u32 } else { 0 };
                let half15 = (lane & !31) + 15;
                let w_bperm = if half15 < active { 100 + half15 as u32 } else { 0 };
                let w_first = if active > 20 { 120 } else { 100 };
                let quad3 = (lane & !3) + 3;
                let w_dpp_bc0 = if quad3 < active { 100 + quad3 as u32 } else { 0xCCCC };
                let w_dpp_bc1 = if quad3 < active { 100 + quad3 as u32 } else { 0 };
                let w_swizzle = if quad3 < active { 100 + quad3 as u32 } else { 0 };
                let oct7 = (lane & !7) + 7;
                let w_dpp8 = if oct7 < active { 100 + oct7 as u32 } else { 0xEEEE };
                [w_perm16, w_permx16, w_bperm, w_first, w_dpp_bc0, w_dpp_bc1, w_swizzle, w_dpp8]
            } else {
                [0; 8]
            };
            assert_eq!(got, want, "count {count}, lane {lane}");
        }
    }
    eprintln!("ran on the hardware device: partial_active excluded unlaunched lanes in permlane16/x16, ds_bpermute, readfirstlane, DPP16, ds_swizzle, and DPP8");
}

/// One console wave of 64 threads with 512 bytes of LDS. The threads clear
/// the flag, then the threads below s0 raise it and read it back inside
/// their branch. Every thread stores the final flag plus one at its own
/// UAV offset. `second` adds another write in the branch after the read.
fn flag_program(second: Option<u32>) -> NativeProgram {
    let mut body = vec![
        0x7E12_0280, // v_mov_b32 v9, 0
    ];
    body.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 9)); // ds_write_b32 v9, v9
    let mut branch = vec![0x7E04_0281]; // v_mov_b32 v2, 1
    branch.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 2)); // ds_write_b32 v9, v2
    // (a scalar exit from the branch makes it blocks of their own, which only its lanes run)
    let mut rest = ds(DS_READ_B32, 3, 9, 0).to_vec(); // ds_read_b32 v3, v9
    if let Some(k) = second {
        rest.push(0x7E0C_02FF); // v_mov_b32 v6, lit
        rest.push(k);
        rest.extend_from_slice(&ds(DS_WRITE_B32, 0, 9, 6)); // ds_write_b32 v9, v6
    }
    branch.extend_from_slice(&[
        0xBF07_8001,                       // s_cmp_lg_u32 s1, 0
        0xBF85_0000 | rest.len() as u32, // s_cbranch_scc1 past the rest
    ]);
    branch.extend_from_slice(&rest);
    body.extend_from_slice(&[
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | branch.len() as u32, // s_cbranch_execz past the branch
    ]);
    body.extend_from_slice(&branch);
    body.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
    body.extend_from_slice(&ds(DS_READ_B32, 5, 9, 0)); // ds_read_b32 v5, v9
    body.extend_from_slice(&[
        0x4A0A_0A81, // v_add_nc_u32 v5, 1, v5
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 1, // buffer_store_dword v5, v1, s[8:11], 0 offen
    ]);
    let mut native = native(&body);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [0x36 << 26 | (op as u32) << 18, (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_READ_B32: u8 = 0x36;

/// The wave size `native` lowers to on a host of 32-lane waves.
fn flag_wave(native: &NativeProgram) -> Option<u32> {
    let p = crate::spill::build(native).expect("build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    lowered.expect("lower").wave_size
}

#[test]
fn wave64_thread_ticket_fits_at_the_d3d12_lds_limit() {
    let mut words = vec![
        vop2(0x1A, 1, k(2), 0),  // byte offset 4 * lane
        vop2(0x26, 2, k(63), 0), // mirror lane
        vop2(0x1A, 2, k(2), 2),  // mirror byte offset
    ];
    words.extend(vop3(0x169, 3, v(0), k(3)));
    words.push(vop2(0x25, 3, k(1), 3));
    words.extend([0xD834_0000, 1 | 3 << 8]); // ds_write_b32 v1, v3
    words.extend([0xD8D8_0000, 2 | 4 << 24]); // ds_read_b32 v4, v2
    words.extend(stores([4, 3, 0, 0]));
    let mut native = native(&words);
    let StageIo::Cs(cs) = &mut native.io else { unreachable!() };
    cs.rsrc2 |= 64 << 15; // 32 KiB promised by the shader metadata.
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("split wave64 at the LDS limit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("valid group-shared allocation");
    }
    let want: Vec<_> = (0..64u32).flat_map(|n| [3 * (63 - n) + 1, 3 * n + 1, n, n]).collect();
    let _lock = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(actual) = run_lanes(&shader, [0; 4], 256) else { return };
    same_as_console(&actual, &want);
}

#[test]
fn wave64_exchange_reports_an_lds_overflow() {
    let mut words = readlane(14, 0, 0); // reading any console lane requires cross-half exchange
    words.push(vop1(0x01, 3, 14)); // the read remains live
    words.extend(stores([3, 0, 0, 0]));
    let mut native = native(&words);
    let StageIo::Cs(cs) = &mut native.io else { unreachable!() };
    cs.rsrc2 |= 64 << 15;
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(matches!(lowered(&native, &p, &HashSet::new()), Err(ShaderError::Unsupported {
        what: "compute LDS and wave exchange exceed D3D12's 32 KiB limit",
        ..
    })));
}

#[test]
fn a_branch_that_only_raises_an_lds_flag_runs_in_waves_of_32() {
    // a read that misses the other host wave's write changes nothing the branch leaves behind
    assert_eq!(flag_wave(&flag_program(None)), Some(32));
    // after the read the branch writes the flag again, with the same constant
    assert_eq!(flag_wave(&flag_program(Some(1))), Some(32));
}

#[test]
fn a_branch_that_writes_lds_two_different_values_keeps_64_lanes() {
    // a second value after the read: what the flag ends up holding depends on what was read
    assert_eq!(flag_wave(&flag_program(Some(2))), Some(64));
}

#[test]
fn both_host_waves_observe_the_lds_flag_after_the_join() {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };

    let native = flag_program(Some(1));
    let p = crate::spill::build(&native).expect("build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut validator) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut dxil = shader.dxil.clone();
        validator.validate(&mut dxil).expect("validate the split wave shader");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware adapter, skipping flag dispatches");
        return;
    };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the output points at a feature structure of the specified size.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut waves).cast(),
            size_of_val(&waves) as u32,
        )
    }
    .expect("query host wave sizes");
    if !waves.WaveOps.as_bool() || waves.WaveLaneCountMax != 32 {
        eprintln!("not a 32-lane host, skipping flag dispatches");
        return;
    }
    let (root, pso) = host.compute_pipeline(&shader).expect("create the split wave pipeline");
    for active in [0, 1, 31, 32, 33, 63, 64] {
        // Exercise both sides of the branch's scalar skip.
        for skip in [0, 1] {
            let mut ud = vec![0; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
                    ud[u.index as usize] = [active, skip][d as usize];
                }
            }
            let out = host
                .dispatch(&root, &pso, &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                })
                .expect("dispatch the split wave pipeline");
            assert_eq!(out[0], [1 + u32::from(active != 0); 64], "{active} active lanes, skip {skip}");
        }
    }
    eprintln!("all 14 flag dispatches passed on a 32-lane hardware adapter");
}

/// Lowers `body` as a console program of one 64-lane wave (or 32 when `wave64` is not set)
/// for a host of 32-lane waves, validates it and runs it there; each lane stores v3 to dword
/// v0 of the raw UAV. None where the host device does not run it.
fn permuted(body: &[u32], wave64: bool) -> Option<Vec<u32>> {
    let lanes = if wave64 { 64 } else { 32 };
    let mut words = body.to_vec();
    words.extend_from_slice(&[
        0x340C_0082, // v_lshlrev_b32 v6, 2, v0
        0xE070_1000,
        0x8002_0306, // buffer_store_dword v3, v6, s[8:11], 0 offen
    ]);
    let native = native_of(&words, wave64, lanes);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // The same program for a host of 64-lane waves lowers too.
    let wide = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower for 64-lane waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = wide.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed (64-lane waves)");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]))],
                views: &[View::RawUav(lanes)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][..lanes as usize].to_vec())
}

/// A wave64 `ds_bpermute_b32` permutes lanes 0..31 and 32..63 apart (RDNA 2 ISA, section
/// 10.4.4; upstream 0665a0e8, #573): v1 = ((lane & 31) << 2) + 0x103 with offset 12 drops the
/// high and sub-dword address bits and wraps inside each half (lanes 29..31 read 0..2, lanes
/// 61..63 read 32..34, lane 32 reads lane 35, not lane 3). Run in host waves of 32, the
/// program needs no exchange between them.
#[test]
fn a_wave64_backward_permute_stays_within_each_32_lane_half() {
    let body = [
        0x4A04_00FF,
        0x0000_4000, // v_add_nc_u32 v2, 0x4000, v0
        0x3602_009F, // v_and_b32 v1, 31, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x4A02_02FF,
        0x0000_0103, // v_add_nc_u32 v1, 0x103, v1
        0xDACC_000C,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:12
    ];
    let Some(got) = permuted(&body, true) else { return };
    let want: Vec<u32> = (0..64).map(|l| 0x4000 + ((l & !31) | ((l + 3) & 31))).collect();
    assert_eq!(got, want);
    eprintln!("ran on the hardware device: each half permuted on its own");
}

/// EXEC is per half too: with lanes 0..39 active, lanes 32..39 read upper-half lanes 40..47,
/// which are inactive, and get 0 although lower-half lanes 8..15 are active. Lanes 40..63
/// keep their v3.
#[test]
fn a_wave64_backward_permute_reads_exec_within_each_half() {
    let body = [
        0x4A04_00FF,
        0x0000_5000, // v_add_nc_u32 v2, 0x5000, v0
        0x4A06_00FF,
        0xCAFE_0000, // v_add_nc_u32 v3, 0xcafe0000, v0
        0x4A02_0088, // v_add_nc_u32 v1, 8, v0
        0x3602_029F, // v_and_b32 v1, 31, v1
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xDACC_0000,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    ];
    let Some(got) = permuted(&body, true) else { return };
    let want: Vec<u32> = (0..64)
        .map(|l| match l {
            0..32 => 0x5000 + ((l + 8) & 31),
            32..40 => 0,
            _ => 0xCAFE_0000 + l,
        })
        .collect();
    assert_eq!(got, want);
    eprintln!("ran on the hardware device: inactive lanes of the same half read 0");
}

/// A wave32 backward permute names lanes by bits 6:2 of the address: the high bits (0x100)
/// and sub-dword bits (+3) are dropped.
#[test]
fn a_wave32_backward_permute_drops_the_high_and_sub_dword_address_bits() {
    let body = [
        0x4A04_00FF,
        0x0000_3000, // v_add_nc_u32 v2, 0x3000, v0
        0x4A02_0088, // v_add_nc_u32 v1, 8, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x4A02_02FF,
        0x0000_0103, // v_add_nc_u32 v1, 0x103, v1
        0xDACC_0000,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2
    ];
    let Some(got) = permuted(&body, false) else { return };
    let want: Vec<u32> = (0..32).map(|l| 0x3000 + ((l + 8) & 31)).collect();
    assert_eq!(got, want);
}

/// A console wave32 program with console lanes (`v_mbcnt`) in a 32 x 2 group: thread (x, y)
/// stores the group-info SGPR at dword 32y + x. Every dword must be written once and carry
/// the console wave for its relabelled ID. A 32-lane host need not give a host wave a 32-aligned run
/// of flat ids (2-D groups are filled from 16 x 2 tiles on NVIDIA and RDNA), so the first flat
/// id of a wave rounded down gave two waves the same ids (issue #1243); the waves now take
/// their threads from a group counter. With `VICTORY_THREAD_TICKET=0` (the old rounding) the
/// count of dwords written is printed, not checked: it depends on the card.
#[test]
fn the_console_threads_of_a_two_dimensional_group_are_each_given_once() {
    let words = [
        0xD765_0003, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x0001_007E,
        0x3408_0285, // v_lshlrev_b32 v4, 5, v1
        0x4A08_0104, // v_add_nc_u32 v4, v4, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E0A_0206, // v_mov_b32 v5, s6 (group info)
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4, // buffer_store_dword v5, v4, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, false, 64);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: 5 << 1 | 1 << 7 | 1 << 10 | 1 << 11, // group X in s5, group info in s6, thread x/y
        threads: [32, 2, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lower_with = |ticket: bool| {
        // SAFETY: tests that read this switch hold ONE_AT_A_TIME
        unsafe { std::env::set_var("VICTORY_THREAD_TICKET", if ticket { "1" } else { "0" }) };
        let shader = lowered_on(&native, &p, &HashSet::new(), 32);
        unsafe { std::env::remove_var("VICTORY_THREAD_TICKET") };
        shader.expect("lower")
    };
    let shader = lower_with(true);
    let old = lower_with(false);
    assert_ne!(shader.dxil, old.dxil, "the counter is used for a group of two host waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let run = |shader: &HostShader| -> Option<Vec<u32>> {
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("the hardware device does not take this program: {e}; not run");
                return None;
            }
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        Some(out[0][..64].to_vec())
    };
    let Some(got) = run(&shader) else { return };
    let expected: Vec<_> = (0_u32..64).map(|thread| 2 | ((thread / 32) << 6)).collect();
    assert_eq!(got, expected, "group info must use the relabelled console wave");
    if let Some(before) = run(&old) {
        let written = before.iter().filter(|&&d| d != 0).count();
        eprintln!("the old rounding on this device: {written} of 64 dwords written");
    }
    eprintln!("ran on the hardware device: 64 console threads with the expected wave indices");
}

#[test]
fn readfirstlane_uses_both_halves_of_the_console_wave() {
    let mut words = vec![
        sop1(0x04, 12, EXEC),    // save EXEC
        vopc(0xD3, 0, 0),       // v_cmpx_le_u32 s0, v0
        vop1(0x02, 6, v(0)),    // v_readfirstlane_b32 s6, v0
        sop1(0x04, EXEC, 12),   // restore EXEC for the stores
        vop1(0x01, 2, 6),       // v_mov_b32 v2, s6
    ];
    words.extend(stores([2, 0, 0, 0]));
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("split wave64 first-lane read");
    let _lock = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for threshold in [0, 16, 32, 40, 63] {
        let Some(actual) = run_lanes(&shader, [threshold, 0, 0, 0], 256) else { return };
        let want: Vec<_> = (0..64).flat_map(|n| [threshold, n, n, n]).collect();
        same_as_console(&actual, &want);
    }
}

/// Synthetic 3-level BVH (`box32` root -> 2 `box16` + 2 `box32` children -> 4
/// `triangle` leaves with known geometry) plus a CPU traversal oracle using
/// [`crate::bvh::intersect`].
fn synthetic_bvh_scene() -> (Vec<u32>, u32) {
    use crate::bvh::{self, tests as nodes};

    let mut words: Vec<u32> = Vec::new();
    let add = |words: &mut Vec<u32>, node: [u32; 28], dwords: usize, kind: u32| -> u32 {
        let start_bytes = (words.len() * 4) as u32;
        let take = dwords.min(node.len());
        words.extend_from_slice(&node[..take]);
        if dwords > take {
            words.resize(words.len() + (dwords - take), 0);
        }
        (start_bytes >> 3) | kind
    };

    // Reserve 32 dwords at offset 0 for the root box32 node (filled last).
    words.resize(32, 0);

    // Leaf triangles at known depths & positions (clockwise seen from z=0 looking +z).
    // Tri 0 (front-left, z=4): x in [-3, -1], y in [-1, 1]
    let t0 = [[-2.0, 1.0, 4.0], [-1.0, -1.0, 4.0], [-3.0, -1.0, 4.0]];
    // Tri 1 (back-left behind t0, z=8): x in [-3, -1], y in [-1, 1]
    let t1 = [[-2.0, 1.0, 8.0], [-1.0, -1.0, 8.0], [-3.0, -1.0, 8.0]];
    // Tri 2 (center, z=5): x in [-1, 1], y in [-1, 1]
    let t2 = [[0.0, 1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, -1.0, 5.0]];
    // Tri 3 (right, z=6): x in [1, 3], y in [-1, 1]
    let t3 = [[2.0, 1.0, 6.0], [3.0, -1.0, 6.0], [1.0, -1.0, 6.0]];

    let p_t0 = add(&mut words, nodes::triangles(&t0, 9), 16, 0);
    let p_t1 = add(&mut words, nodes::triangles(&t1, 9), 16, 0);
    let p_t2 = add(&mut words, nodes::triangles(&t2, 9), 16, 0);
    let p_t3 = add(&mut words, nodes::triangles(&t3, 9), 16, 0);

    let off = [f32::NAN; 6];
    let b_t0 = [-3.0, -1.0, 3.5, -1.0, 1.0, 4.5];
    let b_t1 = [-3.0, -1.0, 7.5, -1.0, 1.0, 8.5];
    let b_t2 = [-1.0, -1.0, 4.5, 1.0, 1.0, 5.5];
    let b_t3 = [1.0, -1.0, 5.5, 3.0, 1.0, 6.5];

    // Child 0 (box16): holds t0 (near) and t1 (far behind t0).
    let p_left = add(
        &mut words,
        nodes::box16([p_t0, p_t1, bvh::NONE, bvh::NONE], [b_t0, b_t1, off, off]),
        16,
        4,
    );
    // Child 1 (box32): holds t2 (center).
    let p_mid = add(
        &mut words,
        nodes::box32([p_t2, bvh::NONE, bvh::NONE, bvh::NONE], [b_t2, off, off, off]),
        32,
        5,
    );
    // Child 2 (box16): holds t3 (right).
    let p_right = add(
        &mut words,
        nodes::box16([p_t3, bvh::NONE, bvh::NONE, bvh::NONE], [b_t3, off, off, off]),
        16,
        4,
    );

    let root = nodes::box32(
        [p_left, p_mid, p_right, bvh::NONE],
        [
            [-3.0, -1.0, 3.5, -1.0, 1.0, 8.5],
            [-1.0, -1.0, 4.5, 1.0, 1.0, 5.5],
            [1.0, -1.0, 5.5, 3.0, 1.0, 6.5],
            off,
        ],
    );
    words[..28].copy_from_slice(&root);
    let root_ptr = 5u32; // offset 0, kind 5 (box32)
    (words, root_ptr)
}

/// CPU reference traversal matching [`wave64_bvh_traversal_program`]: pops a
/// node from a per-thread stack, invokes [`crate::bvh::intersect`], pushes
/// valid box children (farthest first so nearest is popped first), and keeps
/// the first/closest triangle hit (`[t, det, i, j]` or `MISS`).
fn cpu_traverse(bvh_words: &[u32], root_ptr: u32, ray: &crate::bvh::Ray, steps: u32) -> [u32; 4] {
    use crate::bvh;
    let mut stack = vec![root_ptr];
    let mut best = bvh::MISS.map(f32::to_bits);
    let mut r = *ray;
    for _ in 0..steps {
        let Some(node_ptr) = stack.pop() else {
            break;
        };
        let kind = node_ptr & 7;
        let off_words = (((node_ptr >> 3) << 6) / 4) as usize;
        let node: [u32; 28] = std::array::from_fn(|k| bvh_words.get(off_words + k).copied().unwrap_or(0));
        let res = bvh::intersect(&node, kind, &r, true);
        if kind >= 4 {
            // Push farthest-first so nearest hit is popped next.
            for &child in res.iter().rev() {
                if child != bvh::NONE {
                    stack.push(child);
                }
            }
        } else if res[0] != f32::INFINITY.to_bits() {
            let t = f32::from_bits(res[0]) / f32::from_bits(res[1]);
            if t < r.extent {
                r.extent = t;
                best = res;
            }
        }
    }
    best
}

/// Builds a wave64 (`threads = [64, 1, 1]`, 4 KiB LDS) multi-level BVH
/// traversal compute shader using `image_bvh64_intersect_ray` and a per-thread
/// bank-conflict-free SoA LDS traversal stack (`addr = (sp << 8) + (v0 << 2)`
/// maintained via `v22 = v0 << 2`, `v22 += 256` on push, `v22 -= 256` on pop).
///
/// - When `divergent` is `true`, the traversal loop uses `s_and_saveexec_b64` +
///   `s_cbranch_execz` around the stack pop/push (a lane-branch region reading
///   and writing LDS, as real RT traversal shaders emit).
/// - When `divergent` is `false`, every lane runs a fixed `s0`-step uniform
///   loop with per-lane `exec` guards around the stack ops (so `origin/main`
///   lowers it to wave32 with `split_lds = true`, allowing a direct apples-to-
///   apples GPU benchmark on the RTX 5070 Ti).
/// - When `with_wave_op` is `true`, a cross-half wave instruction
///   (`v_mbcnt_lo_u32_b32`) runs in entry so `console_lanes` and `crosses` are
///   both set, isolating `split_lds` barrier removal while keeping `halves` +
///   `exchange` + `WaveSize(32)`.
fn wave64_bvh_traversal_program(divergent: bool, with_wave_op: bool) -> NativeProgram {
    // Registers:
    // s0: step count
    // s4: descriptor table byte offset (slot 0: ray SRV, slot 1: out UAV, slot 2: BVH SRV)
    // s5: group_id.x (TGID_X_EN)
    // v[1:12]: image_bvh64_intersect_ray operands (v1=ptr_lo, v2=0, v3=extent, v4..v12=ray)
    // v[13:16]: image_bvh64_intersect_ray result
    // v[24:27]: closest hit accumulator (init to bvh::MISS)
    // v22: per-thread SoA LDS stack byte offset (init v0 << 2, +/- 256 per push/pop)
    // v23: stack depth (init 1)
    let mut w: Vec<u32> = vec![
        0xBE86_0305, // s_mov_b32 s6, s5 (group_id.x)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (ray records SRV)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (output UAV)
        0xF408_0402,
        0xFA00_0040, // s_load_dwordx4 s[16:19], s[4:5], 0x40 (BVH buffer SRV)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    if with_wave_op {
        // Cross-half wave op (`console_lanes = true`, `crosses = true`), result unused in addr.
        w.extend_from_slice(&[
            0xD765_001C,
            0x0001_007E, // v_mbcnt_lo_u32_b32 v28, exec_lo, 0
        ]);
    }
    w.extend_from_slice(&[
        // global_tid = group_id.x * 64 + v0
        0x7E28_0206, // v_mov_b32 v20, s6
        0x3428_2886, // v_lshlrev_b32 v20, 6, v20
        0x4A28_2900, // v_add_nc_u32 v20, v0, v20
        0x342A_2886, // v_lshlrev_b32 v21, 6, v20 (global_tid * 64)
        0x343A_2884, // v_lshlrev_b32 v29, 4, v20 (global_tid * 16 for output store)
        // Load ray record into v[1:12]
        0xE038_1000,
        0x8002_0115, // buffer_load_dwordx4 v[1:4], v21, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0515, // buffer_load_dwordx4 v[5:8], v21, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0915, // buffer_load_dwordx4 v[9:12], v21, s[8:11], 0 offen offset:32
        0xBF8C_0F70, // s_waitcnt vmcnt(0)
        // Init closest hit v[24:27] = MISS (+inf, 1.0, 0.0, 0.0)
        0x7E30_02FF,
        f32::INFINITY.to_bits(), // v_mov_b32 v24, +inf
        0x7E32_02F2,             // v_mov_b32 v25, 1.0
        0x7E34_0280,             // v_mov_b32 v26, 0
        0x7E36_0280,             // v_mov_b32 v27, 0
        // Init per-thread SoA LDS stack: v22 = v0 << 2, push root (v1), v22 += 256, v23 = 1
        0x342C_0082, // v_lshlrev_b32 v22, 2, v0
    ]);
    w.extend_from_slice(&ds(DS_WRITE_B32, 0, 22, 1)); // ds_write_b32 v22, v1
    w.extend_from_slice(&[
        0x4A2C_2CFF,
        256,         // v_add_nc_u32 v22, 256, v22
        0x7E2E_0281, // v_mov_b32 v23, 1 (sp = 1)
        0xBE96_0300, // s_mov_b32 s22, s0 (step counter)
    ]);

    // Build traversal step body (used inside loop).
    let mut step_body: Vec<u32> = Vec::new();
    // Save exec in s[24:25], mask active lanes with sp > 0 (0 < v23)
    step_body.extend_from_slice(&[
        0x7D82_2E80, // v_cmp_lt_u32 vcc, 0, v23
        0xBE98_246A, // s_and_saveexec_b64 s[24:25], vcc
    ]);
    // Inner body: pop node, intersect, push children or record hit.
    let mut inner: Vec<u32> = Vec::new();
    // Pop: v22 -= 256, v23 -= 1, ds_read_b32 v1, v22
    inner.extend_from_slice(&[
        0x4E2C_2CFF,
        256,         // v_subrev_nc_u32 v22, 256, v22
        0x4E2E_2E81, // v_subrev_nc_u32 v23, 1, v23
    ]);
    inner.extend_from_slice(&ds(DS_READ_B32, 1, 22, 0)); // ds_read_b32 v1, v22
    inner.extend_from_slice(&[
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        // image_bvh64_intersect_ray v[13:16], [v1..v12], s[16:19]
        0xF19C_9F07,
        0x0004_0D01,
        0x0504_0302,
        0x0908_0706,
        0x000C_0B0A,
        0xBF8C_0F70, // s_waitcnt vmcnt(0)
        // Check box vs triangle: v17 = v1 & 7; box iff 4 <= v17 (3 < v17)
        0x3622_0287, // v_and_b32 v17, 7, v1
        0x7D82_2283, // v_cmp_lt_u32 vcc, 3, v17 (box?)
        0xBE9A_246A, // s_and_saveexec_b64 s[26:27], vcc
    ]);
    // For box lanes: push v16, v15, v14, v13 if != NONE (0xffff_ffff)
    for child_reg in [16u8, 15, 14, 13] {
        // v_cmp_ne_u32 vcc, -1, v<child_reg>
        inner.push(0x7D8A_00C1 | (u32::from(child_reg) << 9));
        inner.push(0xBE9C_246A); // s_and_saveexec_b64 s[28:29], vcc
        inner.extend_from_slice(&ds(DS_WRITE_B32, 0, 22, child_reg));
        inner.extend_from_slice(&[
            0x4A2C_2CFF,
            256,         // v_add_nc_u32 v22, 256, v22
            0x4A2E_2E81, // v_add_nc_u32 v23, 1, v23
            0xBEFE_041C, // s_mov_b64 exec, s[28:29]
        ]);
    }
    // Switch to triangle lanes: s_andn2_b64 exec, s[26:27], exec
    inner.extend_from_slice(&[
        0x8AFE_7E1A, // s_andn2_b64 exec, s[26:27], exec
        // Hit check: v13 != +inf
        0x7D8A_1AFF,
        f32::INFINITY.to_bits(), // v_cmp_ne_u32 vcc, +inf, v13
        0xBE9C_246A,             // s_and_saveexec_b64 s[28:29], vcc
        // t = v13 * rcp(v14) (exact for power-of-two denominators in our scene)
        0x7E24_550E, // v_rcp_f32 v18, v14
        0x1024_1B12, // v_mul_f32 v18, v18, v13 (t = v13 * rcp(v14))
        // Keep closest: v18 < v3 (extent)
        0x7C02_0712, // v_cmp_lt_f32 vcc, v18, v3
        0xBE9E_246A, // s_and_saveexec_b64 s[30:31], vcc
        0x7E06_0312, // v_mov_b32 v3, v18 (update extent)
        0x7E30_030D, // v_mov_b32 v24, v13
        0x7E32_030E, // v_mov_b32 v25, v14
        0x7E34_030F, // v_mov_b32 v26, v15
        0x7E36_0310, // v_mov_b32 v27, v16
        0xBEFE_041E, // s_mov_b64 exec, s[30:31]
        0xBEFE_041C, // s_mov_b64 exec, s[28:29]
        0xBEFE_041A, // s_mov_b64 exec, s[26:27]
    ]);
    if divergent {
        // Explicit branch skipping inner when no lane has sp > 0 -> creates a
        // lane-branch region containing both ds_read_b32 and ds_write_b32.
        step_body.push(0xBF88_0000 | inner.len() as u32); // s_cbranch_execz +inner.len()
    }
    step_body.extend_from_slice(&inner);
    step_body.extend_from_slice(&[
        0xBEFE_0418, // s_mov_b64 exec, s[24:25]
        0x8096_8116, // s_sub_u32 s22, s22, 1
        0xBF08_8016, // s_cmp_gt_u32 s22, 0
    ]);
    let back_dwords = (step_body.len() + 1) as i16;
    step_body.push(0xBF85_0000 | u32::from((-back_dwords) as u16)); // s_cbranch_scc1 loop_top

    w.extend_from_slice(&step_body);
    w.extend_from_slice(&[
        // Store closest hit v[24:27] at global_tid * 16
        0xE078_1000,
        0x8003_181D, // buffer_store_dwordx4 v[24:27], v29, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);

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
    NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
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
        used: vec![
            table(ResourceType::Srv, 0),
            table(ResourceType::Uav, 8),
            table(ResourceType::Srv, 16),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | (1 << 7) | (8 << 15), // s0..s4 user, s5 = group_id.x, 4 KiB LDS (depth up to 16)
            threads: [64, 1, 1],
        }),
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// Builds a deeper 4-level overlapping synthetic BVH so benchmark rays exercise
/// 10-16 active stack pops/pushes per ray (stack depth up to 8 entries) with
/// realistic hit/miss/occlusion geometry.
fn synthetic_bvh_scene_deep() -> (Vec<u32>, u32) {
    use crate::bvh::{self, tests as nodes};

    let mut words: Vec<u32> = Vec::new();
    let add = |words: &mut Vec<u32>, node: [u32; 28], dwords: usize, kind: u32| -> u32 {
        let start_bytes = (words.len() * 4) as u32;
        let take = dwords.min(node.len());
        words.extend_from_slice(&node[..take]);
        if dwords > take {
            words.resize(words.len() + (dwords - take), 0);
        }
        (start_bytes >> 3) | kind
    };
    words.resize(32, 0);
    let off = [f32::NAN; 6];

    // 4 depth layers along z = 4, 6, 8, 10; each layer has 4 x-bands covering [-4, 4].
    // Rays aimed at x in [-3, 3] intersect all 4 overlapping mid-boxes along z,
    // pushing multiple children onto the per-thread LDS stack before culling by extent.
    let mut mid_ptrs = [bvh::NONE; 4];
    let mut mid_boxes = [off; 4];
    for layer in 0..4usize {
        let z = 4.0 + 2.0 * (layer as f32);
        let mut leaf_ptrs = [bvh::NONE; 4];
        let mut leaf_boxes = [off; 4];
        for band in 0..4usize {
            let x0 = -3.0 + 1.5 * (band as f32);
            let x1 = x0 + 1.5;
            let xm = 0.5 * (x0 + x1);
            // Alternate triangle y-coverage so some layers miss the triangle inside its AABB
            // and keep traversing deeper layers.
            let y_lo = if (layer + band) % 2 == 0 { -1.0 } else { 0.25 };
            let y_hi = if (layer + band) % 2 == 0 { 1.0 } else { 1.0 };
            let tri = [[xm, y_hi, z], [x1, y_lo, z], [x0, y_lo, z]];
            leaf_ptrs[band] = add(&mut words, nodes::triangles(&tri, 9), 16, 0);
            leaf_boxes[band] = [x0 - 0.1, -1.1, z - 0.5, x1 + 0.1, 1.1, z + 0.5];
        }
        // Alternate box16 and box32 internal nodes.
        let (node, dwords, kind) = if layer % 2 == 0 {
            (nodes::box16(leaf_ptrs, leaf_boxes), 16, 4)
        } else {
            (nodes::box32(leaf_ptrs, leaf_boxes), 32, 5)
        };
        mid_ptrs[layer] = add(&mut words, node, dwords, kind);
        mid_boxes[layer] = [-3.1, -1.1, z - 0.5, 3.1, 1.1, z + 0.5];
    }
    let root = nodes::box32(mid_ptrs, mid_boxes);
    words[..28].copy_from_slice(&root);
    (words, 5u32)
}

/// Lowers `native` for a 32-lane host with `VICTORY_LDS_PRIVATE` set to `enabled`.
fn lower_wave32_with_private(
    native: &NativeProgram,
    enabled: bool,
) -> (lower::Lowered, Vec<resource::RootIndexedView>, bool) {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
    let io = io::raise(&p, native).expect("io");
    let parts = lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    };
    let is_private = super::lds_private::is_lane_private(&p, &st, native);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let was_enabled = super::TEST_LANE_PRIVATE_LDS.replace(Some(enabled));
    let lowered = lower::lower(&parts);
    super::TEST_LANE_PRIVATE_LDS.set(was_enabled);
    lower::TEST_HOST_WAVE.set(was);
    (lowered.expect("lower"), res.root_indexed_views, is_private)
}

/// Builds ray records for `count` threads over [`synthetic_bvh_scene`]: mixes
/// hits on all 4 triangles (including occluded front/back hits on `t0`/`t1`),
/// misses beside the boxes, and back-facing hits.
fn synthetic_traversal_rays(count: usize, root_ptr: u32) -> (Vec<u32>, Vec<crate::bvh::Ray>) {
    use crate::bvh::Ray;
    let templates = [
        // Hits t0 (z=4, occludes t1 at z=8)
        Ray::new([-2.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0),
        // Hits t2 (z=5, center)
        Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0),
        // Hits t3 (z=6, right)
        Ray::new([2.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0),
        // Misses completely (outside all boxes)
        Ray::new([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0),
        // Hits t1 from behind (z=10 looking -z, occludes t0 at z=4)
        Ray::new([-2.0, 0.0, 10.0], [0.0, 0.0, -1.0], 100.0),
        // Hits box of t2 but misses triangle t2 (corner of its AABB)
        Ray::new([0.9, 0.9, 0.0], [0.0, 0.0, 1.0], 100.0),
        // Short extent misses t3 (z=6)
        Ray::new([2.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0),
        // Angled ray hitting t0 near vertex 1
        Ray::new([-1.5, -0.5, 0.0], [0.0, 0.0, 1.0], 100.0),
    ];
    let mut records = vec![0u32; count * 16];
    let mut rays = Vec::with_capacity(count);
    for t in 0..count {
        let base = templates[t % templates.len()];
        // Tiny deterministic jitter that keeps hit/miss class distinct across lanes.
        let jitter = ((t % 17) as f32 - 8.0) * 0.01;
        let ray = Ray::new(
            [base.origin[0] + jitter, base.origin[1] - jitter * 0.5, base.origin[2]],
            base.direction,
            base.extent,
        );
        let r = &mut records[t * 16..t * 16 + 16];
        r[0] = root_ptr;
        r[1] = 0;
        r[2] = ray.extent.to_bits();
        for a in 0..3 {
            r[3 + a] = ray.origin[a].to_bits();
            r[6 + a] = ray.direction[a].to_bits();
            r[9 + a] = ray.inverse[a].to_bits();
        }
        rays.push(ray);
    }
    (records, rays)
}

/// A console wave64 multi-level BVH traversal shader with a per-thread SoA LDS
/// stack inside a divergent traversal loop must lower cleanly and exactly to
/// wave32 on 32-lane hardware (RTX 5070 Ti), without `loose_lds` approximation
/// or `split_lds` group barriers, and match the CPU BVH traversal oracle bit
/// for bit across hits, misses, and occluders.
#[test]
fn wave64_bvh_traversal_with_private_lds_stack_runs_on_wave32_without_barriers() {
    let (bvh_words, root_ptr) = synthetic_bvh_scene();
    let (records, rays) = synthetic_traversal_rays(64, root_ptr);
    let steps = 8u32;

    for (label, divergent, with_wave_op) in [
        ("divergent_loop", true, false),
        ("uniform_loop", false, false),
        ("divergent_loop_with_wave_op", true, true),
        ("uniform_loop_with_wave_op", false, true),
    ] {
        let native = wave64_bvh_traversal_program(divergent, with_wave_op);
        // Confirm legacy behavior reproduces the failure / barrier overhead before our pass.
        let (legacy, _, is_private) = lower_wave32_with_private(&native, false);
        assert!(is_private, "{label}: expected lds_lane_private == true");
        if divergent {
            assert_eq!(
                legacy.wave_size,
                Some(64),
                "{label}: legacy lowering without lds_lane_private should refuse wave32"
            );
        } else {
            assert!(
                legacy.barriers >= 6,
                "{label}: legacy lowering should emit split_lds barriers (got {})",
                legacy.barriers
            );
        }

        let (lowered, root_indexed_views, _) = lower_wave32_with_private(&native, true);
        if with_wave_op {
            assert_eq!(
                lowered.wave_size,
                Some(32),
                "{label}: cross-half wave op with private LDS must lower to WaveSize(32)"
            );
            if !divergent {
                assert!(
                    lowered.barriers < legacy.barriers,
                    "{label}: private LDS must eliminate split_lds barriers ({} vs {})",
                    lowered.barriers,
                    legacy.barriers
                );
            }
        } else {
            assert_eq!(
                lowered.wave_size, None,
                "{label}: private LDS without wave ops must not force a WaveSize"
            );
            assert_eq!(
                lowered.barriers, 0,
                "{label}: private LDS without wave ops must emit zero dx.op.barrier calls"
            );
        }
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views,
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("{label}: DXIL validation failed: {e}"));
        }
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut ran = 0;
        for (dev_label, warp) in [("hardware", false), ("warp", true)] {
            // WARP does not accept explicit `WaveSize(32)` pipelines (only `WaveSize(None)`).
            if warp && lowered.wave_size.is_some() {
                continue;
            }
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            if !warp {
                eprintln!("wave64_bvh_traversal ({label}) adapter: {:?}", host.adapter_name());
            }
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label}/{dev_label}: compute_pipeline failed: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(5) as usize];
            ud[0] = steps;
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[
                            super::host::View::DeviceRawSrv(&records),
                            View::RawUav(64 * 4),
                            super::host::View::DeviceRawSrv(&bvh_words),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}/{dev_label}: dispatch failed: {e}"));
            for t in 0..64usize {
                let want = cpu_traverse(&bvh_words, root_ptr, &rays[t], steps);
                let got = &out[0][4 * t..4 * t + 4];
                assert_eq!(
                    got, want,
                    "{label}/{dev_label} thread {t}: got {got:x?} want {want:x?}"
                );
            }
            ran += 1;
        }
        assert!(ran > 0, "no SM 6.6 device available");
    }
}

/// Builds a small wave64 shader with 2 KiB LDS and a divergent `s_and_saveexec_b64` +
/// `s_cbranch_execz` branch around `body` (which uses LDS), then stores `v5` per thread.
fn divergent_lds_probe(body: &[u32], threads: [u32; 3], rsrc2_extra: u32) -> NativeProgram {
    let mut w = vec![
        0x7D88_0080, // v_cmp_gt_u32 vcc, s0, v0 (v0 < s0)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz +body.len()
    ];
    w.extend_from_slice(body);
    w.extend_from_slice(&[
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 1, // buffer_store_dword v5, v1, s[8:11], 0 offen
    ]);
    let mut prog = native(&w);
    if let StageIo::Cs(cs) = &mut prog.io {
        cs.threads = threads;
        cs.rsrc2 |= (4 << 15) | rsrc2_extra; // 2 KiB LDS
    }
    prog
}

/// Positive tests for `lds_lane_private`: AoS per-thread slice stacks, 2D `[8, 8, 1]`
/// groups with flattened thread IDs, SoA `(sp << 8) + (v0 << 2)`, and odd stride `v0 * 12`.
#[test]
fn lds_lane_private_accepts_soa_aos_and_2d_group_stacks() {
    // 1. AoS per-thread 32-byte slice: addr = (v0 << 5) | ((s1 & 7) << 2)
    let mut aos = vec![
        0x3404_0085, // v_lshlrev_b32 v2, 5, v0
        0x3606_0287, // v_and_b32 v3, 7, v1 (uninit/runtime index masked to 0..7)
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
        0x3804_0702, // v_or_b32 v2, v2, v3
        0x4A08_0081, // v_add_nc_u32 v4, 1, v0
    ];
    aos.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    aos.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));
    let aos_prog = divergent_lds_probe(&aos, [64, 1, 1], 0);
    let (lowered, _, is_priv) = lower_wave32_with_private(&aos_prog, true);
    assert!(is_priv, "AoS slice (v0 << 5) | ((sp & 7) << 2) must be lane-private");
    assert_eq!(lowered.wave_size, None);
    assert_eq!(lowered.barriers, 0);

    // 2. 2D [8, 8, 1] group SoA stack: flat = (v1 << 3) | v0; addr = (flat << 2) + (s1 << 8)
    let mut d2 = vec![
        0x3404_0283, // v_lshlrev_b32 v2, 3, v1
        0x3804_0102, // v_or_b32 v2, v2, v0 (flat = (v1 << 3) | v0)
        0x3404_0482, // v_lshlrev_b32 v2, 2, v2 (flat << 2)
        0x8F03_8801, // s_lshl_b32 s3, s1, 8
        0x4A04_0403, // v_add_nc_u32 v2, s3, v2
        0x4A08_0081, // v_add_nc_u32 v4, 1, v0
    ];
    d2.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    d2.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));
    // rsrc2 bit 11 = TIDIG_COMP_CNT = 1 (v0 and v1 enabled)
    let d2_prog = divergent_lds_probe(&d2, [8, 8, 1], 1 << 11);
    let (lowered_2d, _, is_priv_2d) = lower_wave32_with_private(&d2_prog, true);
    assert!(is_priv_2d, "2D [8, 8, 1] flattened SoA stack must be lane-private");
    assert_eq!(lowered_2d.wave_size, None);
    assert_eq!(lowered_2d.barriers, 0);

    // 3. Odd-multiplied stride: addr = v0 * 12 (odd = 3, sh = 2, bijection mod 256)
    let mut odd = vec![
        0x1604_008C, // v_mul_u32_u24 v2, 12, v0
        0x4A08_0081, // v_add_nc_u32 v4, 1, v0
    ];
    odd.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    odd.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));
    let odd_prog = divergent_lds_probe(&odd, [64, 1, 1], 0);
    let (lowered_odd, _, is_priv_odd) = lower_wave32_with_private(&odd_prog, true);
    assert!(is_priv_odd, "odd stride v0 * 12 must be lane-private");
    assert_eq!(lowered_odd.wave_size, None);
    assert_eq!(lowered_odd.barriers, 0);
}

/// Negative soundness tests for `lds_lane_private`: every shared, aliasing, atomic,
/// overlapping-offset, or unbounded-stride LDS pattern must be rejected (`false`)
/// and keep `WaveSize(64)` when inside a divergent read+write branch.
#[test]
fn lds_lane_private_rejects_shared_and_unsafe_patterns() {
    // 1. Constant address 0 (shared across all lanes)
    let mut shared_const = vec![0x7E04_0280, 0x4A08_0081]; // v2 = 0, v4 = v0 + 1
    shared_const.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    shared_const.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));

    // 2. Cross-half alias: (v0 ^ 32) << 2 read alongside v0 << 2 write
    let mut xor_alias = vec![
        0x3404_0082, // v2 = v0 << 2
        0x3A06_00A0, // v_xor_b32 v3, 32, v0
        0x3406_0682, // v3 = (v0 ^ 32) << 2
        0x4A08_0081,
    ];
    xor_alias.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    xor_alias.extend_from_slice(&ds(DS_READ_B32, 5, 3, 0));

    // 3. 32-lane wrap alias: (v0 & 31) << 2 (lanes 0..31 collide with 32..63)
    let mut mask31 = vec![
        0x3604_009F, // v_and_b32 v2, 31, v0
        0x3404_0482, // v2 <<= 2
        0x4A08_0081,
    ];
    mask31.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    mask31.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));

    // 4. Overlapping static offset: write at v0 << 2, read at (v0 << 2) + 4
    let mut overlap_off = vec![0x3404_0082, 0x4A08_0081];
    overlap_off.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    // ds_read_b32 v5, v2 offset:4
    overlap_off.extend_from_slice(&[0x36 << 26 | (DS_READ_B32 as u32) << 18 | 4, (5 << 24) | 2]);

    // 5. Mixed strides: write at v0 << 2, read at v0 << 3
    let mut mixed_sh = vec![0x3404_0082, 0x3406_0083, 0x4A08_0081];
    mixed_sh.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    mixed_sh.extend_from_slice(&ds(DS_READ_B32, 5, 3, 0));

    // 6. AoS slice overflow: (v0 << 5) | ((v1 & 15) << 2) -> max 60 + 4 = 64 > 32
    let mut aos_overflow = vec![
        0x3404_0085, // v2 = v0 << 5
        0x3606_028F, // v3 = v1 & 15
        0x3406_0682, // v3 <<= 2
        0x3804_0702, // v2 |= v3
        0x4A08_0081,
    ];
    aos_overflow.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    aos_overflow.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));

    // 7. LDS atomic (ds_add_u32 op = 0x00) even at v0 << 2
    let mut atomic = vec![0x3404_0082, 0x4A08_0081];
    atomic.extend_from_slice(&ds(0x00, 0, 2, 4)); // ds_add_u32 v2, v4
    atomic.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));

    // 8. 2D [8, 8, 1] group using only v0 << 2 (thread (0,0) collides with (0,1))
    let mut d2_bad = vec![0x3404_0082, 0x4A08_0081];
    d2_bad.extend_from_slice(&ds(DS_WRITE_B32, 0, 2, 4));
    d2_bad.extend_from_slice(&ds(DS_READ_B32, 5, 2, 0));

    let cases: [(&str, NativeProgram); 8] = [
        ("shared_const", divergent_lds_probe(&shared_const, [64, 1, 1], 0)),
        ("xor_half_alias", divergent_lds_probe(&xor_alias, [64, 1, 1], 0)),
        ("mask31_alias", divergent_lds_probe(&mask31, [64, 1, 1], 0)),
        ("overlap_offset4", divergent_lds_probe(&overlap_off, [64, 1, 1], 0)),
        ("mixed_shift", divergent_lds_probe(&mixed_sh, [64, 1, 1], 0)),
        ("aos_overflow", divergent_lds_probe(&aos_overflow, [64, 1, 1], 0)),
        ("ds_atomic", divergent_lds_probe(&atomic, [64, 1, 1], 0)),
        ("2d_v0_only", divergent_lds_probe(&d2_bad, [8, 8, 1], 1 << 11)),
    ];
    for (name, prog) in cases {
        let (lowered, _, is_priv) = lower_wave32_with_private(&prog, true);
        assert!(!is_priv, "{name}: unsafe LDS pattern must be rejected by lds_lane_private");
        assert_eq!(
            lowered.wave_size,
            Some(64),
            "{name}: rejected divergent LDS program must keep WaveSize(64)"
        );
    }
}

/// GPU timestamp benchmark comparing wave64 BVH traversal with a per-thread SoA
/// LDS stack on the hardware GPU (1,048,576 rays x 32 steps over a 4-level
/// synthetic BVH, verified bit-for-bit against [`cpu_traverse`]).
///
/// Compares:
/// - `uniform_no_waveop`: legacy (`split_lds` + `halves` + `WaveSize(32)`) vs
///   private-LDS (`WaveSize(None)`, zero barriers, zero `halves` atomic).
/// - `uniform_with_waveop`: legacy vs private-LDS with `console_lanes = true`
///   (isolates `split_lds` barrier removal while keeping `halves` + `exchange`).
/// - `divergent_no_waveop`: private-LDS early-exit divergent traversal (refused
///   with `WaveSize(64)` on legacy lowering).
#[test]
#[ignore]
fn bench_wave64_bvh_traversal_lds_stack_on_gpu() {
    const GROUPS: u32 = 16384;
    const TOTAL_THREADS: usize = (GROUPS * 64) as usize;
    const STEPS: u32 = 32;

    let (bvh_words, root_ptr) = synthetic_bvh_scene_deep();
    let (records, rays) = synthetic_traversal_rays(TOTAL_THREADS, root_ptr);
    let expected: Vec<[u32; 4]> = (0..128)
        .map(|t| cpu_traverse(&bvh_words, root_ptr, &rays[t], STEPS))
        .collect();

    let cases: [(&str, bool, bool, bool); 5] = [
        ("uniform_no_waveop_legacy", false, false, false),
        ("uniform_no_waveop_private", false, false, true),
        ("uniform_with_waveop_legacy", false, true, false),
        ("uniform_with_waveop_private", false, true, true),
        ("divergent_no_waveop_private", true, false, true),
    ];

    let mut prepared = Vec::new();
    for (label, divergent, with_wave_op, enable_private) in cases {
        let native = wave64_bvh_traversal_program(divergent, with_wave_op);
        let (lowered, root_indexed_views, _) = lower_wave32_with_private(&native, enable_private);
        let barriers = lowered.barriers;
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views,
        };
        prepared.push((label, lowered.wave_size, barriers, shader));
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).expect("SM 6.6 hardware GPU required");
    let adapter = host.adapter_name().unwrap_or_else(|| "unknown".into());
    eprintln!("WAVE64-LDS-BENCH adapter={adapter} groups={GROUPS} rays={TOTAL_THREADS} steps={STEPS}");

    for (label, wave_size, barriers, shader) in &prepared {
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(5) as usize];
        ud[0] = STEPS;
        let (out, mut times_ms) = host
            .dispatch_timed(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        super::host::View::DeviceRawSrv(&records),
                        View::RawUav((TOTAL_THREADS * 4) as u32),
                        super::host::View::DeviceRawSrv(&bvh_words),
                    ],
                    samplers: &[],
                    groups: [GROUPS, 1, 1],
                },
                10,
                40,
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch_timed failed: {e}"));

        for (t, want) in expected.iter().enumerate() {
            let got = &out[0][4 * t..4 * t + 4];
            assert_eq!(got, want, "{label}: bit mismatch at thread {t}");
        }
        times_ms.sort_by(|a, b| a.total_cmp(b));
        let median = times_ms[times_ms.len() / 2];
        let p95 = times_ms[(times_ms.len() * 95 / 100).min(times_ms.len() - 1)];
        let min = times_ms[0];
        let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        eprintln!(
            "  {label:30} wave={wave_size:?} barriers={barriers:<2} dxil={:5}B | median {median:.4} ms | p95 {p95:.4} ms | min {min:.4} ms | mean {mean:.4} ms",
            shader.dxil.len()
        );
    }
}
