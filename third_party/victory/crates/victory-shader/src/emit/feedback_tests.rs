use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, dxbc, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A wave32 compute program of one thread: 4 root constants in s0..s3 and, in s4, a table
/// with a texture's descriptor at dword 0 and a raw UAV's at dword 8; `body` follows the
/// loads of the texture's descriptor into s[8:15] and the UAV's into s[16:19].
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, 8, 0)
}

/// [`native`] with a view of `dwords` dwords (8: a texture, 4: a buffer of elements of
/// `stride` bytes) at dword 0, loaded into s8 on.
fn native_of(body: &[u32], dwords: u8, stride: u32) -> NativeProgram {
    let load = match dwords {
        8 => 0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], null
        _ => 0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], null
    };
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        load,
        0xFA00_0000,
        0xF408_0402,
        0xFA00_0020, // s_load_dwordx4 s[16:19], s[4:5], 0x20
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 4,
        slot: 1,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
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
        used: vec![
            range(ResourceType::Srv, 0, dwords, stride),
            range(ResourceType::Uav, 8, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` translated as the title's programs are, and validated.
fn translate(native: &NativeProgram) -> HostShader {
    let container = dxbc::build(&[]);
    let c = Container::parse(&container).expect("container");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    let shader = super::emit(&c, native, &p, &st, &io, &res).expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// [`run`] with any views at heap indices 0 on; every UAV's dwords, in order, per device.
fn run_views(shader: &HostShader, s0: u32, views: &[View]) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // The constant is s0; the table starts at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.dword) {
                (RootType::Constants, 0) => s0,
                _ => 0,
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words));
    }
    out
}

/// (Neptune) Bindless usage feedback: with a feedback slot, a texture taken from the view heap
/// stores 1 into dword `index` of the raw buffer at that slot, in a container `dxil.dll`
/// accepts; the program's own work is unchanged.
#[test]
fn a_texture_taken_from_the_heap_is_noted_in_the_feedback_buffer() {
    let program = native(&[
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBE95_03FF,
        0x00C0_0000, // s_mov_b32 s21, 0x00c00000 (levels up to 12)
        0xBE96_0380, // s_mov_b32 s22, 0 (point filtering)
        0xBE97_0380, // s_mov_b32 s23, 0
        0x7E02_02FF,
        0x3E80_0000, // v_mov_b32 v1, 0.25
        0x7E04_02FF,
        0x3F00_0000, // v_mov_b32 v2, 0.5
        0xF09C_0F08,
        0x00A2_0301, // image_sample_lz v[3:6], v[1:2], s[8:15], s[20:23] dmask:0xf 2D
        0xE070_0000,
        0x8004_0300, // buffer_store_dword v3, off, s[16:19], 0
    ]);
    // The feedback buffer at heap slot 2, for this thread's translation only.
    super::shader::FEEDBACK_TEST.with(|f| f.set(Some(2)));
    let shader = translate(&program);
    super::shader::FEEDBACK_TEST.with(|f| f.set(None));
    let plain = translate(&program);
    assert_ne!(shader.dxil, plain.dxil, "the feedback store is in the translation");
    let texels = [[3.0f32, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]];
    for (label, uavs) in run_views(&shader, 0, &[View::Texture(2, 1, &texels), View::RawUav(4), View::RawUav(4)]) {
        assert_eq!(f32::from_bits(uavs[0][0]), 3.0, "{label}: the sample as without feedback");
        // Slot 0 (the texture) noted; slot 1 (a raw buffer, not a texture) not.
        assert_eq!(uavs[1], [1, 0, 0, 0], "{label}: the feedback buffer");
    }
}
