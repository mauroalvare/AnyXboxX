use super::*;

/// DLSS's optimal render sizes for a 2560x1440 output, as the 310 SDK
/// reports them (ultra quality has none).
fn optimal_1440p(mode: u32) -> Option<[u32; 2]> {
    Some(match mode {
        abi::DLSS_DLAA => [2560, 1440],
        abi::DLSS_MAX_QUALITY => [1707, 960],
        abi::DLSS_BALANCED => [1485, 835],
        abi::DLSS_MAX_PERFORMANCE => [1280, 720],
        abi::DLSS_ULTRA_PERFORMANCE => [853, 480],
        _ => return None,
    })
}

#[test]
fn the_title_render_sizes_pick_their_mode() {
    let out = [2560, 1440];
    for (mode, sl) in [
        (Mode::Dlaa, abi::DLSS_DLAA),
        (Mode::Quality, abi::DLSS_MAX_QUALITY),
        (Mode::Balanced, abi::DLSS_BALANCED),
        (Mode::Performance, abi::DLSS_MAX_PERFORMANCE),
        (Mode::UltraPerformance, abi::DLSS_ULTRA_PERFORMANCE),
    ] {
        let render = mode.render_size(out);
        assert_eq!(pick_mode(render, out, optimal_1440p), Some(sl), "{mode:?} {render:?}");
        assert_eq!(mode.sl_mode(), sl);
        assert_eq!(Mode::from_name(mode.name()), Some(mode));
    }
    assert_eq!(Mode::Quality.render_size(out), [1706, 960]);
    assert_eq!(Mode::Balanced.render_size(out), [1484, 835]);
    // Series S: a 1280x720 scene for a 1920x1080 output is quality.
    let s = |mode: u32| optimal_1440p(mode).map(|[w, h]| [w * 3 / 4, h * 3 / 4]);
    assert_eq!(pick_mode([1280, 720], [1920, 1080], s), Some(abi::DLSS_MAX_QUALITY));
    // A size between modes takes the next larger one.
    assert_eq!(pick_mode([1600, 900], out, optimal_1440p), Some(abi::DLSS_MAX_QUALITY));
    // Nothing for a render larger than the output, or empty.
    assert_eq!(pick_mode([2561, 1440], out, optimal_1440p), None);
    assert_eq!(pick_mode([0, 1440], out, optimal_1440p), None);
    assert_eq!(pick_mode([1280, 720], out, |_| None), None);
}

#[test]
fn names_and_presets() {
    assert_eq!(Mode::from_name(" Quality "), Some(Mode::Quality));
    assert_eq!(Mode::from_name("ultra-performance"), Some(Mode::UltraPerformance));
    assert_eq!(Mode::from_name("fast"), None);
    assert_eq!(Mode::Balanced.ffx_mode(), 2);
    assert_eq!(preset("K"), Some(11));
    assert_eq!(preset("default"), Some(0));
    assert_eq!(preset("a"), None);
    assert_eq!(sl_mode_name(abi::DLSS_DLAA), "DLAA");
}

#[test]
fn states_read_as_the_pc_backend_reads_them() {
    assert_eq!(host_state(0xc), Some(0xc0));
    assert_eq!(host_state(0x2), Some(0x8));
    assert_eq!(host_state(0x200), Some(0x10));
    assert_eq!(host_state(0x14), None);
    // Inputs a texture cannot be read in are read as shader resources.
    assert_eq!(input_state(0x14), 0xc0);
    assert_eq!(input_state(0x10), 0xc0);
    assert_eq!(input_state(0x4), 0x40);
    assert_eq!(input_state(0x200), 0x10);
    assert_eq!(output_state(0x2), 0x8);
    assert_eq!(output_state(0x100), 0x8);
}

fn camera(flags: u32) -> Camera {
    Camera {
        jitter: [0.25, -0.5],
        motion_scale: [1706.0, 960.0],
        render: [1706, 960],
        output: [2560, 1440],
        near: 0.1,
        far: 10_000.0,
        fov_y: 1.0,
        reset: true,
        flags,
    }
}

#[test]
fn constants_follow_the_upscaling() {
    let c = camera(FFX_HIGH_DYNAMIC_RANGE | FFX_DEPTH_INVERTED | FFX_AUTO_EXPOSURE).constants([1.0, 1.0], [1.0, 1.0]);
    // Motion vectors in UV space scaled to pixels: Streamline gets them
    // back to [-1, 1].
    assert_eq!(c.mvec_scale, [1.0, 1.0]);
    assert_eq!(c.jitter_offset, [0.25, -0.5]);
    assert_eq!((c.depth_inverted, c.reset, c.camera_motion_included), (abi::TRUE, abi::TRUE, abi::TRUE));
    assert_eq!(c.motion_vectors_jittered, abi::FALSE);
    assert!((c.camera_aspect_ratio - 1706.0 / 960.0).abs() < 1e-6);
    let flipped = camera(0).constants([-1.0, 1.0], [1.0, -2.0]);
    assert_eq!(flipped.jitter_offset, [-0.25, -0.5]);
    assert_eq!(flipped.mvec_scale, [1.0, -2.0]);
    assert_eq!(flipped.depth_inverted, abi::FALSE);
    // Display-resolution motion vectors are scaled by the output size.
    let display = camera(FFX_DISPLAY_RESOLUTION_MOTION_VECTORS);
    assert_eq!(display.motion_size(), [2560, 1440]);
    assert_eq!(display.mvec_scale([1.0, 1.0]), [1706.0 / 2560.0, 960.0 / 1440.0]);
}

/// Points at the near and far planes land at the depth the convention
/// says; the inverse maps clip space back.
#[test]
fn projections_place_near_and_far() {
    let depth = |m: &abi::Float4x4, z: f32| {
        let (zc, wc) = (z * m[2][2] + m[3][2], z * m[2][3] + m[3][3]);
        zc / wc
    };
    let standard = projection(0.5, 100.0, 1.2, 16.0 / 9.0, false);
    assert!((depth(&standard, 0.5) - 0.0).abs() < 1e-5);
    assert!((depth(&standard, 100.0) - 1.0).abs() < 1e-5);
    let reversed = projection(0.5, 100.0, 1.2, 16.0 / 9.0, true);
    assert!((depth(&reversed, 0.5) - 1.0).abs() < 1e-5);
    assert!((depth(&reversed, 100.0) - 0.0).abs() < 1e-5);
    let infinite = projection(0.5, f32::MAX, 1.2, 1.0, true);
    assert!((depth(&infinite, 0.5) - 1.0).abs() < 1e-5);
    assert!(depth(&infinite, 1e7) < 1e-6);
    assert_eq!(projection(0.0, 100.0, 1.2, 1.0, false), IDENTITY);
    let inverse = invert(&standard).unwrap();
    let product: abi::Float4x4 = std::array::from_fn(|r| std::array::from_fn(|c| (0..4).map(|k| standard[r][k] * inverse[k][c]).sum()));
    for r in 0..4 {
        for c in 0..4 {
            let want = if r == c { 1.0 } else { 0.0 };
            assert!((product[r][c] - want).abs() < 1e-4, "{r} {c} {}", product[r][c]);
        }
    }
    assert_eq!(invert(&[[0.0; 4]; 4]), None);
}

/// The title names the far plane first (inverted depth): in order, the
/// projection puts the nearer plane at depth 1 and the farther at 0; as
/// named, it took the farther for the near plane of an endless range.
#[test]
fn planes_named_far_first_are_put_in_order() {
    let mut named = camera(FFX_DEPTH_INVERTED);
    (named.near, named.far) = (10_000.0, 0.15);
    let ordered = named.planes_in_order();
    assert_eq!((ordered.near, ordered.far), (0.15, 10_000.0));
    assert_eq!(ordered.planes_in_order(), ordered);
    let c = ordered.constants([1.0, 1.0], [1.0, 1.0]);
    assert_eq!((c.camera_near, c.camera_far), (0.15, 10_000.0));
    let depth = |z: f32| {
        let m = &c.camera_view_to_clip;
        (z * m[2][2] + m[3][2]) / (z * m[2][3] + m[3][3])
    };
    assert!((depth(0.15) - 1.0).abs() < 1e-5 && depth(10_000.0).abs() < 1e-5);
    // As named: an endless range from 10000 on.
    let wrong = named.constants([1.0, 1.0], [1.0, 1.0]);
    assert_eq!(wrong.camera_view_to_clip[3][2], 10_000.0);
    // An endless far plane, or a plane at 0, is left alone.
    let mut endless = camera(FFX_DEPTH_INVERTED);
    (endless.near, endless.far) = (0.15, f32::MAX);
    assert_eq!(endless.planes_in_order(), endless);
    (endless.near, endless.far) = (0.15, 0.0);
    assert_eq!(endless.planes_in_order(), endless);
}

#[test]
fn multiplier_pairs() {
    assert_eq!(pair(None), [1.0, 1.0]);
    assert_eq!(pair(Some("-1, 1")), [-1.0, 1.0]);
    assert_eq!(pair(Some("1")), [1.0, 1.0]);
    assert_eq!(pair(Some("1,2,3")), [1.0, 1.0]);
    assert_eq!(pair(Some("x,1")), [1.0, 1.0]);
}
