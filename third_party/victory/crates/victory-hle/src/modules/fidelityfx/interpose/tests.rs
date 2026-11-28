use super::*;
use std::mem::offset_of;
use std::ptr::null_mut;

#[test]
fn descriptions_lay_out_as_the_sdk() {
    assert_eq!(offset_of!(CreateUpscale, flags), 0x10);
    assert_eq!(offset_of!(CreateUpscale, max_output), 0x1c);
    assert_eq!(offset_of!(Resource, state), 0x28);
    assert_eq!(offset_of!(DispatchUpscale, list), 0x10);
    assert_eq!(offset_of!(DispatchUpscale, resources), 0x18);
    assert_eq!(offset_of!(DispatchUpscale, jitter), 0x168);
    assert_eq!(offset_of!(DispatchUpscale, render_size), 0x178);
    assert_eq!(offset_of!(DispatchUpscale, upscale_size), 0x180);
    assert_eq!(offset_of!(DispatchUpscale, enable_sharpening), 0x188);
    assert_eq!(offset_of!(DispatchUpscale, frame_time_delta), 0x190);
    assert_eq!(offset_of!(DispatchUpscale, reset), 0x198);
    assert_eq!(offset_of!(DispatchUpscale, camera_near), 0x19c);
    assert_eq!(offset_of!(DispatchUpscale, flags), 0x1ac);
}

fn resource(at: usize, state: u32) -> Resource {
    Resource { resource: at as *mut c_void, description: [1, 10, 2560, 1440, 1, 1, 0, 0], state }
}

fn dispatch(output: [u32; 2]) -> DispatchUpscale {
    DispatchUpscale {
        header: Header { kind: DISPATCH_UPSCALE, next: null_mut() },
        list: 0x1000 as *mut c_void,
        resources: [
            resource(0x10, 0xc),
            resource(0x20, 0xc),
            resource(0x30, 0xc),
            resource(0, 0),
            resource(0x50, 0xc),
            resource(0, 0),
            resource(0x70, 0x2),
        ],
        jitter: [0.25, -0.125],
        motion_scale: [1706.0, 960.0],
        render_size: [1706, 960],
        upscale_size: output,
        enable_sharpening: 0,
        sharpness: 0.0,
        frame_time_delta: 33.3,
        pre_exposure: 1.5,
        reset: 1,
        camera_near: 0.1,
        camera_far: 5000.0,
        camera_fov_angle_vertical: 0.9,
        view_space_to_meters_factor: 1.0,
        flags: 0,
    }
}

/// A dispatch reads as the upscaling it names; without an output size
/// the context's largest is the output.
#[test]
fn a_dispatch_reads_as_its_upscaling() {
    let created = Created { flags: 0x29, max_render: [2560, 1440], max_output: [2560, 1440], clamped: false };
    let up = upscale(&dispatch([2560, 1440]), &created, 0xabc);
    assert_eq!(up.context, 0xabc);
    assert_eq!(up.list as usize, 0x1000);
    assert_eq!(up.resources[streamline::COLOUR], (0x10 as *const Object, 0xc));
    assert_eq!(up.resources[streamline::EXPOSURE].0 as usize, 0);
    assert_eq!(up.resources[streamline::OUTPUT], (0x70 as *const Object, 0x2));
    assert_eq!(up.camera.render, [1706, 960]);
    assert_eq!(up.camera.output, [2560, 1440]);
    assert_eq!(up.camera.flags, 0x29);
    assert!(up.camera.reset && up.camera.depth_inverted());
    assert_eq!((up.pre_exposure, up.frame_ms), (1.5, 33.3));
    let up = upscale(&dispatch([0, 0]), &Created { max_output: [1920, 1080], ..created }, 0);
    assert_eq!(up.camera.output, [1920, 1080]);
}

/// A context at other sizes than the shipping FSR4's is made at those
/// (1280x720 in, 1920x1080 out); one at them is left as it is.
#[test]
fn contexts_are_made_at_the_shipping_sizes() {
    let shipping = Some(([1280, 720], [1920, 1080]));
    assert_eq!(shipping_sizes([1706, 960], [2560, 1440]), shipping);
    assert_eq!(shipping_sizes([2560, 1440], [2560, 1440]), shipping);
    assert_eq!(shipping_sizes([1280, 720], [2560, 1440]), shipping);
    assert_eq!(shipping_sizes([1280, 720], [1920, 1080]), None);
    assert_eq!(shipping_sizes([1920, 1080], [1920, 1080]), None);
}

struct Fixed;
impl ImportResolver for Fixed {
    fn resolve(&self, dll: &str, _: &ImportSymbol) -> Resolution {
        if dll.eq_ignore_ascii_case(LOADER_DLL) {
            Resolution::Found { address: 0x1234, provenance: Provenance::Title }
        } else {
            Resolution::Missing
        }
    }
}

/// Without Streamline the title's imports bind as before; the names we
/// serve are the API's five.
#[test]
fn without_streamline_the_titles_own_binds() {
    let name = |n: &str| ImportSymbol::Name { hint: 0, name: n.into() };
    let interposed = Interpose { inner: &Fixed };
    assert_eq!(
        interposed.resolve("AMD_FIDELITYFX_LOADER_XBOX.DLL", &name("ffxDispatch")),
        Resolution::Found { address: 0x1234, provenance: Provenance::Title }
    );
    assert_eq!(interposed.resolve("other.dll", &name("ffxDispatch")), Resolution::Missing);
    assert_eq!(interpose("ffxGetVersion", 1), None);
    assert!(interpose("ffxQuery", 0x5678).is_some());
    assert_eq!(original(3), Some(0x5678 as *const ()));
    ORIGINALS[3].store(0, Relaxed);
}

#[test]
fn a_declined_dispatch_uses_native_fidelityfx_only_at_compatible_sizes() {
    const NATIVE_RESULT: u32 = 17;
    unsafe extern "C" fn native(_: *mut *mut c_void, _: *mut Header) -> u32 {
        NATIVE_RESULT
    }
    let mut fixture = dispatch([1920, 1080]);
    let mut handle = std::ptr::from_mut(&mut fixture).cast::<c_void>();
    let context = handle as usize;
    let created = Created { flags: 0, max_render: [1280, 720], max_output: [1920, 1080], clamped: false };
    ORIGINALS[4].store(native as *const () as usize, Relaxed);
    contexts(|c| c.insert(context, created));
    assert_eq!(ffxDispatch(&raw mut handle, &raw mut fixture.header), NATIVE_RESULT);
    contexts(|c| c.insert(context, Created { clamped: true, ..created }));
    assert_eq!(ffxDispatch(&raw mut handle, &raw mut fixture.header), PARAMETER);
    contexts(|c| c.remove(&context));
    ORIGINALS[4].store(0, Relaxed);
}
