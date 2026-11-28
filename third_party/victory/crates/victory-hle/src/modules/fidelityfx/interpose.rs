//! NVIDIA Streamline's way into the title's FidelityFX (`VICTORY_DLSS=1`,
//! `VICTORY_FRAMEGEN=1`), beside the host upscalers of the parent module
//! (FidelityFX PC, DLSS through NGX), which serve the title's imports
//! instead when Streamline is not on (the two are never both on:
//! `victory_gpu::streamline::take_over`).
//!
//! `amd_fidelityfx_loader_xbox.dll`: the title's FidelityFX API, five
//! functions over chains of descriptions, each starting with a type id and
//! the next one's address. With DLSS or frame generation on
//! (`victory_gpu::streamline`, `VICTORY_DLSS=1`, `VICTORY_FRAMEGEN=1`), the
//! title's imports of it are bound to the functions here ([`Interpose`]),
//! which call the title's own DLL for everything except an upscale dispatch
//! DLSS takes: DLSS then records the upscaling at that place of the title's
//! command list instead, from the same textures into the same output. With
//! frame generation alone the title's own upscaling runs and its depth and
//! motion vectors are noted for the present. Without either switch, or
//! without Streamline, the imports are bound as before.
//!
//! The title's upscaler context stays the title's own (its FidelityFX makes
//! it), so the title's upscaling runs again on any frame DLSS does not take.
//!
//! The description layouts are the FidelityFX SDK's API (MIT) as DexNom's fork
//! reads them (DexNom/VIctory, `crates/victory-hle/src/modules/fidelityfx.rs`,
//! GPL-3.0-or-later), from which they are ported.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use victory_gpu::object::Object;
use victory_gpu::streamline::{self, Taken, Upscale, plan::Camera};
use victory_loader::{ImportResolver, Provenance, Resolution};
use victory_pe::ImportSymbol;

/// The DLL the title imports the API from (it loads the upscaler itself).
pub const LOADER_DLL: &str = "amd_fidelityfx_loader_xbox.dll";

// Return codes.
const OK: u32 = 0;
const PARAMETER: u32 = 6;

// Description types.
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

/// A resource, its description (type, format, sizes, mips, flags, usage)
/// and the state it is in.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; jitter,
/// motion-vector scale, sizes, sharpening, frame time (ms), pre-exposure,
/// reset, camera and flags. (The C `bool`s are read as bytes.)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DispatchUpscale {
    header: Header,
    list: *mut c_void,
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    enable_sharpening: u8,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: u8,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);

type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;

/// The title's own functions, by [`NAMES`].
static ORIGINALS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
const NAMES: [&str; 5] = ["ffxCreateContext", "ffxDestroyContext", "ffxConfigure", "ffxQuery", "ffxDispatch"];

/// An upscaler context the title made: its creation flags and largest
/// sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Created {
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    /// Made smaller than asked for the shipping FSR4 (DLSS on): the title's
    /// own upscaling must not run on it at the size asked.
    clamped: bool,
}

static CONTEXTS: Mutex<Option<HashMap<usize, Created>>> = Mutex::new(None);

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

/// Binds the title's imports of [`LOADER_DLL`] to the functions here when
/// DLSS or frame generation is on, keeping what `inner` binds them to (the
/// title's own DLL) for them to call; otherwise binds as `inner` does.
pub struct Interpose<'a> {
    pub inner: &'a dyn ImportResolver,
}

impl ImportResolver for Interpose<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        let found = self.inner.resolve(dll, symbol);
        if !dll.eq_ignore_ascii_case(LOADER_DLL) || !streamline::serving() {
            return found;
        }
        match (&found, symbol) {
            (&Resolution::Found { address, .. }, ImportSymbol::Name { name, .. }) => match interpose(name, address) {
                Some(ours) => Resolution::Found { address: ours, provenance: Provenance::Hle },
                None => found,
            },
            _ => found,
        }
    }
}

/// Keeps `original` as the title's `name` and gives ours, if we have one.
fn interpose(name: &str, original: usize) -> Option<usize> {
    let i = NAMES.iter().position(|n| *n == name)?;
    ORIGINALS[i].store(original, Relaxed);
    let ours = [
        ffxCreateContext as *const () as usize,
        ffxDestroyContext as *const () as usize,
        ffxConfigure as *const () as usize,
        ffxQuery as *const () as usize,
        ffxDispatch as *const () as usize,
    ][i];
    tracing::info!(
        name,
        original = format_args!("{original:#x}"),
        "FidelityFX: the title's import goes through ours (DLSS / frame generation)"
    );
    Some(ours)
}

fn original(i: usize) -> Option<*const ()> {
    let address = ORIGINALS[i].load(Relaxed);
    (address != 0).then_some(address as *const ())
}

fn contexts<T>(f: impl FnOnce(&mut HashMap<usize, Created>) -> T) -> T {
    let mut contexts = CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
    f(contexts.get_or_insert_with(HashMap::new))
}

// Shipping FSR4 context sizes: assumed (measured by VIctory-NVIDIA on 26/11), to confirm in game.
const SHIPPING_MAX_OUTPUT: [u32; 2] = [1920, 1080];
const SHIPPING_MAX_RENDER: [u32; 2] = [1280, 720];

/// The sizes the title's shipping FSR4 accepts for a context, if `render` and
/// `output` are not those: (render, output).
fn shipping_sizes(render: [u32; 2], output: [u32; 2]) -> Option<([u32; 2], [u32; 2])> {
    (output != SHIPPING_MAX_OUTPUT || (render != SHIPPING_MAX_RENDER && render != SHIPPING_MAX_OUTPUT))
        .then_some((SHIPPING_MAX_RENDER, SHIPPING_MAX_OUTPUT))
}

extern "C" fn ffxCreateContext(context: *mut *mut c_void, desc: *mut Header, memory: *const c_void) -> u32 {
    let Some(create) = original(0) else { return PARAMETER };
    // SAFETY: the title's own function, with its arguments.
    let call = |desc: *mut Header| unsafe { std::mem::transmute::<*const (), Create>(create)(context, desc, memory) };
    // SAFETY: the title passes a chain of descriptions.
    let upscaler = !desc.is_null() && unsafe { (*desc).kind } == CREATE_UPSCALE;
    if !upscaler {
        return call(desc);
    }
    // SAFETY: an upscaler description, by its type.
    let asked = unsafe { desc.cast::<CreateUpscale>().read() };
    // With DLSS taking the upscalings, the title's own context only has to
    // exist (for its queries and our bookkeeping): it is made at a size its
    // shipping tables accept, and DLSS gets the sizes the title asked for.
    let shipping = streamline::upscaling().then(|| shipping_sizes(asked.max_render, asked.max_output)).flatten();
    let clamped = shipping.is_some();
    let code = if let Some((render, output)) = shipping {
        let mut fitted = asked;
        fitted.max_render = render;
        fitted.max_output = output;
        // SAFETY: the title's description, rewritten for the call and put back.
        unsafe { desc.cast::<CreateUpscale>().write(fitted) };
        let code = call(desc);
        // SAFETY: as above.
        unsafe { desc.cast::<CreateUpscale>().write(asked) };
        tracing::info!(
            asked_render = ?asked.max_render,
            asked_output = ?asked.max_output,
            made_render = ?fitted.max_render,
            made_output = ?fitted.max_output,
            code,
            "FidelityFX: the title's upscaler made at a size its shipping FSR4 accepts; DLSS upscales to the size asked"
        );
        code
    } else {
        call(desc)
    };
    if code == OK && (context.is_null() || unsafe { (*context).is_null() }) {
        tracing::error!("FidelityFX returned success without an upscaler context");
        return PARAMETER;
    }
    if code != OK {
        // Frame generation without DLSS leaves the upscaling to the title's
        // own FidelityFX, which is made for the Series S sizes only.
        if code != OK && !streamline::upscaling() {
            tracing::error!(
                code,
                max_render = ?asked.max_render,
                max_output = ?asked.max_output,
                "FidelityFX: the title's own upscaler was not made at these sizes (its shipping FSR4 takes 1280x720 or 1920x1080 in, 1920x1080 out): with frame generation at another output size, set VICTORY_DLSS=1 as well; the title fails at its first upscaling without an upscaler"
            );
        }
        return code;
    }
    // SAFETY: the context was made.
    let handle = unsafe { *context } as usize;
    let created = Created { flags: asked.flags, max_render: asked.max_render, max_output: asked.max_output, clamped };
    tracing::info!(
        context = format_args!("{handle:#x}"),
        flags = format_args!("{:#x}", created.flags),
        max_render = ?created.max_render,
        max_output = ?created.max_output,
        "FidelityFX: the title made its upscaler"
    );
    contexts(|c| c.insert(handle, created));
    code
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    let Some(destroy) = original(1) else { return PARAMETER };
    // The native function may clear the handle on success.
    let handle = unsafe { context.as_ref() }.map(|p| *p as usize);
    let code = unsafe { std::mem::transmute::<*const (), Destroy>(destroy)(context, memory) };
    if code == OK
        && let Some(handle) = handle
        && contexts(|c| c.remove(&handle)).is_some()
    {
        streamline::upscaler_destroyed(handle);
    }
    code
}

extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(configure) = original(2) else { return PARAMETER };
    // SAFETY: the title's own function, with its arguments.
    unsafe { std::mem::transmute::<*const (), Call>(configure)(context, desc) }
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(query) = original(3) else { return PARAMETER };
    if !desc.is_null() {
        // SAFETY: the title passes a description.
        let kind = unsafe { (*desc).kind };
        static SEEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&kind) && seen.len() < 32 {
            seen.push(kind);
            tracing::info!(kind = format_args!("{kind:#x}"), "FidelityFX: the title asks a query of this kind (answered by its own DLL)");
        }
    }
    // SAFETY: the title's own function, with its arguments.
    unsafe { std::mem::transmute::<*const (), Call>(query)(context, desc) }
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(dispatch) = original(4) else { return PARAMETER };
    // SAFETY: the title's own function, with its arguments.
    let own = || unsafe { std::mem::transmute::<*const (), Call>(dispatch)(context, desc) };
    if context.is_null() || desc.is_null() {
        return own();
    }
    // SAFETY: the title passes a description and its context.
    let (kind, handle) = unsafe { ((*desc).kind, *context as usize) };
    if kind != DISPATCH_UPSCALE {
        return own();
    }
    let Some(created) = contexts(|c| c.get(&handle).copied()) else {
        once!(warn, "FidelityFX: an upscale dispatch on a context we did not see made; the title's own runs");
        return own();
    };
    // SAFETY: an upscaling, by its type.
    let d = unsafe { desc.cast::<DispatchUpscale>().read() };
    first_dispatch(&d, &created);
    let up = upscale(&d, &created, handle);
    match streamline::upscale(&up) {
        Taken::Dlss => OK,
        Taken::Noted => own(),
        Taken::No(why) if created.clamped => {
            // The title's FidelityFX on a context made for 1080p at the
            // output size asked would read past its tables: no upscaling
            // this frame (the output keeps the last one) rather than that.
            tracing::error!(why, "FidelityFX: DLSS failed and the native context cannot serve the requested dimensions");
            PARAMETER
        }
        Taken::No(why) => {
            streamline::forwarded();
            refused(why);
            own()
        }
    }
}

/// Each reason the title's own upscaling ran instead, once.
fn refused(why: &'static str) {
    static SEEN: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&why) {
        seen.push(why);
        tracing::warn!("FidelityFX: the title's own upscaling runs: {why}");
    }
}

/// The first upscale dispatch, all of it.
fn first_dispatch(d: &DispatchUpscale, created: &Created) {
    once!(
        info,
        render = ?d.render_size,
        output = ?d.upscale_size,
        max_render = ?created.max_render,
        max_output = ?created.max_output,
        create_flags = format_args!("{:#x}", created.flags),
        flags = format_args!("{:#x}", d.flags),
        jitter = ?d.jitter,
        motion_vector_scale = ?d.motion_scale,
        near = d.camera_near,
        far = d.camera_far,
        fov = d.camera_fov_angle_vertical,
        frame_ms = d.frame_time_delta,
        pre_exposure = d.pre_exposure,
        reset = d.reset != 0,
        sharpening = d.enable_sharpening != 0,
        resources = ?d.resources.iter().map(|r| (r.resource as usize != 0, r.description[1], r.description[2], r.description[3], r.state)).collect::<Vec<_>>(),
        "FidelityFX: the title's first upscale dispatch (resources: present, format, width, height, state)"
    );
}

/// An upscaling in `victory_gpu::streamline`'s terms. The output size is the
/// context's largest when the dispatch names none (as FidelityFX reads it).
fn upscale(d: &DispatchUpscale, created: &Created, context: usize) -> Upscale {
    let output = if d.upscale_size == [0, 0] { created.max_output } else { d.upscale_size };
    Upscale {
        context,
        list: d.list.cast::<Object>().cast_const(),
        resources: d.resources.map(|r| (r.resource.cast::<Object>().cast_const(), r.state)),
        camera: Camera {
            jitter: d.jitter,
            motion_scale: d.motion_scale,
            render: d.render_size,
            output,
            near: d.camera_near,
            far: d.camera_far,
            fov_y: d.camera_fov_angle_vertical,
            reset: d.reset != 0,
            flags: created.flags,
        },
        pre_exposure: d.pre_exposure,
        frame_ms: d.frame_time_delta,
    }
}

#[cfg(test)]
mod tests;
