//! NVIDIA Streamline 2.14.16, loaded from the user's signed runtime.
mod abi;
mod config;
mod library;
pub mod plan;
mod trust;

pub use config::{Asked, Available};
use library::*;
pub use plan::Mode;

use crate::object::Object;
use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

mod check;
mod device;
mod frames;
mod hudless;
pub(crate) mod latency;
mod options;
mod present;
mod stats;
pub use check::check;
use device::{COPY_DEST, texture};
pub(crate) use frames::{Noted, submitted};
use frames::*;
use hudless::*;
pub(crate) use hudless::{Copied, before_target_draw, hudless_resized, hudless_submitted, hudless_wanted};
pub(crate) use present::{Planes, present_begins, present_ends, resizing};
use stats::*;
pub(crate) use stats::{line, present_called, present_returned};

mod upscale;
pub(crate) use device::{device_made, present_hooked, swap_chain_factory, swap_chain_made};
pub use upscale::*;

const VIEWPORT_FG: u32 = 0;
const VIEWPORT_DLSS: u32 = 1;
const INTERPOSER: &str = "sl.interposer.dll";
/// The project id NGX is given (engine "custom", no NVIDIA application id).
const PROJECT_ID: &CStr = c"6f0d7b2e-3a41-4c1e-9b57-2d8e4f1a9c30";
const ENGINE_VERSION: &CStr = c"VIctory 0.1";
static SL: OnceLock<Sl> = OnceLock::new();

/// What is on once the library is in: the switches and the features found.
struct Plan {
    asked: Asked,
    dlss: bool,
    framegen: bool,
}

static PLAN: OnceLock<Plan> = OnceLock::new();

/// Loads Streamline from `asked.dir` and asks for what `asked` names; call
/// once, before the host device is made (`crate::host::host`) and before the
/// title's imports are bound. Fails soft: what is not there is logged and
/// left off.
///
/// The first call decides; later ones (`take_over` early in the launch,
/// `title_mode` once the device can be made) get the same answer.
pub fn init(asked: &Asked) -> Available {
    static INIT: OnceLock<Available> = OnceLock::new();
    if !asked.any() {
        return Available::default();
    }
    *INIT.get_or_init(|| match load_and_init(asked, true) {
        Ok(available) => available,
        Err(why) => {
            tracing::warn!(
                folder = %asked.dir.display(),
                "Streamline: {why}; DLSS and frame generation stay off, the title's own upscaling and our present run"
            );
            Available::default()
        }
    })
}

/// Set by [`take_over`] once Streamline is asked for and loaded.
static TAKEN_OVER: OnceLock<bool> = OnceLock::new();

/// Whether `VICTORY_DLSS=1` or `VICTORY_FRAMEGEN=1` names Streamline.
fn switches_on() -> bool {
    ["VICTORY_DLSS", "VICTORY_FRAMEGEN"].iter().any(|key| std::env::var_os(key).is_some_and(|v| v == "1"))
}

/// Whether Streamline is in charge of upscaling and frame generation: its
/// switches are on and its library loaded with a feature it supports. Then
/// the other upscalers of this tree (FidelityFX PC, DLSS through NGX, the
/// native frame generation) are not set up; where they were asked for, the
/// caller says so. Loads the library on its first call, which has to come
/// before the host device is made and the title's imports are bound.
pub fn take_over() -> bool {
    *TAKEN_OVER.get_or_init(|| {
        if !switches_on() {
            return false;
        }
        let Ok(asked) = Asked::from_env() else { return false };
        let available = init(&asked);
        available.dlss || available.framegen
    })
}

/// [`take_over`]'s answer, without deciding it: false until it has been asked.
pub fn taken_over() -> bool {
    TAKEN_OVER.get().copied().unwrap_or(false)
}

fn load_and_init(asked: &Asked, require_line: bool) -> Result<Available, String> {
    let dir = trust::runtime_dir(&asked.dir)?;
    let path = dir.join(INTERPOSER);
    let framegen_asked = asked.framegen && (!require_line || line_asked());
    let sl = load(&dir, asked.dlss, framegen_asked)?;
    let sl = SL.get_or_init(|| sl);
    let dir_wide = wide(&dir);
    let paths = [dir_wide.as_ptr()];
    let log_wide = asked.log_dir.as_deref().map(wide);
    // Frame generation presents from the line's thread, in order with the
    // submissions: without the line it is not loaded at all.
    if asked.framegen && !framegen_asked {
        tracing::warn!(
            "DLSS Frame Generation needs the line of submissions (VICTORY_THIN=1 or VICTORY_ASYNC_SUBMIT=1): frame generation stays off"
        );
    }
    let mut features = Vec::new();
    if asked.dlss {
        features.push(abi::FEATURE_DLSS);
    }
    if framegen_asked {
        features.extend([abi::FEATURE_DLSS_G, abi::FEATURE_REFLEX, abi::FEATURE_PCL]);
    }
    let preferences = abi::Preferences {
        base: abi::BaseStructure::new(abi::PREFERENCES, 1),
        show_console: false,
        log_level: if asked.log_dir.is_some() { abi::LOG_VERBOSE } else { abi::LOG_DEFAULT },
        paths_to_plugins: paths.as_ptr(),
        num_paths_to_plugins: 1,
        path_to_logs_and_data: log_wide.as_ref().map_or(std::ptr::null(), |w| w.as_ptr()),
        allocate_callback: std::ptr::null(),
        release_callback: std::ptr::null(),
        log_message_callback: Some(log_message),
        // No downloads (OTA): only what is in the folder runs.
        flags: abi::DISABLE_CL_STATE_TRACKING
            | abi::DISABLE_DEBUG_TEXT
            | abi::USE_MANUAL_HOOKING
            | abi::USE_DXGI_FACTORY_PROXY
            | abi::USE_FRAME_BASED_RESOURCE_TAGGING,
        features_to_load: features.as_ptr(),
        num_features_to_load: features.len() as u32,
        application_id: 0,
        engine: abi::ENGINE_CUSTOM,
        engine_version: ENGINE_VERSION.as_ptr(),
        project_id: PROJECT_ID.as_ptr(),
        render_api: abi::RENDER_API_D3D12,
    };
    // SAFETY: a filled preferences structure; the library copies what it keeps.
    let code = unsafe { (sl.init)(&preferences, abi::SDK_VERSION) };
    if code != 0 {
        return Err(format!("slInit failed: {} ({code})", abi::result_name(code)));
    }
    let adapter = default_adapter();
    let luid = adapter.as_ref().map(|a| a.1);
    let supported = |feature: u32| -> Result<(), String> {
        let Some(luid) = luid else { return Err("no adapter".into()) };
        let info = abi::AdapterInfo {
            base: abi::BaseStructure::new(abi::ADAPTER_INFO, 1),
            device_luid: std::ptr::from_ref(&luid).cast(),
            device_luid_size: size_of_val(&luid) as u32,
            vk_physical_device: std::ptr::null_mut(),
        };
        // SAFETY: an adapter description that lives through the call.
        let code = unsafe { (sl.is_feature_supported)(feature, &info) };
        if code == 0 { Ok(()) } else { Err(format!("{} ({code})", abi::result_name(code))) }
    };
    let dlss = asked.dlss && report("DLSS Super Resolution", abi::FEATURE_DLSS, supported(abi::FEATURE_DLSS), sl);
    let framegen = framegen_asked
        && report("DLSS Frame Generation", abi::FEATURE_DLSS_G, supported(abi::FEATURE_DLSS_G), sl)
        && report("Reflex", abi::FEATURE_REFLEX, supported(abi::FEATURE_REFLEX), sl)
        && report("PCL latency markers", abi::FEATURE_PCL, supported(abi::FEATURE_PCL), sl);
    tracing::info!(
        library = %path.display(),
        sdk = "2.14.16",
        adapter = adapter.as_ref().map_or("?", |a| a.0.as_str()),
        dlss,
        framegen,
        mode = asked.mode.name(),
        preset = asked.preset,
        "Streamline loaded (manual hooking, no downloads)"
    );
    let _ = PLAN.set(Plan { asked: asked.clone(), dlss, framegen });
    Ok(Available { dlss, framegen })
}

/// Logs whether a feature is supported, with its requirements; whether it is.
fn report(name: &str, feature: u32, supported: Result<(), String>, sl: &Sl) -> bool {
    let mut requirements: abi::FeatureRequirements = unsafe_zeroed_requirements();
    // SAFETY: a structure to fill.
    let asked = unsafe { (sl.get_feature_requirements)(feature, &mut requirements) };
    let needs = (asked == 0).then(|| {
        format!(
            "OS {} (needs {}), driver {} (needs {}){}{}",
            requirements.os_version_detected,
            requirements.os_version_required,
            requirements.driver_version_detected,
            requirements.driver_version_required,
            if requirements.flags & abi::REQUIRES_HARDWARE_SCHEDULING != 0 { ", hardware GPU scheduling" } else { "" },
            if requirements.flags & abi::REQUIRES_VSYNC_OFF != 0 { ", v-sync off" } else { "" },
        )
    });
    match supported {
        Ok(()) => {
            tracing::info!(requirements = needs.as_deref().unwrap_or("?"), "Streamline: {name} is supported on this GPU");
            true
        }
        Err(why) => {
            tracing::warn!(requirements = needs.as_deref().unwrap_or("?"), "Streamline: {name} is not supported here: {why}");
            false
        }
    }
}

fn unsafe_zeroed_requirements() -> abi::FeatureRequirements {
    // SAFETY: plain data; pointers null, counts zero.
    let mut r: abi::FeatureRequirements = unsafe { std::mem::zeroed() };
    r.base = abi::BaseStructure::new(abi::FEATURE_REQUIREMENTS, 2);
    r
}

/// Whether the switches turn the line of submissions on (frame generation
/// presents from its thread, in order with the submissions).
fn line_asked() -> bool {
    let on = |name: &str| std::env::var_os(name).is_some_and(|v| v == "1");
    on("VICTORY_THIN") || on("VICTORY_ASYNC_SUBMIT")
}

unsafe extern "C" fn log_message(kind: u32, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: a NUL-terminated message for the call.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let text = text.trim_end();
    // (Every plugin in the folder that was not asked for says so.)
    let noise = text.contains("since it is was not requested by the host");
    match kind {
        abi::LOG_ERROR => tracing::error!(target: "streamline", "{text}"),
        abi::LOG_WARN if !noise => tracing::warn!(target: "streamline", "{text}"),
        _ => tracing::debug!(target: "streamline", "{text}"),
    }
}

fn wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain([0]).collect()
}

/// The adapter a device made with no adapter named is on (the first), its
/// name and LUID.
fn default_adapter() -> Option<(String, windows::Win32::Foundation::LUID)> {
    // SAFETY: plain DXGI queries.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let adapter = factory.EnumAdapters1(0).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        Some((String::from_utf16_lossy(&desc.Description[..len]), desc.AdapterLuid))
    }
}

/// The per-feature functions, once the device is set.
#[derive(Default)]
struct Functions {
    dlss_optimal: Option<abi::SlDlssGetOptimalSettings>,
    dlss_set_options: Option<abi::SlDlssSetOptions>,
    dlssg_set_options: Option<abi::SlDlssgSetOptions>,
    dlssg_get_state: Option<abi::SlDlssgGetState>,
    reflex_set_options: Option<abi::SlReflexSetOptions>,
    pcl_set_marker: Option<abi::SlPclSetMarker>,
}

static FUNCTIONS: OnceLock<Functions> = OnceLock::new();
static DLSS_READY: AtomicBool = AtomicBool::new(false);
static FG_READY: AtomicBool = AtomicBool::new(false);
static SWAP_CHAIN_PROXIED: AtomicBool = AtomicBool::new(false);
/// Our swap chain is Streamline's proxy of it ([`present_hooked`]).
static PRESENT_HOOKED: AtomicBool = AtomicBool::new(false);

/// Library calls that are not thread safe, and what the upscaling keeps
/// between frames, one caller at a time.
#[derive(Default)]
struct Calls {
    /// Optimal render sizes by output size and mode.
    optimal: Vec<([u32; 2], u32, Option<[u32; 2]>)>,
    /// The mode, sizes and preset the last options named.
    last: Option<(u32, [u32; 2], [u32; 2])>,
    /// Last accepted frame generation setting.
    framegen_on: bool,
    framegen_times: u32,
    /// A successful state read delimits the next displayed-frame count.
    fg_count_valid: bool,
}

static CALLS: Mutex<Calls> = Mutex::new(Calls {
    optimal: Vec::new(), last: None, framegen_on: false, framegen_times: 0, fg_count_valid: false,
});

fn calls() -> std::sync::MutexGuard<'static, Calls> {
    CALLS.lock().unwrap_or_else(|e| e.into_inner())
}
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
/// [`check`] presents from its one thread, with nothing else submitting.
static CHECKING: AtomicBool = AtomicBool::new(false);
fn frame_token(sl: &Sl, index: u32) -> Option<*mut abi::FrameToken> {
    let mut token = std::ptr::null_mut();
    // SAFETY: an out pointer and an index.
    let code = unsafe { (sl.get_new_frame_token)(&mut token, &index) };
    if code != 0 || token.is_null() {
        once!(warn, index, code, result = abi::result_name(code), "Streamline did not supply a frame token");
        return None;
    }
    Some(token)
}

/// Select the title's FidelityFX path only after the host device is ready.
pub fn title_mode() -> Option<Mode> {
    if !switches_on() {
        return None;
    }
    let asked = Asked::from_env().map_err(|why| tracing::warn!("Streamline: {why}; using the title's upscaler")).ok()?;
    let available = init(&asked);
    if !available.dlss && !available.framegen {
        return None;
    }
    if crate::host::host().is_none() || !serving() {
        tracing::warn!("Streamline device unavailable; using the title's upscaler");
        return None;
    }
    Some(asked.mode)
}

pub(crate) fn tracking_frames() -> bool {
    FG_READY.load(Acquire)
}

pub(crate) fn checking() -> bool {
    CHECKING.load(Relaxed)
}

#[cfg(test)]
mod take_over_tests {
    use super::*;

    #[test]
    fn without_its_switches_streamline_does_not_take_over() {
        if switches_on() {
            return;
        }
        assert!(!take_over());
        assert!(!taken_over());
        // Nothing of AMD's frame generation is displaced by it either.
        assert_eq!(crate::framegen::requested(), std::env::var_os("VICTORY_FRAMEGEN").is_some_and(|v| v == "1"));
    }
}
