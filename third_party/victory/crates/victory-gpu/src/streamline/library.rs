//! Every DLL of the Streamline folder is verified (NVIDIA Authenticode) before the
//! interposer is loaded, and kept open read-only so none can be replaced meanwhile.

use super::*;

struct Library(windows_sys::Win32::Foundation::HMODULE);

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this owns the reference LoadLibraryExW took.
        unsafe { windows_sys::Win32::Foundation::FreeLibrary(self.0) };
    }
}

// SAFETY: a module handle.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

/// The library's exported functions.
pub(super) struct Sl {
    _library: Library,
    _verified: Vec<std::fs::File>,
    pub(super) init: abi::SlInit,
    pub(super) is_feature_supported: abi::SlIsFeatureSupported,
    pub(super) evaluate_feature: abi::SlEvaluateFeature,
    pub(super) free_resources: abi::SlFreeResources,
    pub(super) set_tag_for_frame: abi::SlSetTagForFrame,
    pub(super) get_feature_requirements: abi::SlGetFeatureRequirements,
    pub(super) get_feature_version: abi::SlGetFeatureVersion,
    pub(super) upgrade_interface: abi::SlUpgradeInterface,
    pub(super) set_constants: abi::SlSetConstants,
    pub(super) get_native_interface: abi::SlGetNativeInterface,
    pub(super) get_feature_function: abi::SlGetFeatureFunction,
    pub(super) get_new_frame_token: abi::SlGetNewFrameToken,
    pub(super) set_d3d_device: abi::SlSetD3DDevice,
    pub(super) shutdown: abi::SlShutdown,
}

pub(super) fn load(dir: &Path, dlss: bool, framegen: bool) -> Result<Sl, String> {
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };
    let verified = super::trust::verify_runtime(dir, dlss, framegen)?;
    let file = wide(&dir.join(INTERPOSER));
    // Its own folder and System32 only, never the current folder.
    // SAFETY: a NUL-terminated full path.
    let module =
        unsafe { LoadLibraryExW(file.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32) };
    if module.is_null() {
        return Err(format!("{INTERPOSER} did not load ({})", std::io::Error::last_os_error()));
    }
    let library = Library(module);
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{INTERPOSER} has no {}", name.to_string_lossy()))
    };
    macro_rules! get {
        ($name:literal, $t:ty) => {
            // SAFETY: the export of that name has the header's signature.
            unsafe { std::mem::transmute::<*const c_void, $t>(find($name)?) }
        };
    }
    Ok(Sl {
        init: get!(c"slInit", abi::SlInit),
        is_feature_supported: get!(c"slIsFeatureSupported", abi::SlIsFeatureSupported),
        evaluate_feature: get!(c"slEvaluateFeature", abi::SlEvaluateFeature),
        free_resources: get!(c"slFreeResources", abi::SlFreeResources),
        set_tag_for_frame: get!(c"slSetTagForFrame", abi::SlSetTagForFrame),
        get_feature_requirements: get!(c"slGetFeatureRequirements", abi::SlGetFeatureRequirements),
        get_feature_version: get!(c"slGetFeatureVersion", abi::SlGetFeatureVersion),
        upgrade_interface: get!(c"slUpgradeInterface", abi::SlUpgradeInterface),
        set_constants: get!(c"slSetConstants", abi::SlSetConstants),
        get_native_interface: get!(c"slGetNativeInterface", abi::SlGetNativeInterface),
        get_feature_function: get!(c"slGetFeatureFunction", abi::SlGetFeatureFunction),
        get_new_frame_token: get!(c"slGetNewFrameToken", abi::SlGetNewFrameToken),
        set_d3d_device: get!(c"slSetD3DDevice", abi::SlSetD3DDevice),
        shutdown: get!(c"slShutdown", abi::SlShutdown),
        _library: library,
        _verified: verified,
    })
}
