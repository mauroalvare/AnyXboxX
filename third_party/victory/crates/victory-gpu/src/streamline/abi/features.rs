use super::*;

/// `sl::DLSSOptions` (version 3).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DlssOptions {
    pub base: BaseStructure,
    pub mode: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub sharpness: f32,
    pub pre_exposure: f32,
    pub exposure_scale: f32,
    pub color_buffers_hdr: u8,
    pub indicator_invert_axis_x: u8,
    pub indicator_invert_axis_y: u8,
    pub dlaa_preset: u32,
    pub quality_preset: u32,
    pub balanced_preset: u32,
    pub performance_preset: u32,
    pub ultra_performance_preset: u32,
    pub ultra_quality_preset: u32,
    pub use_auto_exposure: u8,
    pub alpha_upscaling_enabled: u8,
}

impl DlssOptions {
    /// The header's defaults, with `mode` for an output of `output`.
    pub fn new(mode: u32, output: [u32; 2], preset: u32) -> Self {
        DlssOptions {
            base: BaseStructure::new(DLSS_OPTIONS, 3),
            mode,
            output_width: output[0],
            output_height: output[1],
            sharpness: 0.0,
            pre_exposure: 1.0,
            exposure_scale: 1.0,
            color_buffers_hdr: TRUE,
            indicator_invert_axis_x: FALSE,
            indicator_invert_axis_y: FALSE,
            dlaa_preset: preset,
            quality_preset: preset,
            balanced_preset: preset,
            performance_preset: preset,
            ultra_performance_preset: preset,
            ultra_quality_preset: preset,
            use_auto_exposure: FALSE,
            alpha_upscaling_enabled: FALSE,
        }
    }
}

/// `sl::DLSSOptimalSettings`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DlssOptimalSettings {
    pub base: BaseStructure,
    pub optimal_render_width: u32,
    pub optimal_render_height: u32,
    pub optimal_sharpness: f32,
    pub render_width_min: u32,
    pub render_height_min: u32,
    pub render_width_max: u32,
    pub render_height_max: u32,
}

impl Default for DlssOptimalSettings {
    fn default() -> Self {
        DlssOptimalSettings {
            base: BaseStructure::new(DLSS_OPTIMAL_SETTINGS, 1),
            optimal_render_width: 0,
            optimal_render_height: 0,
            optimal_sharpness: 0.0,
            render_width_min: 0,
            render_height_min: 0,
            render_width_max: 0,
            render_height_max: 0,
        }
    }
}

/// `sl::DLSSGOptions` (version 5).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DlssgOptions {
    pub base: BaseStructure,
    pub mode: u32,
    pub num_frames_to_generate: u32,
    pub flags: u32,
    pub dynamic_res_width: u32,
    pub dynamic_res_height: u32,
    pub num_back_buffers: u32,
    pub mvec_depth_width: u32,
    pub mvec_depth_height: u32,
    pub color_width: u32,
    pub color_height: u32,
    pub color_buffer_format: u32,
    pub mvec_buffer_format: u32,
    pub depth_buffer_format: u32,
    pub hud_less_buffer_format: u32,
    pub ui_buffer_format: u32,
    pub on_error_callback: Option<ApiErrorCallback>,
    pub reserved15: u8,
    pub queue_parallelism_mode: u32,
    pub enable_user_interface_recomposition: u8,
    pub dynamic_target_frame_rate: f32,
}

/// `PFunOnAPIErrorCallback`: the HRESULT a present returned (a union of
/// HRESULT and VkResult, passed by reference).
pub type ApiErrorCallback = unsafe extern "C" fn(error: *const i32);

impl DlssgOptions {
    pub fn new(mode: u32) -> Self {
        DlssgOptions {
            base: BaseStructure::new(DLSSG_OPTIONS, 5),
            mode,
            num_frames_to_generate: crate::frame_rate::FG_MULTIPLIER - 1,
            flags: 0,
            dynamic_res_width: 0,
            dynamic_res_height: 0,
            num_back_buffers: 0,
            mvec_depth_width: 0,
            mvec_depth_height: 0,
            color_width: 0,
            color_height: 0,
            color_buffer_format: 0,
            mvec_buffer_format: 0,
            depth_buffer_format: 0,
            hud_less_buffer_format: 0,
            ui_buffer_format: 0,
            on_error_callback: None,
            reserved15: INVALID,
            queue_parallelism_mode: 0,
            enable_user_interface_recomposition: FALSE,
            dynamic_target_frame_rate: 0.0,
        }
    }
}

/// `sl::DLSSGState` (version 4).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DlssgState {
    pub base: BaseStructure,
    pub estimated_vram_usage_in_bytes: u64,
    pub status: u32,
    pub min_width_or_height: u32,
    pub num_frames_actually_presented: u32,
    pub num_frames_to_generate_max: u32,
    pub reserved4: u8,
    pub is_vsync_support_available: u8,
    pub inputs_processing_completion_fence: *mut c_void,
    pub last_present_inputs_processing_completion_fence_value: u64,
    pub is_dynamic_mfg_supported: u8,
}

impl Default for DlssgState {
    fn default() -> Self {
        DlssgState {
            base: BaseStructure::new(DLSSG_STATE, 4),
            estimated_vram_usage_in_bytes: 0,
            status: 0,
            min_width_or_height: 0,
            num_frames_actually_presented: 0,
            num_frames_to_generate_max: 0,
            reserved4: 0,
            is_vsync_support_available: 0,
            inputs_processing_completion_fence: std::ptr::null_mut(),
            last_present_inputs_processing_completion_fence_value: 0,
            is_dynamic_mfg_supported: 0,
        }
    }
}

/// `sl::ReflexOptions`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReflexOptions {
    pub base: BaseStructure,
    pub mode: u32,
    pub frame_limit_us: u32,
    pub use_markers_to_optimize: bool,
    pub virtual_key: u16,
    pub id_thread: u32,
}

/// An opaque `sl::FrameToken` (a structure with a virtual table).
pub type FrameToken = c_void;

// The exported functions (`sl_core_api.h`): C++ references are pointers;
// `sl::Result` is an `int`.
pub type SlInit = unsafe extern "C" fn(*const Preferences, u64) -> i32;
pub type SlShutdown = unsafe extern "C" fn() -> i32;
pub type SlIsFeatureSupported = unsafe extern "C" fn(u32, *const AdapterInfo) -> i32;
pub type SlSetFeatureLoaded = unsafe extern "C" fn(u32, bool) -> i32;
pub type SlEvaluateFeature = unsafe extern "C" fn(u32, *const FrameToken, *const *const BaseStructure, u32, *mut c_void) -> i32;
pub type SlFreeResources = unsafe extern "C" fn(u32, *const ViewportHandle) -> i32;
pub type SlSetTagForFrame = unsafe extern "C" fn(*const FrameToken, *const ViewportHandle, *const ResourceTag, u32, *mut c_void) -> i32;
pub type SlGetFeatureRequirements = unsafe extern "C" fn(u32, *mut FeatureRequirements) -> i32;
pub type SlGetFeatureVersion = unsafe extern "C" fn(u32, *mut FeatureVersion) -> i32;
pub type SlUpgradeInterface = unsafe extern "C" fn(*mut *mut c_void) -> i32;
pub type SlSetConstants = unsafe extern "C" fn(*const Constants, *const FrameToken, *const ViewportHandle) -> i32;
pub type SlGetNativeInterface = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32;
pub type SlGetFeatureFunction = unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> i32;
pub type SlGetNewFrameToken = unsafe extern "C" fn(*mut *mut FrameToken, *const u32) -> i32;
pub type SlSetD3DDevice = unsafe extern "C" fn(*mut c_void) -> i32;

// Per-feature functions (`slGetFeatureFunction`).
pub type SlDlssGetOptimalSettings = unsafe extern "C" fn(*const DlssOptions, *mut DlssOptimalSettings) -> i32;
pub type SlDlssSetOptions = unsafe extern "C" fn(*const ViewportHandle, *const DlssOptions) -> i32;
pub type SlDlssgSetOptions = unsafe extern "C" fn(*const ViewportHandle, *const DlssgOptions) -> i32;
pub type SlDlssgGetState = unsafe extern "C" fn(*const ViewportHandle, *mut DlssgState, *const DlssgOptions) -> i32;
pub type SlReflexSetOptions = unsafe extern "C" fn(*const ReflexOptions) -> i32;
pub type SlPclSetMarker = unsafe extern "C" fn(u32, *const FrameToken) -> i32;

/// The name of an `sl::Result`, for the log.
pub fn result_name(code: i32) -> &'static str {
    const NAMES: [&str; 40] = [
        "ok",
        "IO error",
        "driver out of date",
        "OS out of date",
        "OS hardware scheduling disabled",
        "device not created",
        "no supported adapter found",
        "adapter not supported",
        "no plugins",
        "Vulkan API",
        "DXGI API",
        "D3D API",
        "NRD API",
        "NVAPI",
        "Reflex API",
        "NGX failed",
        "JSON parsing",
        "missing proxy",
        "missing resource state",
        "invalid integration",
        "missing input parameter",
        "not initialized",
        "compute failed",
        "init not called",
        "exception handler",
        "invalid parameter",
        "missing constants",
        "duplicated constants",
        "missing or invalid API",
        "common constants missing",
        "unsupported interface",
        "feature missing",
        "feature not supported",
        "feature missing hooks",
        "feature failed to load",
        "feature wrong priority",
        "feature missing dependency",
        "feature manager invalid state",
        "invalid state",
        "warning: out of VRAM",
    ];
    usize::try_from(code).ok().and_then(|i| NAMES.get(i)).copied().unwrap_or("unknown result")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The offsets `sl.api/sl.cpp` asserts (they may only grow at the end).
    #[test]
    fn offsets_match_the_sdk_asserts() {
        assert_eq!(size_of::<BaseStructure>(), 32);
        assert_eq!(offset_of!(Preferences, render_api), 136);
        assert_eq!(offset_of!(ResourceTag, extent), 48);
        assert_eq!(offset_of!(Resource, internal_flags), 104);
        assert_eq!(offset_of!(Resource, reserved), 106);
        assert_eq!(offset_of!(Constants, motion_vectors_jittered), 450);
        assert_eq!(offset_of!(FeatureVersion, version_ngx), 44);
        assert_eq!(offset_of!(FeatureRequirements, vk_num_optical_flow_queues_required), 176);
        assert_eq!(offset_of!(AdapterInfo, vk_physical_device), 48);
    }

    /// Fields the SDK does not assert, laid out as the headers' C++ would.
    #[test]
    fn feature_structures_lay_out_as_the_headers() {
        assert_eq!(offset_of!(Constants, jitter_offset), 352);
        assert_eq!(offset_of!(Constants, camera_near), 424);
        assert_eq!(offset_of!(Constants, depth_inverted), 444);
        assert_eq!(offset_of!(Constants, min_relative_linear_depth_object_separation), 452);
        assert_eq!(size_of::<Constants>(), 456);
        assert_eq!(size_of::<ViewportHandle>(), 40);
        assert_eq!(offset_of!(DlssOptions, color_buffers_hdr), 56);
        assert_eq!(offset_of!(DlssOptions, dlaa_preset), 60);
        assert_eq!(offset_of!(DlssOptions, use_auto_exposure), 84);
        assert_eq!(size_of::<DlssOptions>(), 88);
        assert_eq!(offset_of!(DlssOptimalSettings, render_height_max), 56);
        assert_eq!(offset_of!(DlssgOptions, on_error_callback), 96);
        assert_eq!(offset_of!(DlssgOptions, queue_parallelism_mode), 108);
        assert_eq!(offset_of!(DlssgOptions, dynamic_target_frame_rate), 116);
        assert_eq!(size_of::<DlssgOptions>(), 120);
        assert_eq!(offset_of!(DlssgState, num_frames_actually_presented), 48);
        assert_eq!(offset_of!(DlssgState, inputs_processing_completion_fence), 64);
        assert_eq!(offset_of!(DlssgState, is_dynamic_mfg_supported), 80);
        assert_eq!(offset_of!(ReflexOptions, virtual_key), 42);
        assert_eq!(size_of::<ReflexOptions>(), 48);
        assert_eq!(size_of::<Resource>(), 112);
        assert_eq!(size_of::<ResourceTag>(), 64);
    }

    #[test]
    fn the_sdk_version_is_2_14_16() {
        assert_eq!(SDK_VERSION >> 48, 2);
        assert_eq!((SDK_VERSION >> 32) & 0xffff, 14);
        assert_eq!((SDK_VERSION >> 16) & 0xffff, 16);
        assert_eq!(SDK_VERSION & 0xffff, 0xfedc);
        assert_eq!(result_name(0), "ok");
        assert_eq!(result_name(26), "missing constants");
        assert_eq!(result_name(39), "warning: out of VRAM");
        assert_eq!(result_name(-1), "unknown result");
    }
}
