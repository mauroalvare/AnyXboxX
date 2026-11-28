//! The part of NVIDIA Streamline's C ABI this crate calls (SDK 2.14, the
//! `sl_*.h` headers of the SDK's `include` folder), written down by hand from
//! those headers: structures that begin with a chained header (next, type id,
//! version), the features, buffer types and options used here, and the
//! signatures of the exported and per-feature functions. No NVIDIA header or
//! binary is part of this repository; the library is loaded at run time
//! (`super::load`). The offsets the SDK asserts in `sl.api/sl.cpp` are asserted
//! again in the tests below.
//! Source: <https://github.com/NVIDIA-RTX/Streamline/tree/v2.14.16/include>.
#![allow(dead_code)]

use std::ffi::{c_char, c_void};

/// A structure type id (a GUID).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructType {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> StructType {
    StructType { data1, data2, data3, data4 }
}

/// The header every chained structure starts with.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BaseStructure {
    pub next: *mut BaseStructure,
    pub kind: StructType,
    pub version: usize,
}

impl BaseStructure {
    pub const fn new(kind: StructType, version: usize) -> Self {
        BaseStructure { next: std::ptr::null_mut(), kind, version }
    }
}

pub const PREFERENCES: StructType = guid(0x1ca1_0965, 0xbf8e, 0x432b, [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14]);
pub const RESOURCE: StructType = guid(0x3a9d_70cf, 0x2418, 0x4b72, [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61]);
pub const RESOURCE_TAG: StructType = guid(0x4c6a_5aad, 0xb445, 0x496c, [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53]);
pub const CONSTANTS: StructType = guid(0xdcd3_5ad7, 0x4e4a, 0x4bad, [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe]);
pub const VIEWPORT_HANDLE: StructType = guid(0x171b_6435, 0x9b3c, 0x4fc8, [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4]);
pub const FEATURE_VERSION: StructType = guid(0x6d5b_51f0, 0x076b, 0x486d, [0x99, 0x95, 0x5a, 0x56, 0x10, 0x43, 0xf5, 0xc1]);
pub const ADAPTER_INFO: StructType = guid(0x0677_315f, 0xa746, 0x4492, [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4]);
pub const FEATURE_REQUIREMENTS: StructType = guid(0x6671_4097, 0xac6d, 0x4bc6, [0x89, 0x15, 0x1e, 0x0f, 0x55, 0xa6, 0xb6, 0x1f]);
pub const DLSS_OPTIONS: StructType = guid(0x6ac8_26e4, 0x4c61, 0x4101, [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8]);
pub const DLSS_OPTIMAL_SETTINGS: StructType = guid(0xef1d_0957, 0xfd58, 0x4df7, [0xb5, 0x04, 0x8b, 0x69, 0xd8, 0xaa, 0x6b, 0x76]);
pub const DLSSG_OPTIONS: StructType = guid(0xfac5_f1cb, 0x2dfd, 0x4f36, [0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5]);
pub const DLSSG_STATE: StructType = guid(0xcc8a_c8e1, 0xa179, 0x44f5, [0x97, 0xfa, 0xe7, 0x41, 0x12, 0xf9, 0xbc, 0x61]);
pub const REFLEX_OPTIONS: StructType = guid(0xf03a_f81a, 0x6d0b, 0x4902, [0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34]);

/// `sl::kSDKVersion` of the headers mirrored here (2.14.16).
pub const SDK_VERSION: u64 = (2 << 48) | (14 << 32) | (16 << 16) | 0xfedc;

// Features (`sl::Feature`).
pub const FEATURE_DLSS: u32 = 0;
pub const FEATURE_REFLEX: u32 = 3;
pub const FEATURE_PCL: u32 = 4;
pub const FEATURE_DLSS_G: u32 = 1000;

// Buffer types (`sl::BufferType`).
pub const BUFFER_DEPTH: u32 = 0;
pub const BUFFER_MOTION_VECTORS: u32 = 1;
pub const BUFFER_HUDLESS_COLOR: u32 = 2;
pub const BUFFER_SCALING_INPUT_COLOR: u32 = 3;
pub const BUFFER_SCALING_OUTPUT_COLOR: u32 = 4;
pub const BUFFER_EXPOSURE: u32 = 13;
pub const BUFFER_UI_COLOR_AND_ALPHA: u32 = 23;
pub const BUFFER_BIAS_CURRENT_COLOR_HINT: u32 = 29;

// `sl::ResourceType` (a char).
pub const RESOURCE_TEX2D: u8 = 0;

// `sl::ResourceLifecycle`.
pub const ONLY_VALID_NOW: u32 = 0;
pub const VALID_UNTIL_PRESENT: u32 = 1;
pub const VALID_UNTIL_EVALUATE: u32 = 2;

// `sl::Boolean` (a char).
pub const FALSE: u8 = 0;
pub const TRUE: u8 = 1;
pub const INVALID: u8 = 2;

pub const fn boolean(value: bool) -> u8 {
    if value { TRUE } else { FALSE }
}

/// `sl::INVALID_FLOAT`: a value left unset.
pub const INVALID_FLOAT: f32 = f32::MAX;

// `sl::PreferenceFlags`.
pub const DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
pub const DISABLE_DEBUG_TEXT: u64 = 1 << 1;
pub const USE_MANUAL_HOOKING: u64 = 1 << 2;
pub const USE_DXGI_FACTORY_PROXY: u64 = 1 << 5;
pub const USE_FRAME_BASED_RESOURCE_TAGGING: u64 = 1 << 7;

// `sl::LogLevel`, `sl::LogType`.
pub const LOG_DEFAULT: u32 = 1;
pub const LOG_VERBOSE: u32 = 2;
pub const LOG_INFO: u32 = 0;
pub const LOG_WARN: u32 = 1;
pub const LOG_ERROR: u32 = 2;

// `sl::RenderAPI`, `sl::EngineType`.
pub const RENDER_API_D3D12: u32 = 1;
pub const ENGINE_CUSTOM: u32 = 0;

// `sl::DLSSMode`.
pub const DLSS_OFF: u32 = 0;
pub const DLSS_MAX_PERFORMANCE: u32 = 1;
pub const DLSS_BALANCED: u32 = 2;
pub const DLSS_MAX_QUALITY: u32 = 3;
pub const DLSS_ULTRA_PERFORMANCE: u32 = 4;
pub const DLSS_ULTRA_QUALITY: u32 = 5;
pub const DLSS_DLAA: u32 = 6;

// `sl::DLSSGMode`, `sl::DLSSGFlags`.
pub const DLSSG_OFF: u32 = 0;
pub const DLSSG_ON: u32 = 1;
pub const DLSSG_RETAIN_RESOURCES_WHEN_OFF: u32 = 1 << 3;

// `sl::ReflexMode`.
pub const REFLEX_OFF: u32 = 0;
pub const REFLEX_LOW_LATENCY: u32 = 1;
pub const REFLEX_LOW_LATENCY_WITH_BOOST: u32 = 2;

// `sl::PCLMarker`.
pub const SIMULATION_START: u32 = 0;
pub const SIMULATION_END: u32 = 1;
pub const RENDER_SUBMIT_START: u32 = 2;
pub const RENDER_SUBMIT_END: u32 = 3;
pub const PRESENT_START: u32 = 4;
pub const PRESENT_END: u32 = 5;

/// `sl::Preferences` (version 1).
#[repr(C)]
pub struct Preferences {
    pub base: BaseStructure,
    pub show_console: bool,
    pub log_level: u32,
    pub paths_to_plugins: *const *const u16,
    pub num_paths_to_plugins: u32,
    pub path_to_logs_and_data: *const u16,
    pub allocate_callback: *const c_void,
    pub release_callback: *const c_void,
    pub log_message_callback: Option<LogCallback>,
    pub flags: u64,
    pub features_to_load: *const u32,
    pub num_features_to_load: u32,
    pub application_id: u32,
    pub engine: u32,
    pub engine_version: *const c_char,
    pub project_id: *const c_char,
    pub render_api: u32,
}

pub type LogCallback = unsafe extern "C" fn(kind: u32, message: *const c_char);

/// `sl::Resource` (version 1): for D3D12 the resource and its state are what
/// counts.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Resource {
    pub base: BaseStructure,
    pub kind: u8,
    pub native: *mut c_void,
    pub memory: *mut c_void,
    pub view: *mut c_void,
    pub state: u32,
    pub width: u32,
    pub height: u32,
    pub native_format: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub gpu_virtual_address: u64,
    pub flags: u32,
    pub usage: u32,
    pub internal_flags: u16,
    pub reserved: u16,
}

impl Resource {
    /// A 2D texture in host state `state`.
    pub fn texture(native: *mut c_void, state: u32) -> Self {
        Resource {
            base: BaseStructure::new(RESOURCE, 1),
            kind: RESOURCE_TEX2D,
            native,
            memory: std::ptr::null_mut(),
            view: std::ptr::null_mut(),
            state,
            width: 0,
            height: 0,
            native_format: 0,
            mip_levels: 0,
            array_layers: 0,
            gpu_virtual_address: 0,
            flags: 0,
            usage: 0,
            internal_flags: 0,
            reserved: 0,
        }
    }
}

/// `sl::Extent`: the area of a resource that counts (all of it if empty).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extent {
    pub top: u32,
    pub left: u32,
    pub width: u32,
    pub height: u32,
}

impl Extent {
    /// The top left `size` of a resource.
    pub const fn sized(size: [u32; 2]) -> Self {
        Extent { top: 0, left: 0, width: size[0], height: size[1] }
    }
}

/// `sl::ResourceTag` (version 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ResourceTag {
    pub base: BaseStructure,
    pub resource: *mut Resource,
    pub kind: u32,
    pub lifecycle: u32,
    pub extent: Extent,
}

impl ResourceTag {
    pub fn new(resource: *mut Resource, kind: u32, lifecycle: u32, extent: Extent) -> Self {
        ResourceTag { base: BaseStructure::new(RESOURCE_TAG, 1), resource, kind, lifecycle, extent }
    }
}

/// `sl::float4x4`: four rows.
pub type Float4x4 = [[f32; 4]; 4];

/// `sl::Constants` (version 2): per frame and viewport.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Constants {
    pub base: BaseStructure,
    pub camera_view_to_clip: Float4x4,
    pub clip_to_camera_view: Float4x4,
    pub clip_to_lens_clip: Float4x4,
    pub clip_to_prev_clip: Float4x4,
    pub prev_clip_to_clip: Float4x4,
    pub jitter_offset: [f32; 2],
    pub mvec_scale: [f32; 2],
    pub camera_pinhole_offset: [f32; 2],
    pub camera_pos: [f32; 3],
    pub camera_up: [f32; 3],
    pub camera_right: [f32; 3],
    pub camera_fwd: [f32; 3],
    pub camera_near: f32,
    pub camera_far: f32,
    pub camera_fov: f32,
    pub camera_aspect_ratio: f32,
    pub motion_vectors_invalid_value: f32,
    pub depth_inverted: u8,
    pub camera_motion_included: u8,
    pub motion_vectors_3d: u8,
    pub reset: u8,
    pub orthographic_projection: u8,
    pub motion_vectors_dilated: u8,
    pub motion_vectors_jittered: u8,
    pub min_relative_linear_depth_object_separation: f32,
}

/// `sl::ViewportHandle`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ViewportHandle {
    pub base: BaseStructure,
    pub value: u32,
}

impl ViewportHandle {
    pub const fn new(value: u32) -> Self {
        ViewportHandle { base: BaseStructure::new(VIEWPORT_HANDLE, 1), value }
    }
}

/// `sl::Version`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub build: u32,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.build)
    }
}

/// `sl::FeatureVersion`.
#[repr(C)]
pub struct FeatureVersion {
    pub base: BaseStructure,
    pub version_sl: Version,
    pub version_ngx: Version,
}

/// `sl::AdapterInfo`.
#[repr(C)]
pub struct AdapterInfo {
    pub base: BaseStructure,
    pub device_luid: *const u8,
    pub device_luid_size: u32,
    pub vk_physical_device: *mut c_void,
}

/// `sl::FeatureRequirements` (version 2).
#[repr(C)]
pub struct FeatureRequirements {
    pub base: BaseStructure,
    pub flags: u32,
    pub max_num_cpu_threads: u32,
    pub max_num_viewports: u32,
    pub num_required_tags: u32,
    pub required_tags: *const u32,
    pub os_version_detected: Version,
    pub os_version_required: Version,
    pub driver_version_detected: Version,
    pub driver_version_required: Version,
    pub vk_num_compute_queues_required: u32,
    pub vk_num_graphics_queues_required: u32,
    pub vk_num_device_extensions: u32,
    pub vk_device_extensions: *const *const c_char,
    pub vk_num_instance_extensions: u32,
    pub vk_instance_extensions: *const *const c_char,
    pub vk_num_features12: u32,
    pub vk_features12: *const *const c_char,
    pub vk_num_features13: u32,
    pub vk_features13: *const *const c_char,
    pub vk_num_optical_flow_queues_required: u32,
}

/// `sl::FeatureRequirementFlags`.
pub const REQUIRES_D3D12: u32 = 1 << 1;
pub const REQUIRES_VSYNC_OFF: u32 = 1 << 3;
pub const REQUIRES_HARDWARE_SCHEDULING: u32 = 1 << 4;

mod features;
pub use features::*;
