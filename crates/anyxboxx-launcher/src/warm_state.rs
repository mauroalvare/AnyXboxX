//! Is the shader cache ready for this VIctory build and this environment ("smart warm")?
//!
//! After a successful `victory warm` the launcher writes a marker holding a fingerprint: the
//! build of victory.exe, the VICTORY_* variables (the cache key depends on them) and the game's
//! archives. The next launch compares the fingerprint of the moment with the marker: a missing,
//! unreadable or different marker means the warm runs again before the game starts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyxboxx::run::Env;

pub const MARKER_FILE: &str = "warm-state.txt";
/// `VICTORY_*` variables that affect `victory warm`: cache location/key, compute pipeline filters
/// and root/descriptor layout (`victory-gpu/src`), host wave/view overrides, and shader translator
/// switches (`third_party/victory/crates/victory-shader`). Runtime-only options (resolution,
/// window mode, keyboard, overlay, Auto-A, frametime dump, presets, relay, XG DLL, log level)
/// leave the warmed cache unchanged and must not force a redundant ~7-minute re-warm.
///
/// The guard test in `warm_state_tests.rs` scans both `victory-shader/src` and `victory-gpu/src`
/// so any newly added `VICTORY_*` switch fails CI until explicitly classified here or as
/// runtime-only.
const CACHE_KEYS: &[&str] = &[
    // Cache location and cache-key mode (`victory-core/src/config.rs`, `shader_cache.rs`).
    "VICTORY_CACHE_DIR",
    "VICTORY_CONFIG",
    "VICTORY_SHADER_KEY",
    "VICTORY_SHADER_FRESH",
    // Compute pipelines, root layout, descriptor layout, and device selection (`victory-gpu/src`).
    "VICTORY_ACTIVATE_RAYS",
    "VICTORY_BVH_BUILDER",
    "VICTORY_BVH_QUARANTINE",
    "VICTORY_D3D12_WARP",
    "VICTORY_DYNAMIC_SLOTS",
    "VICTORY_EARLY_ALL",
    "VICTORY_PIPELINE_HISTORY",
    "VICTORY_RAW_VIEWS",
    "VICTORY_ROOT_SRV_DIRECT",
    "VICTORY_RT_RESOLVE",
    "VICTORY_SKIP_PIPELINES",
    // Host wave/view overrides that enter `translation_fingerprint`.
    "VICTORY_HOST_MAX_WAVE",
    "VICTORY_HOST_WAVE",
    "VICTORY_VIEW_LIMIT",
    "VICTORY_WARP",
    // Shader translator switches (`victory-shader/src`, checked by a test against the tree).
    "VICTORY_DESC_DATA",
    "VICTORY_F16_NEAREST",
    "VICTORY_FIRSTLANE_OWN",
    "VICTORY_FIRST_HALF",
    "VICTORY_GROUP_SYNC",
    "VICTORY_MESH_OUTPUT_STRICT",
    "VICTORY_SCC_LOCAL_ANY",
    "VICTORY_SHADER_LOOP_LIMIT",
    "VICTORY_SHADER_OPT",
    "VICTORY_STRICT_JOINS",
    "VICTORY_WAVE32_NVIDIA",
    "VICTORY_WAVE_DEEP_SKIPS",
    "VICTORY_WAVE_LDS_LOOSE",
    "VICTORY_WAVE_LOCAL",
    "VICTORY_WAVE_OWN_BITS",
    "VICTORY_WAVE_PAIRS",
    "VICTORY_WAVE_WIDE",
    "VICTORY_BVH_BOUNDS",
    "VICTORY_BVH_FENCES",
    "VICTORY_HELPER_OWN",
    "VICTORY_LDS_FLAGS",
    "VICTORY_NAN_GUARD",
    "VICTORY_PC_DXC",
    "VICTORY_PORT_CBV",
    "VICTORY_RDNA_FLOAT",
    "VICTORY_THREAD_TICKET",
    "VICTORY_WAVE_WIDE_ONLY",
];

/// Whether `key` can change what `victory warm` writes into the shader or pipeline cache.
pub fn affects_shader_cache(key: &str) -> bool {
    CACHE_KEYS.contains(&key)
}

/// `%LOCALAPPDATA%\AnyXboxX\launcher\warm-state.txt`, `None` when LOCALAPPDATA is not set or empty.
pub fn marker_path(local_app_data: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    local_app_data.filter(|v| !v.is_empty()).map(|local| {
        PathBuf::from(local)
            .join("AnyXboxX")
            .join("launcher")
            .join(MARKER_FILE)
    })
}

/// FNV-1a, 64 bits: a stable fingerprint without a new dependency (not a security hash).
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Identity of victory.exe: its size and modification time in seconds. A missing file (not built
/// yet) has its own identity, so the first build always counts as a change.
pub fn build_id(exe: &Path) -> String {
    match std::fs::metadata(exe) {
        Ok(meta) => {
            let secs = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            format!("{}:{secs}", meta.len())
        }
        Err(_) => "not-built".to_string(),
    }
}

/// The cache-relevant variables, sorted and deduplicated (last entry wins), one `KEY=VALUE` per line.
pub fn env_key(env: &Env) -> String {
    let mut effective: BTreeMap<&str, String> = BTreeMap::new();
    for (key, value) in env {
        if affects_shader_cache(key.as_str()) {
            effective.insert(key.as_str(), value.to_string_lossy().into_owned());
        }
    }
    effective
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `name:size:modified` of each archive, in the order given.
pub fn archive_identities(archives: &[PathBuf]) -> Vec<String> {
    archives
        .iter()
        .map(|path| {
            let name = path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            match std::fs::metadata(path) {
                Ok(meta) => {
                    let secs = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_secs());
                    format!("{name}:{}:{secs}", meta.len())
                }
                Err(_) => format!("{name}:missing"),
            }
        })
        .collect()
}

/// The fingerprint of build, environment and archives, as a 16-digit hex string.
pub fn fingerprint(build: &str, env: &Env, archives: &[String]) -> String {
    let text = format!(
        "build={build}\nenv:\n{}\narchives:\n{}",
        env_key(env),
        archives.join("\n")
    );
    format!("{:016x}", fnv1a(&text))
}

/// Whether the warm must run: always when asked to, otherwise when the marker is missing or
/// holds another fingerprint (`marker` is the file's text, `None` when it cannot be read).
pub fn needs_warm(always: bool, marker: Option<&str>, fingerprint: &str) -> bool {
    always || marker.map(str::trim) != Some(fingerprint)
}

/// Reads the marker file; `None` when it is missing or unreadable.
pub fn read_marker(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Writes the fingerprint after a successful warm, creating the folder when needed.
pub fn write_marker(path: &Path, fingerprint: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, format!("{fingerprint}\n"))
}

/// Deletes the marker (the shader cache was cleaned): the next launch warms again. A missing
/// marker is not an error.
pub fn forget_marker(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "warm_state_tests.rs"]
mod tests;
