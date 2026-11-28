use super::*;
use std::ffi::OsString;

/// Explicit non-cache `VICTORY_*` literals in `victory-shader/src`: compile-time keys,
/// diagnostic-only loggers, and test-only shader overrides. Any new switch in
/// `victory-shader/src` fails the guard test until classified into `CACHE_KEYS` or here.
const NON_CACHE_SHADER_KEYS: &[&str] = &[
    "VICTORY_DESC_WHY",
    "VICTORY_REF_PS",
    "VICTORY_REF_VS",
    "VICTORY_SHADER_SOURCE_HASH",
    "VICTORY_SHADER_SWITCHES",
    "VICTORY_WAVE_DUMP",
    "VICTORY_WAVE_WHY",
];

/// Explicit runtime-only, diagnostic, memory-budget, and test `VICTORY_*` literals in
/// `victory-gpu/src` that do not change compiled pipelines or the warmed shader cache.
/// Any new switch in `victory-gpu/src` fails the guard test until classified into
/// `CACHE_KEYS` or here (fail-closed).
const NON_CACHE_GPU_KEYS: &[&str] = &[
    "VICTORY_A",
    "VICTORY_AB",
    "VICTORY_ALIAS_COPY",
    "VICTORY_AMBIENT",
    "VICTORY_AMBIENT_AIR",
    "VICTORY_AMBIENT_RT",
    "VICTORY_AMBIENT_SETTLE",
    "VICTORY_AMBIENT_SHADE",
    "VICTORY_ASK_EARLY",
    "VICTORY_ASK_NORESET",
    "VICTORY_ASK_ONCE",
    "VICTORY_ASK_SPREAD",
    "VICTORY_ASYNC_SUBMIT",
    "VICTORY_B",
    "VICTORY_BAKE_EXACT",
    "VICTORY_BAKE_TEXTURES_MS",
    "VICTORY_BARRIER_TEST",
    "VICTORY_BENCH_CS",
    "VICTORY_BENCH_FILE",
    "VICTORY_BENCH_HOLES",
    "VICTORY_BENCH_N",
    "VICTORY_BENCH_S",
    "VICTORY_BENCH_THREADS",
    "VICTORY_BENCH_WARP",
    "VICTORY_BIND_CACHE",
    "VICTORY_BLOCK_WAY",
    "VICTORY_BOOK",
    "VICTORY_BUILD",
    "VICTORY_BURY_LAG",
    "VICTORY_BVH_FAULTS_ALL",
    "VICTORY_BVH_SIZES_WAIT",
    "VICTORY_BVH_TRACE",
    "VICTORY_CACHE_CORES",
    "VICTORY_CAPTURE",
    "VICTORY_CAPTURE_FRAMES",
    "VICTORY_CAPTURE_WORLD",
    "VICTORY_CAPTURE_WORLD_ALSO",
    "VICTORY_CARD_ACCOUNTING",
    "VICTORY_CARD_ALL",
    "VICTORY_CARD_AWAKE",
    "VICTORY_CARD_BUFFERS",
    "VICTORY_CARD_BUFFERS_MB",
    "VICTORY_CARD_BUFFERS_MIN_KB",
    "VICTORY_CARD_BUFFERS_MIN_MB",
    "VICTORY_CARD_HASH_LEN",
    "VICTORY_CARD_REHASH_RUNS",
    "VICTORY_CARD_RESERVE_BUDGETS",
    "VICTORY_CARD_ROOM_FOR_BUFFERS",
    "VICTORY_CARD_ROOM_FOR_TARGETS_MB",
    "VICTORY_CARD_SPANS",
    "VICTORY_CARD_STEPS",
    "VICTORY_CARD_WINDOWS_MB",
    "VICTORY_COMMITTED_QUERY",
    "VICTORY_COPIES_IN_PLACE",
    "VICTORY_COPY_ASK",
    "VICTORY_COPY_BESIDE",
    "VICTORY_COPY_MODE",
    "VICTORY_COPY_OVER_TITLE",
    "VICTORY_CUT_WAITS",
    "VICTORY_CUT_WRITES",
    "VICTORY_D3D12_BACKGROUND",
    "VICTORY_D3D12_DEBUG",
    "VICTORY_D3D12_DRED",
    "VICTORY_D3D12_GBV",
    "VICTORY_DECLARE_ONCE",
    "VICTORY_DEPTH_TWINS",
    "VICTORY_DIRECT_WRITES",
    "VICTORY_DISPLAY_MODE",
    "VICTORY_DUMP_TARGETS",
    "VICTORY_EARLY_TEXTURES_MS",
    "VICTORY_EXACT_BACK",
    "VICTORY_FAST_CORES",
    "VICTORY_FAULT_BOOK",
    "VICTORY_FAULT_BOOK_AT",
    "VICTORY_FETCH_CHECK",
    "VICTORY_FILL_TITLE",
    "VICTORY_FRAMETIME_DUMP",
    "VICTORY_FRAME_INTERVAL_MS",
    "VICTORY_FULL_CARD_PASS",
    "VICTORY_GPU_LOAD_MS",
    "VICTORY_GPU_TIMING",
    "VICTORY_GUARD_ASK",
    "VICTORY_GUARD_HOLES",
    "VICTORY_GUARD_HOT",
    "VICTORY_GUARD_REST",
    "VICTORY_HEAP_PASS_ASIDE",
    "VICTORY_HEAP_RECHECK",
    "VICTORY_HEAP_SPREAD",
    "VICTORY_HEAP_SYNC",
    "VICTORY_HEAP_SYNC_ASIDE",
    "VICTORY_HEAP_SYNC_EARLY",
    "VICTORY_HEAP_TEXTURES",
    "VICTORY_HELD_LEVELS",
    "VICTORY_INDIRECT_BATCH",
    "VICTORY_INDIRECT_CARD_BUDGET",
    "VICTORY_INDIRECT_REST",
    "VICTORY_INDIRECT_WIDE",
    "VICTORY_INDIRECT_WINDOW",
    "VICTORY_INPLACE",
    "VICTORY_INPLACE_MB",
    "VICTORY_INPLACE_PROBE",
    "VICTORY_INPLACE_PROBE_MB",
    "VICTORY_JOIN_BARRIERS",
    "VICTORY_JOIN_COPIES",
    "VICTORY_JOIN_LISTS",
    "VICTORY_JOIN_PARTS",
    "VICTORY_KEEP_FREED",
    "VICTORY_KEEP_SHOWN",
    "VICTORY_KEEP_SNAPSHOTS",
    "VICTORY_KEEP_SNAPSHOTS_MB",
    "VICTORY_KEEP_VIEWS",
    "VICTORY_KNOWN_HOLES",
    "VICTORY_LAND",
    "VICTORY_LAND_ANYTIME",
    "VICTORY_LAND_BESIDE",
    "VICTORY_LAND_ECHO",
    "VICTORY_LAND_SKIP",
    "VICTORY_LAND_TAKEN",
    "VICTORY_LAND_THREADS",
    "VICTORY_LANE_CORE",
    "VICTORY_LATE_WAIT",
    "VICTORY_LATE_WAITS",
    "VICTORY_LATE_WAIT_LISTS",
    "VICTORY_LATE_WAIT_WRITES",
    "VICTORY_LAZY_PRELUDE",
    "VICTORY_LAZY_TEXTURES",
    "VICTORY_LENT_HEAP",
    "VICTORY_LEVELS_MIN_KB",
    "VICTORY_LEVELS_ON_DEMAND",
    "VICTORY_LIBRARY_HANDLE",
    "VICTORY_LIBRARY_LOCK",
    "VICTORY_LIBRARY_MAP",
    "VICTORY_LIBRARY_TRIM",
    "VICTORY_LIGHT_PROBE",
    "VICTORY_LIGHT_PROBE_LUX",
    "VICTORY_LIGHT_REFRESH",
    "VICTORY_LIST_POOL",
    "VICTORY_LIST_SLOTS",
    "VICTORY_LIST_VIEWS",
    "VICTORY_MAPPED_BY_PAGE",
    "VICTORY_MAPPED_BY_VIEWS",
    "VICTORY_MEMORY_CARD_MB",
    "VICTORY_MEMORY_CATEGORIES",
    "VICTORY_MEMORY_DUMP",
    "VICTORY_MEMORY_MAP",
    "VICTORY_MESH_TEST",
    "VICTORY_METER_THREAD",
    "VICTORY_NONE_SUCH",
    "VICTORY_OPEN_ONE_VIEW",
    "VICTORY_OVERLAY",
    "VICTORY_OWNERLESS_VIEWS",
    "VICTORY_PAGE_PARTS",
    "VICTORY_PAGE_WRITES",
    "VICTORY_PANEL",
    "VICTORY_PICKED_MB",
    "VICTORY_PICKED_WRITES",
    "VICTORY_PIPELINE_BOOK",
    "VICTORY_PIPELINE_LOWEST",
    "VICTORY_PIPELINE_SLOW",
    "VICTORY_PIPELINE_WAIT_MS",
    "VICTORY_PIPELINE_WAIT_SHARED",
    "VICTORY_PIPELINE_WORKERS",
    "VICTORY_PLACED_FILLER",
    "VICTORY_PLACED_ORDER",
    "VICTORY_PLACED_READERS",
    "VICTORY_PLANES",
    "VICTORY_POOL_SMALL",
    "VICTORY_POP_ALL",
    "VICTORY_POP_FOLDER",
    "VICTORY_POP_WATCH",
    "VICTORY_PORT_CBV_PLACE",
    "VICTORY_PORT_CONSTANTS",
    "VICTORY_PORT_CONSTANTS_ORDER",
    "VICTORY_PORT_CONSTANTS_ORDER_ALONE",
    "VICTORY_PORT_DESCRIPTORS",
    "VICTORY_PORT_DESCRIPTORS_CHECK",
    "VICTORY_PORT_DESCRIPTORS_EVERY",
    "VICTORY_PORT_GUARD",
    "VICTORY_PORT_GUARD_PLAN",
    "VICTORY_PORT_JOURNAL",
    "VICTORY_QUEUE_WAIT",
    "VICTORY_QUICK_FIND",
    "VICTORY_QUICK_MIRROR",
    "VICTORY_READY_ONCE",
    "VICTORY_READ_BACK",
    "VICTORY_RECHECK_SOON",
    "VICTORY_RECORD",
    "VICTORY_RECORD_ALWAYS",
    "VICTORY_RECORD_AT",
    "VICTORY_RECORD_FRAMES",
    "VICTORY_REFRESH_AHEAD",
    "VICTORY_REFRESH_CHECK",
    "VICTORY_REFRESH_FRONT",
    "VICTORY_RELAY",
    "VICTORY_RELAY_COPY",
    "VICTORY_REPLAY_GBV",
    "VICTORY_REPLAY_PEEK",
    "VICTORY_RESIDENCY_LIMITED",
    "VICTORY_REUPLOAD_WHEN_FULL",
    "VICTORY_RIPE_BLOCKS",
    "VICTORY_ROOT_UAV_MB",
    "VICTORY_ROOT_VIEW_MB",
    "VICTORY_RTV_FORMAT_LOG",
    "VICTORY_SAFE_COPY",
    "VICTORY_SAME_RECORDS",
    "VICTORY_SCISSOR_TEST",
    "VICTORY_SCRATCH_POOL",
    "VICTORY_SCREEN_WAIT_MS",
    "VICTORY_SEND_BAR",
    "VICTORY_SEND_CHANGED",
    "VICTORY_SEND_OWN",
    "VICTORY_SHADER_SAVE",
    "VICTORY_SHOW_AFTER",
    "VICTORY_SHOW_NEW_LATER",
    "VICTORY_SLOT_CHECK",
    "VICTORY_SLOT_HOLDS",
    "VICTORY_SNAPSHOT_POOL",
    "VICTORY_SPANS_QUICK",
    "VICTORY_SPLIT_WAITS",
    "VICTORY_STALE_TEXTURES",
    "VICTORY_STORE_BESIDE",
    "VICTORY_STRUCTURED_RAW",
    "VICTORY_SUBMIT_DEPTH",
    "VICTORY_SUBMIT_FRAMES",
    "VICTORY_SWEEP",
    "VICTORY_SYNC_FINE",
    "VICTORY_SYNC_IN_PLACE",
    "VICTORY_TABLED_TEXTURES",
    "VICTORY_TAKE_PARTS",
    "VICTORY_TEST_COLD_PARALLEL_UPLOADS",
    "VICTORY_TEST_ROUNDS",
    "VICTORY_TEXTURE_AWAY_MS",
    "VICTORY_TEXTURE_BACK",
    "VICTORY_TEXTURE_BUDGET_MB",
    "VICTORY_TEXTURE_CATCHUP",
    "VICTORY_TEXTURE_COPY",
    "VICTORY_TEXTURE_FILL",
    "VICTORY_TEXTURE_GUARD",
    "VICTORY_TEXTURE_HOT",
    "VICTORY_TEXTURE_KEEP",
    "VICTORY_TEXTURE_LOOK_MS",
    "VICTORY_TEXTURE_NO_WAIT",
    "VICTORY_TEXTURE_ONCE",
    "VICTORY_TEXTURE_POOL",
    "VICTORY_TEXTURE_ROOM_PERCENT",
    "VICTORY_TEXTURE_SAME",
    "VICTORY_TEXTURE_STAGE_MB",
    "VICTORY_TEXTURE_STAGE_MS",
    "VICTORY_TEXTURE_STORE",
    "VICTORY_TEXTURE_SWEEP",
    "VICTORY_THIN",
    "VICTORY_THIN_THREADS",
    "VICTORY_THREAD_DIMENSIONS",
    "VICTORY_TILES_TEST_CARD",
    "VICTORY_TIMELINE",
    "VICTORY_TIMELINE_AT",
    "VICTORY_TIMELINE_FRAMES",
    "VICTORY_TIMER_LOG",
    "VICTORY_TRAIL",
    "VICTORY_TRANSLATOR",
    "VICTORY_UNDER_FUSED",
    "VICTORY_UNDER_THREADS",
    "VICTORY_UNDER_WIDE",
    "VICTORY_UNDER_WORDS",
    "VICTORY_UNWATCHED_FRAMES",
    "VICTORY_UPCOPY_CHECK",
    "VICTORY_UPLOAD_AHEAD",
    "VICTORY_UPLOAD_LOW",
    "VICTORY_UP_QUEUE",
    "VICTORY_VIEW_BYTES",
    "VICTORY_VIEW_COPY",
    "VICTORY_VIEW_EXTENT",
    "VICTORY_VIEW_GUARD",
    "VICTORY_VIEW_INLINE",
    "VICTORY_VIEW_THREADS",
    "VICTORY_WAIT_ASK",
    "VICTORY_WAIT_FORCE",
    "VICTORY_WAIT_RESIDENT",
    "VICTORY_WALLS_TRACE",
    "VICTORY_WIDE_BACK",
    "VICTORY_WIDE_COPY",
    "VICTORY_WIDE_FILLS",
    "VICTORY_WIDE_MERGE",
    "VICTORY_WIDE_REFRESH",
    "VICTORY_WIDE_SPIN",
    "VICTORY_WIDE_TEXTURES",
    "VICTORY_WINDOWS_BAR_MB",
    "VICTORY_WINDOWS_OWN",
    "VICTORY_WINDOW_BUDGET_MB",
    "VICTORY_WINDOW_KB",
    "VICTORY_WINDOW_MODE",
    "VICTORY_WINDOW_TAILS",
    "VICTORY_WRITE_SITES",
    "VICTORY_WRITE_SKIP",
];

#[test]
fn a_forgotten_marker_makes_the_next_launch_warm() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("warm-state.txt");
    write_marker(&path, "fp").expect("write");
    assert!(!needs_warm(false, read_marker(&path).as_deref(), "fp"));
    forget_marker(&path).expect("forget");
    assert!(needs_warm(false, read_marker(&path).as_deref(), "fp"));
    forget_marker(&path).expect("a missing marker is fine");
}

fn env(pairs: &[(&str, &str)]) -> Env {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), OsString::from(*v)))
        .collect()
}

fn archives() -> Vec<String> {
    vec!["a.rpf:10:5".to_string()]
}

#[test]
fn a_missing_marker_means_the_warm_runs() {
    let fp = fingerprint("1:2", &env(&[("VICTORY_RELAY", "2")]), &archives());
    assert!(needs_warm(false, None, &fp));
}

#[test]
fn the_same_fingerprint_in_the_marker_skips_the_warm() {
    let fp = fingerprint("1:2", &env(&[("VICTORY_RELAY", "2")]), &archives());
    assert!(!needs_warm(false, Some(&format!("{fp}\n")), &fp));
}

#[test]
fn a_different_marker_means_the_warm_runs() {
    let fp = fingerprint("1:2", &env(&[]), &archives());
    assert!(needs_warm(false, Some("0000000000000000"), &fp));
}

#[test]
fn always_re_warm_ignores_a_current_marker() {
    let fp = fingerprint("1:2", &env(&[]), &archives());
    assert!(needs_warm(true, Some(&fp), &fp));
}

#[test]
fn a_new_victory_build_changes_the_fingerprint() {
    let before = fingerprint("100:1", &env(&[]), &archives());
    let after = fingerprint("100:2", &env(&[]), &archives());
    assert_ne!(before, after);
}

#[test]
fn a_changed_cache_variable_changes_the_fingerprint() {
    let before = fingerprint("1:1", &env(&[("VICTORY_SHADER_OPT", "1")]), &archives());
    let after = fingerprint("1:1", &env(&[("VICTORY_SHADER_OPT", "0")]), &archives());
    assert_ne!(before, after);
}

#[test]
fn the_game_folder_and_log_level_do_not_change_the_fingerprint() {
    let base = fingerprint("1:1", &env(&[("VICTORY_SHADER_OPT", "1")]), &archives());
    let moved = fingerprint(
        "1:1",
        &env(&[
            ("VICTORY_SHADER_OPT", "1"),
            ("VICTORY_GAME_DIR", "D:\\Other"),
            ("VICTORY_LOG", "debug"),
        ]),
        &archives(),
    );
    assert_eq!(base, moved);
}

#[test]
fn non_cache_variables_are_not_part_of_the_key() {
    assert_eq!(
        env_key(&env(&[
            ("PATH", "x"),
            ("VKD3D_CONFIG", "y"),
            ("VICTORY_RELAY", "2"),
            ("VICTORY_DISPLAY_MODE", "1920x1080"),
        ])),
        ""
    );
    assert_eq!(
        env_key(&env(&[
            ("VICTORY_SHADER_OPT", "2"),
            ("VICTORY_GROUP_SYNC", "1")
        ])),
        "VICTORY_GROUP_SYNC=1\nVICTORY_SHADER_OPT=2"
    );
}

#[test]
fn a_changed_archive_changes_the_fingerprint() {
    let before = fingerprint("1:1", &env(&[]), &["a.rpf:10:5".to_string()]);
    let after = fingerprint("1:1", &env(&[]), &["a.rpf:11:5".to_string()]);
    assert_ne!(before, after);
}

#[test]
fn the_marker_round_trips_through_its_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir
        .path()
        .join("AnyXboxX")
        .join("launcher")
        .join(MARKER_FILE);
    assert!(read_marker(&path).is_none());
    write_marker(&path, "abc").expect("write");
    let text = read_marker(&path).expect("read");
    assert!(!needs_warm(false, Some(&text), "abc"));
}

#[test]
fn the_marker_lives_under_appdata() {
    let path = marker_path(Some(std::ffi::OsStr::new(r"C:\L"))).expect("path");
    assert!(path.ends_with(Path::new("AnyXboxX").join("launcher").join(MARKER_FILE)));
    assert!(marker_path(None).is_none());
    assert!(marker_path(Some(std::ffi::OsStr::new(""))).is_none());
}

#[test]
fn overridden_victory_variables_keep_only_the_last_value() {
    let overridden = env(&[
        ("VICTORY_SHADER_OPT", "2"),
        ("VICTORY_GROUP_SYNC", "1"),
        ("VICTORY_SHADER_OPT", "0"),
    ]);
    let direct = env(&[("VICTORY_GROUP_SYNC", "1"), ("VICTORY_SHADER_OPT", "0")]);
    assert_eq!(
        env_key(&overridden),
        "VICTORY_GROUP_SYNC=1\nVICTORY_SHADER_OPT=0"
    );
    assert_eq!(
        fingerprint("1:1", &overridden, &archives()),
        fingerprint("1:1", &direct, &archives())
    );
}

fn scan_victory_switches(
    src_root: &Path,
    skip_file: impl Fn(&str) -> bool,
    non_cache: &[&str],
    crate_label: &str,
) {
    if !src_root.is_dir() {
        return;
    }
    let mut files = Vec::new();
    collect_rs(src_root, &mut files);
    for path in files {
        let rel = path
            .strip_prefix(src_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if skip_file(&rel) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut rest = text.as_str();
        while let Some(at) = rest.find("\"VICTORY_") {
            let tail = &rest[at + 1..];
            let end = tail
                .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                .unwrap_or(tail.len());
            if tail[end..].starts_with('"') {
                let name = &tail[..end];
                if name.len() > "VICTORY_".len() && !non_cache.contains(&name) {
                    assert!(
                        affects_shader_cache(name),
                        "{name} (in {crate_label}/{rel}) is unclassified: add it to CACHE_KEYS if it affects compiled shaders/pipelines or to the explicit non-cache list if it is runtime-only"
                    );
                }
            }
            rest = &tail[end..];
        }
    }
}

#[test]
fn every_shader_translator_and_gpu_switch_is_classified() {
    for &key in NON_CACHE_SHADER_KEYS.iter().chain(NON_CACHE_GPU_KEYS) {
        assert!(
            !affects_shader_cache(key),
            "{key} cannot be in both CACHE_KEYS and the non-cache allowlist"
        );
    }

    let victory_crates = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("third_party")
        .join("victory")
        .join("crates");

    scan_victory_switches(
        &victory_crates.join("victory-shader").join("src"),
        |rel| {
            rel.ends_with("_tests.rs")
                || matches!(
                    rel,
                    "emit/host.rs" | "emit/validator.rs" | "emit/bitcode/census.rs"
                )
        },
        NON_CACHE_SHADER_KEYS,
        "victory-shader/src",
    );

    scan_victory_switches(
        &victory_crates.join("victory-gpu").join("src"),
        |rel| rel.ends_with("_tests.rs") || rel == "pipeline/list_bench.rs",
        NON_CACHE_GPU_KEYS,
        "victory-gpu/src",
    );
}

fn collect_rs(dir: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, into);
        } else if path.extension().is_some_and(|e| e == "rs") {
            into.push(path);
        }
    }
}

#[test]
fn a_missing_victory_exe_is_its_own_build_identity() {
    assert_eq!(
        build_id(Path::new("does-not-exist/victory.exe")),
        "not-built"
    );
}

#[test]
fn play_page_runtime_options_do_not_invalidate_the_warm_marker() {
    use crate::options::{Keyboard, LaunchOptions, Preset, Resolution, WindowMode};
    use crate::profile::Profiles;

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("game_scarlett_master.exe"), b"x").expect("write");
    let profile = Profiles::embedded()
        .expect("profiles")
        .by_id("windows-nvidia")
        .expect("profile")
        .clone();
    let base_env = crate::env::build_env(dir.path(), &profile, &[], None).expect("env");
    let base_fp = fingerprint("1:1", &base_env, &archives());

    for options in [
        LaunchOptions {
            resolution: Resolution::P1080,
            window: WindowMode::Borderless,
            keyboard: Keyboard::Azerty,
            overlay: false,
            auto_a: true,
            frametime_dump: true,
            ..LaunchOptions::default()
        },
        LaunchOptions {
            preset: Preset::Quality,
            ..LaunchOptions::default()
        },
        LaunchOptions {
            preset: Preset::MidVram,
            ..LaunchOptions::default()
        },
        LaunchOptions {
            preset: Preset::LowVram,
            ..LaunchOptions::default()
        },
    ] {
        let env = crate::env::build_env(dir.path(), &profile, &options.env_pairs(), None)
            .expect("env");
        assert_eq!(
            fingerprint("1:1", &env, &archives()),
            base_fp,
            "runtime-only Play options {options:?} must not force a shader cache re-warm"
        );
    }

    for (key, value) in [
        ("VICTORY_SHADER_OPT", "0"),
        ("VICTORY_RT_RESOLVE", "0"),
        ("VICTORY_CACHE_DIR", "D:\\other-cache"),
        ("VICTORY_SKIP_PIPELINES", "0123456789abcdef"),
        ("VICTORY_ROOT_SRV_DIRECT", "1"),
        ("VICTORY_BVH_QUARANTINE", "1"),
        ("VICTORY_DYNAMIC_SLOTS", "524288"),
        ("VICTORY_RAW_VIEWS", "0"),
        ("VICTORY_D3D12_WARP", "1"),
    ] {
        let overrides = vec![(key.to_string(), value.to_string())];
        let env = crate::env::build_env(dir.path(), &profile, &overrides, None).expect("env");
        assert_ne!(
            fingerprint("1:1", &env, &archives()),
            base_fp,
            "cache-affecting switch {key} must invalidate the warm marker"
        );
    }
}
