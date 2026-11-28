#![cfg(windows)]

use std::fs;
use std::process::Command;

#[test]
fn warm_used_with_only_a_shader_cache_reports_no_recorded_pipelines() {
    let cache = tempfile::tempdir().unwrap();
    // The data directory may not be inside the cache directory.
    let data = tempfile::tempdir().unwrap();
    fs::create_dir(cache.path().join("shaders")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .args(["warm", "--used"])
        .current_dir(cache.path())
        .env_remove("VICTORY_CONFIG")
        .env("VICTORY_CACHE_DIR", cache.path())
        .env("VICTORY_DATA_DIR", data.path())
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0 pipelines known"), "{stdout}");
    // The pipelines folder is made with the shaders one (`warm::pipelines_folder`),
    // empty: nothing is warmed.
    assert!(stdout.contains("0 to make"), "{stdout}");
    assert!(cache.path().join("pipelines").is_dir());
}
