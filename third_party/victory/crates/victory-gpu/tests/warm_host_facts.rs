#![cfg(windows)]

use victory_gpu::shader_cache;
use victory_shader::emit;

#[test]
fn precompile_keeps_explicit_translation_host_settings() {
    let dir = tempfile::tempdir().unwrap();
    shader_cache::use_shader_cache(dir.path().join("shaders"));

    // The CLI sets the view limit and overrides the wave width per warm worker.
    let limit = 524_287;
    emit::set_view_heap_limit(limit);
    std::thread::spawn(move || {
        emit::set_thread_max_wave(Some(64));
        // An invalid container still exercises host initialization and cache-key construction.
        assert!(shader_cache::precompile(b"synthetic").is_err());
        assert_eq!(emit::host_settings()[..2], [64, limit]);
    })
    .join()
    .unwrap();

    // An unspecified limit is still detected for the local adapter.
    emit::set_view_heap_limit(0);
    assert!(shader_cache::precompile(b"synthetic").is_err());
    assert_ne!(emit::view_heap_limit(), 0);
}
