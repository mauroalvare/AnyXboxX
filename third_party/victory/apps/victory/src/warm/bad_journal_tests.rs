use std::fs;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use super::journal::{Journal, fingerprint};
use super::tests::fake_child;
use super::{Options, known, run_with_worker};
use victory_hle::known_pipelines::WarmCacheMetadata;

const KEYS: [&str; 3] = [
    "01010101010101010101010101010101",
    "02020202020202020202020202020202",
    "03030303030303030303030303030303",
];

const CACHE: WarmCacheMetadata = WarmCacheMetadata {
    translation: [1; 32],
    software: false,
    adapter: [24, 18, 7, 1],
    driver: 11,
};

#[test]
fn a_new_backend_translator_adapter_or_driver_reconsiders_both_outcomes() {
    let dir = tempfile::tempdir().unwrap();
    let entries = KEYS.map(|key| known::WarmEntry {
        key: key.to_owned(),
        offset: 0,
        skipped: false,
    });
    let options = Options {
        jobs: Some(1),
        ..Options::default()
    };
    let original = fingerprint(CACHE);

    // Pipeline 0 is made, pipeline 1 loses the device, and pipeline 2 is
    // refused cleanly by the second worker. Only the active crash is quarantined.
    run_with_worker(
        dir.path(),
        options,
        &entries,
        Some(&original),
        Instant::now(),
        |batch| {
            let script = match batch {
                [0, 1, 2] => "echo at 0&echo made 0 4 1&echo at 1&exit /b -2005270523",
                [2] => "echo at 2&echo made 2 2 0",
                _ => panic!("unexpected initial batch {batch:?}"),
            };
            fake_child(script, Stdio::piped())
        },
    )
    .unwrap();

    let done = dir.path().join("warm-done.txt");
    let bad = dir.path().join("warm-bad.txt");
    let completed = Journal::open(&done, Some(&original)).unwrap();
    let failed = Journal::open(&bad, Some(&original)).unwrap();
    assert_eq!(completed.keys, [KEYS[0].to_owned()].into());
    assert_eq!(failed.keys, [KEYS[1].to_owned()].into());
    run_with_worker(
        dir.path(),
        options,
        &entries,
        Some(&original),
        Instant::now(),
        |batch| {
            assert_eq!(batch, [2], "a refused pipeline stays eligible");
            fake_child("echo at 2&echo made 2 2 0", Stdio::piped())
        },
    )
    .unwrap();

    let variants = [
        (
            "backend",
            WarmCacheMetadata {
                software: true,
                ..CACHE
            },
        ),
        (
            "translator",
            WarmCacheMetadata {
                translation: [2; 32],
                ..CACHE
            },
        ),
        (
            "adapter",
            WarmCacheMetadata {
                adapter: [24, 19, 7, 1],
                ..CACHE
            },
        ),
        (
            "driver",
            WarmCacheMetadata {
                driver: 12,
                ..CACHE
            },
        ),
    ];
    for (dimension, context) in variants {
        let changed = fingerprint(context);
        assert_ne!(original, changed, "{dimension}");
        let attempts = AtomicUsize::new(0);
        run_with_worker(
            dir.path(),
            options,
            &entries,
            Some(&changed),
            Instant::now(),
            |batch| {
                assert_eq!(batch, [0, 1, 2], "{dimension}");
                attempts.fetch_add(batch.len(), Ordering::Relaxed);
                fake_child(
                    "echo at 0&echo made 0 4 1&echo at 1&echo made 1 3 1&echo at 2&echo made 2 2 1",
                    Stdio::piped(),
                )
            },
        )
        .unwrap();
        assert_eq!(attempts.into_inner(), 3, "{dimension}");
        assert_eq!(Journal::open(&done, Some(&changed)).unwrap().keys.len(), 3);
        assert!(Journal::open(&bad, Some(&changed)).unwrap().keys.is_empty());
    }

    // The explicit retry requested by --again includes quarantined pipelines.
    let again = Options {
        again: true,
        ..options
    };
    run_with_worker(
        dir.path(),
        again,
        &entries,
        Some(&original),
        Instant::now(),
        |batch| {
            assert_eq!(batch, [0, 1, 2]);
            fake_child(
                "echo at 0&echo made 0 4 1&echo at 1&echo made 1 3 1&echo at 2&echo made 2 2 0",
                Stdio::piped(),
            )
        },
    )
    .unwrap();
}

#[test]
fn appended_records_survive_legacy_records_and_a_torn_tail() {
    let dir = tempfile::tempdir().unwrap();
    let fingerprint = fingerprint(CACHE);
    let intact = "04".repeat(16);
    for (file, legacy) in [
        ("warm-done.txt", format!("0123456789abcdef {}\n", KEYS[0])),
        ("warm-bad.txt", format!("{} device removed\n", KEYS[0])),
    ] {
        let path = dir.path().join(file);
        // Include a newer schema, invalid UTF-8, and a tail with all fields but no terminator.
        let mut before = legacy.into_bytes();
        before.extend(format!("v9 {fingerprint} {} made\tend\n", KEYS[0]).bytes());
        before.extend(format!("v1 {fingerprint} {intact} made\tend\r\n").bytes());
        before.extend(b"\xff\n");
        before.extend(format!("v1 {fingerprint} {} device remo", KEYS[1]).bytes());
        fs::write(&path, &before).unwrap();
        let journal = Journal::open(&path, Some(&fingerprint)).unwrap();
        assert_eq!(journal.keys, [intact.clone()].into());
        journal.append(KEYS[2], "new complete record").unwrap();
        assert!(fs::read(&path).unwrap().starts_with(&before));
        assert_eq!(
            Journal::open(&path, Some(&fingerprint)).unwrap().keys,
            [intact.clone(), KEYS[2].to_owned()].into()
        );
    }
}

#[test]
fn an_unknown_context_cannot_suppress_or_record_an_attempt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("warm-bad.txt");
    let known = fingerprint(CACHE);
    Journal::open(&path, Some(&known))
        .unwrap()
        .append(KEYS[0], "device removed")
        .unwrap();
    let before = fs::read(&path).unwrap();
    let journal = Journal::open(&path, None).unwrap();
    assert!(journal.keys.is_empty());
    journal.append(KEYS[1], "device removed").unwrap();
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn unavailable_warm_device_reports_a_warning() {
    // A separate process selects the adapter before this test binary caches its host.
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "warm::tests::clean_first_run_creates_both_shaders_and_pipelines_cache_dirs",
            "--nocapture",
        ])
        .env(
            "VICTORY_D3D12_ADAPTER",
            "__synthetic_warm_adapter_unavailable__",
        )
        .env_remove("VICTORY_D3D12_WARP")
        .output()
        .unwrap();
    let warning = String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success(), "{warning}");
    assert!(warning.contains("Warning from victory warm"), "{warning}");
    assert!(
        warning.contains("D3D12 warm device unavailable"),
        "{warning}"
    );
    assert!(
        warning.contains("Eligible pipelines will be attempted again"),
        "{warning}"
    );
}
