use super::*;

#[test]
fn rejects_relative_paths_and_the_current_directory() {
    for path in ["", ".", "plugins", r"C:plugins", r"\plugins", r"\\server\plugins"] {
        assert!(runtime_dir(Path::new(path)).unwrap_err().contains("absolute local"));
    }
    assert!(runtime_dir(&std::env::current_dir().unwrap()).unwrap_err().contains("current directory"));
}

#[test]
fn reports_a_missing_interposer() {
    let dir = tempfile::tempdir().unwrap();
    assert!(verify_runtime(dir.path(), true, true).unwrap_err().contains(INTERPOSER));
}

#[test]
fn refuses_unsigned_plugins_before_loading_anything() {
    let dir = tempfile::tempdir().unwrap();
    let dll = dir.path().join("sl.dlss_g.DLL");
    fs::write(&dll, b"generated unsigned input").unwrap();
    let error = verify_runtime(dir.path(), true, true).unwrap_err();
    assert!(error.contains("sl.dlss_g.DLL") && error.contains("not validly signed"), "{error}");
}

#[test]
fn rejects_a_trusted_non_nvidia_publisher() {
    let dll = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\wintrust.dll");
    let file = File::open(&dll).unwrap();
    let why = verify_signature(&dll, &file).unwrap_err();
    assert!(why.contains("expected Authenticode signer NVIDIA Corporation"), "{why}");
}

#[test]
fn plugin_directory_accepts_the_sdk_slash_separator() {
    let tmp = tempfile::tempdir().unwrap();
    let path = runtime_dir(tmp.path()).unwrap();
    assert!(matches!(path.components().next(), Some(Component::Prefix(p)) if matches!(p.kind(), Prefix::Disk(_))));
    let mixed = format!("{}/sl.common.dll", path.display());
    fs::write(tmp.path().join("sl.common.dll"), b"fixture").unwrap();
    assert_eq!(fs::read(mixed).unwrap(), b"fixture");
}
