use super::{INTERPOSER, Path, wide};
use std::fs::{self, File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Component, PathBuf, Prefix};
use windows_sys::Win32::Security::{Cryptography::*, WinTrust::*};

/// The Authenticode signature of `path` (NVIDIA signs its DLLs), checked
/// without going online.
pub(super) fn verify_signature(path: &Path, locked: &std::fs::File) -> Result<(), String> {
    let file = wide(path);
    let mut info = WINTRUST_FILE_INFO {
        cbStruct: size_of::<WINTRUST_FILE_INFO>() as u32,
        pcwszFilePath: file.as_ptr(),
        hFile: std::os::windows::io::AsRawHandle::as_raw_handle(locked),
        pgKnownSubject: std::ptr::null_mut(),
    };
    let mut data = WINTRUST_DATA {
        cbStruct: size_of::<WINTRUST_DATA>() as u32,
        dwUIChoice: WTD_UI_NONE,
        fdwRevocationChecks: WTD_REVOKE_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 { pFile: &mut info },
        dwStateAction: WTD_STATEACTION_VERIFY,
        dwProvFlags: WTD_CACHE_ONLY_URL_RETRIEVAL | WTD_REVOCATION_CHECK_NONE,
        ..Default::default()
    };
    let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    // SAFETY: filled structures that live through both calls.
    let code = unsafe { WinVerifyTrust(std::ptr::null_mut(), &mut action, (&raw mut data).cast()) };
    let result = if code == 0 {
        signer_is_nvidia(&data).map_err(|why| format!("{}: {why}", path.display()))
    } else {
        Err(format!("{} is not validly signed ({:#x})", path.display(), code as u32))
    };
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    // SAFETY: as above; closes the state the first call opened.
    unsafe { WinVerifyTrust(std::ptr::null_mut(), &mut action, (&raw mut data).cast()) };
    result
}

fn signer_is_nvidia(data: &WINTRUST_DATA) -> Result<(), String> {
    // SAFETY: WinVerifyTrust accepted this state. Read its primary signer's
    // leaf certificate before WTD_STATEACTION_CLOSE releases the chain.
    let organization = unsafe {
        let provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        let signer = provider.as_mut().and_then(|p| WTHelperGetProvSignerFromChain(p, 0, 0, 0).as_mut());
        let cert = signer
            .and_then(|s| WTHelperGetProvCertFromChain(s, 0).as_ref())
            .ok_or("Authenticode signer certificate unavailable")?;
        let mut name = [0u16; 256];
        let count = CertGetNameStringW(
            cert.pCert,
            CERT_NAME_ATTR_TYPE,
            0,
            szOID_ORGANIZATION_NAME.cast(),
            name.as_mut_ptr(),
            name.len() as u32,
        );
        if count < 2 || count as usize >= name.len() {
            return Err("Authenticode signer organization unavailable".into());
        }
        String::from_utf16_lossy(&name[..count as usize - 1])
    };
    if organization == "NVIDIA Corporation" {
        Ok(())
    } else {
        Err(format!("expected Authenticode signer NVIDIA Corporation, got {organization:?}"))
    }
}

/// Reject ambiguous locations before inspecting the filesystem.
pub(super) fn runtime_dir(path: &Path) -> Result<PathBuf, String> {
    fn local_drive(path: &Path) -> bool {
        path.is_absolute()
            && matches!(path.components().next(), Some(Component::Prefix(p))
            if matches!(p.kind(), Prefix::Disk(_) | Prefix::VerbatimDisk(_)))
    }
    if !local_drive(path) {
        return Err("VICTORY_STREAMLINE_DIR must be an absolute local directory".into());
    }
    let dir = fs::canonicalize(path).map_err(|e| format!("Streamline directory {}: {e}", path.display()))?;
    if !local_drive(&dir) || !dir.is_dir() {
        return Err(format!("{} is not a local directory", dir.display()));
    }
    let cwd = fs::canonicalize(".").map_err(|e| format!("current directory: {e}"))?;
    if dir == cwd {
        return Err("refusing to load Streamline from the current directory".into());
    }
    // 2.14 appends "/sl.*.dll". A slash is literal in a verbatim path,
    // so its own signature check would reject every plugin. Keep the
    // canonical destination, expressed as a normal absolute drive path.
    if let Some(Component::Prefix(prefix)) = dir.components().next()
        && let Prefix::VerbatimDisk(letter) = prefix.kind()
    {
        let mut dos = PathBuf::from(format!("{}:\\", char::from(letter)));
        dos.extend(dir.components().skip(2));
        if fs::canonicalize(&dos).ok().as_ref() != Some(&dir) {
            return Err("Streamline directory cannot be represented as an absolute DOS path".into());
        }
        return Ok(dos);
    }
    Ok(dir)
}

/// Streamline enumerates plugins at slInit and loads NGX lazily. Verify every
/// DLL in its search directory before loading even the interposer; retain the
/// read-only shared handles so verified files cannot be replaced meanwhile.
pub(super) fn verify_runtime(dir: &Path, dlss: bool, framegen: bool) -> Result<Vec<File>, String> {
    let mut verified = Vec::new();
    let mut names = Vec::new();
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("dll")) {
            continue;
        }
        if !entry.file_type().map_err(|e| e.to_string())?.is_file() {
            return Err(format!("expected a regular DLL at {}", path.display()));
        }
        // FILE_SHARE_READ excludes both write and delete access.
        let file = OpenOptions::new()
            .read(true)
            .share_mode(1)
            .open(&path)
            .map_err(|e| format!("opening {}: {e}", path.display()))?;
        verify_signature(&path, &file)?;
        names.push(entry.file_name());
        verified.push(file);
    }
    let mut required = vec![INTERPOSER, "sl.common.dll"];
    if dlss {
        required.extend(["sl.dlss.dll", "nvngx_dlss.dll"]);
    }
    if framegen {
        required.extend(["sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll", "nvngx_dlssg.dll"]);
    }
    for name in required {
        if !names.iter().any(|found| found.eq_ignore_ascii_case(name)) {
            return Err(format!("missing {name} in {}", dir.display()));
        }
    }
    Ok(verified)
}

#[cfg(test)]
mod tests;
