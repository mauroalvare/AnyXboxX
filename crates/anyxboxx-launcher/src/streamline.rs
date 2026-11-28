//! Check the Authenticode signer of the selected interposer without loading the DLL.

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

use windows::Win32::Foundation::{HANDLE, HWND};
use windows::Win32::Security::Cryptography::{
    CERT_NAME_ATTR_TYPE, CertGetNameStringW, szOID_ORGANIZATION_NAME,
};
use windows::Win32::Security::WinTrust::*;
use windows::core::PCWSTR;

pub const RELEASES: &str = "https://github.com/NVIDIA-RTX/Streamline/releases";
const DLL: &str = "sl.interposer.dll";

pub fn check(folder: &Path) -> Result<(), String> {
    if !folder.is_absolute() {
        return Err(format!(
            "Choose the full path to the Streamline bin\\x64 folder containing {DLL}."
        ));
    }
    let dll = folder.join(DLL);
    if !dll.is_file() {
        return Err(format!("{DLL} was not found in {}.", folder.display()));
    }
    let name: Vec<u16> = dll.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut file = WINTRUST_FILE_INFO {
        cbStruct: size_of::<WINTRUST_FILE_INFO>() as u32,
        pcwszFilePath: PCWSTR(name.as_ptr()),
        ..Default::default()
    };
    let mut data = WINTRUST_DATA {
        cbStruct: size_of::<WINTRUST_DATA>() as u32,
        dwUIChoice: WTD_UI_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 { pFile: &mut file },
        dwStateAction: WTD_STATEACTION_VERIFY,
        dwProvFlags: WTD_CACHE_ONLY_URL_RETRIEVAL,
        ..Default::default()
    };
    let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    // SAFETY: file and its NUL-terminated path live through both calls. Read the
    // leaf of the verified signer chain while the trust provider still owns it.
    unsafe {
        let status = WinVerifyTrust(HWND::default(), &mut action, (&raw mut data).cast());
        let result = if status != 0 {
            Err(format!(
                "{DLL} has no trusted Authenticode signature (0x{status:08X})."
            ))
        } else if signer_organization(data.hWVTStateData).as_deref() != Some("NVIDIA Corporation") {
            Err(format!("{DLL} must be signed by NVIDIA Corporation."))
        } else {
            Ok(())
        };
        data.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrust(HWND::default(), &mut action, (&raw mut data).cast());
        result
    }
}

unsafe fn signer_organization(state: HANDLE) -> Option<String> {
    // SAFETY: state belongs to a successful, still-open verification. Each
    // helper result is checked before passing it to the next helper.
    unsafe {
        let provider = WTHelperProvDataFromStateData(state).as_mut()?;
        let signer = WTHelperGetProvSignerFromChain(provider, 0, false, 0).as_mut()?;
        let cert = WTHelperGetProvCertFromChain(signer, 0)
            .as_ref()?
            .pCert
            .as_ref()?;
        let oid = Some(szOID_ORGANIZATION_NAME.as_ptr().cast());
        let len = CertGetNameStringW(cert, CERT_NAME_ATTR_TYPE, 0, oid, None);
        let mut name = vec![0; len as usize];
        if len <= 1 || CertGetNameStringW(cert, CERT_NAME_ATTR_TYPE, 0, oid, Some(&mut name)) != len
        {
            return None;
        }
        String::from_utf16(&name[..name.len() - 1]).ok()
    }
}

pub fn check_launch(env: &anyxboxx::run::Env) -> Result<(), String> {
    let get = |name| {
        env.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_os_str())
    };
    if [crate::dlss::SR, crate::dlss::FG]
        .into_iter()
        .any(|key| get(key) == Some(OsStr::new("1")))
    {
        check(Path::new(get(crate::dlss::DIR).unwrap_or_default()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_and_unsigned_interposers_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(check(dir.path()).unwrap_err().contains("was not found"));
        std::fs::write(dir.path().join(DLL), b"unsigned test input").unwrap();
        assert!(
            check(dir.path())
                .unwrap_err()
                .contains("no trusted Authenticode signature")
        );
    }

    #[test]
    #[ignore = "requires a local NVIDIA Streamline SDK in VICTORY_STREAMLINE_DIR"]
    fn accepts_the_nvidia_release_interposer() {
        let dir = std::env::var_os(crate::dlss::DIR).expect("set VICTORY_STREAMLINE_DIR");
        check(Path::new(&dir)).unwrap();
    }
}
