//! Availability of DLSS Frame Generation on the launcher's main DXGI adapter.

use crate::host::GpuVendor;
use std::mem::size_of_val;
use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS};
use windows::Win32::System::Registry::{
    HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RRF_RT_REG_SZ, RegGetValueW,
};
use windows::core::w;

pub fn unavailable(gpu: GpuVendor, name: &str, hags: bool) -> Option<&'static str> {
    let mut words = name.split_ascii_whitespace();
    let model = words
        .find(|w| w.eq_ignore_ascii_case("RTX"))
        .and_then(|_| words.next());
    let supported = model
        .and_then(|s| {
            s.trim_end_matches(|c: char| c.is_ascii_alphabetic())
                .parse::<u16>()
                .ok()
        })
        .is_some_and(|n| matches!(n, 4001..=4099 | 5001..=5099));
    if gpu != GpuVendor::Nvidia || !supported {
        Some("Frame Generation requires a GeForce RTX 40 or 50 series GPU.")
    } else if !hags {
        Some(
            "Enable hardware-accelerated GPU scheduling in Windows Graphics settings and restart the PC.",
        )
    } else {
        None
    }
}

fn hags_enabled(mode: Option<u32>, windows_build: u32) -> bool {
    match mode {
        Some(2) => true,
        None => windows_build >= 22000, // First Windows 11 release.
        _ => false,
    }
}

pub fn detect_hags() -> bool {
    let mut mode = 0u32;
    let mut size = size_of_val(&mode) as u32;
    // SAFETY: each buffer is writable for its stated byte count.
    let result = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!("SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers"),
            w!("HwSchMode"),
            RRF_RT_REG_DWORD,
            None,
            Some((&raw mut mode).cast()),
            Some(&mut size),
        )
    };
    match result {
        ERROR_SUCCESS => hags_enabled(Some(mode), 0),
        ERROR_FILE_NOT_FOUND => {
            let mut build = [0u16; 32];
            let mut size = size_of_val(&build) as u32;
            let result = unsafe {
                RegGetValueW(
                    HKEY_LOCAL_MACHINE,
                    w!("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion"),
                    w!("CurrentBuildNumber"),
                    RRF_RT_REG_SZ,
                    None,
                    Some(build.as_mut_ptr().cast()),
                    Some(&mut size),
                )
            };
            let end = build.iter().position(|&n| n == 0).unwrap_or(build.len());
            result == ERROR_SUCCESS
                && hags_enabled(
                    None,
                    String::from_utf16_lossy(&build[..end]).parse().unwrap_or(0),
                )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkbox_requires_both_a_supported_adapter_and_hags() {
        for name in ["NVIDIA GeForce RTX 4070 SUPER", "RTX 5070Ti Laptop GPU"] {
            assert_eq!(unavailable(GpuVendor::Nvidia, name, true), None);
            assert!(unavailable(GpuVendor::Nvidia, name, false).is_some());
            assert!(unavailable(GpuVendor::Amd, name, true).is_some());
        }
        for name in ["", "RTX 3080 Ti", "RTX A5000", "RTX 4000 Ada Generation"] {
            assert!(unavailable(GpuVendor::Nvidia, name, true).is_some());
        }
    }

    #[test]
    fn explicit_hags_wins_over_the_windows_default() {
        assert!(hags_enabled(None, 22000));
        assert!(!hags_enabled(None, 19045));
        assert!(hags_enabled(Some(2), 19045));
        assert!(!hags_enabled(Some(1), 26100));
    }
}
