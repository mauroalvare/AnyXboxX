//! The title window's name ("AnyXboxX") and icon. The icon is the game's own
//! logo, read at run time from the user's game folder: its
//! `MicrosoftGame.Config` names the PNG files (`Square44x44Logo` for the
//! small icon, `Square150x150Logo` for the big one). Nothing from the game is
//! stored in the repository, and any failure leaves the default icon.

use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, HICON};

/// The game's description file, beside its executable.
const CONFIG_FILE: &str = "MicrosoftGame.Config";
/// Attribute naming the small logo.
const SMALL_LOGO: &str = "Square44x44Logo";
/// Attribute naming the big logo.
const BIG_LOGO: &str = "Square150x150Logo";
/// Size limits, so a wrong file cannot exhaust memory.
const MAX_CONFIG_BYTES: u64 = 4 << 20;
const MAX_LOGO_BYTES: u64 = 8 << 20;
/// `CreateIconFromResourceEx` version for icon resources (PNG accepted).
const ICON_VERSION: u32 = 0x0003_0000;

/// The value of attribute `name` (`name="value"` or `name='value'`) in a
/// config text. The name must start a token, so `Foo` never matches `XFoo`.
pub fn config_attribute<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let mut from = 0;
    while let Some(found) = text[from..].find(name) {
        let start = from + found;
        let end = start + name.len();
        from = end;
        let boundary = text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| c.is_whitespace() || c == '<' || c == '"' || c == '\'');
        if !boundary {
            continue;
        }
        let rest = text[end..].trim_start().strip_prefix('=')?.trim_start();
        let quote = rest.chars().next().filter(|q| *q == '"' || *q == '\'')?;
        let value = &rest[1..];
        return value.find(quote).map(|close| &value[..close]);
    }
    None
}

/// `dir` joined with the relative path `file`, or `None` if it is empty,
/// absolute, has a drive or root, or climbs out of `dir` with `..`.
pub fn logo_path(dir: &Path, file: &str) -> Option<PathBuf> {
    let file = file.trim();
    if file.is_empty() || file.contains('\0') {
        return None;
    }
    let mut joined = dir.to_path_buf();
    for component in Path::new(file).components() {
        match component {
            Component::Normal(part) => joined.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (joined != dir).then_some(joined)
}

/// Reads at most `max` bytes of a regular file, with the reason on failure.
fn read_limited(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() > max {
        return Err(format!("{}: not a regular file of at most {max} bytes", path.display()));
    }
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The bytes of the logo named by `attribute` in the config of `dir`.
fn logo_bytes(config: &str, dir: &Path, attribute: &str) -> Result<Vec<u8>, String> {
    let file = config_attribute(config, attribute).ok_or_else(|| format!("no {attribute} in {CONFIG_FILE}"))?;
    let path = logo_path(dir, file).ok_or_else(|| format!("{attribute} {file:?} leaves the game folder"))?;
    read_limited(&path, MAX_LOGO_BYTES)
}

/// An icon built from PNG bytes, drawn at `size` pixels.
fn icon_from_png(bytes: &[u8], size: i32) -> Result<HICON, String> {
    let len = u32::try_from(bytes.len()).map_err(|_| "logo too large".to_string())?;
    // SAFETY: a buffer of `len` bytes; the call copies what it needs.
    let icon = unsafe {
        wm::CreateIconFromResourceEx(bytes.as_ptr(), len, 1, ICON_VERSION, size, size, wm::LR_DEFAULTCOLOR)
    };
    if icon.is_null() {
        Err(format!("the system refused the PNG ({})", std::io::Error::last_os_error()))
    } else {
        Ok(icon)
    }
}

fn load_icon(config: &str, dir: &Path, attribute: &str, metric: wm::SYSTEM_METRICS_INDEX) -> Option<usize> {
    // SAFETY: a plain system metric query.
    let size = unsafe { wm::GetSystemMetrics(metric) };
    match logo_bytes(config, dir, attribute).and_then(|bytes| icon_from_png(&bytes, size)) {
        Ok(icon) => Some(icon as usize),
        Err(why) => {
            tracing::warn!("window icon ({attribute}): {why}; keeping the default icon");
            None
        }
    }
}

/// The small and big icons, loaded once; a missing one stays `None`.
fn icons() -> &'static (Option<usize>, Option<usize>) {
    static ICONS: OnceLock<(Option<usize>, Option<usize>)> = OnceLock::new();
    ICONS.get_or_init(|| {
        let dir = crate::runtime::main_image()
            .and_then(crate::runtime::image_path)
            .and_then(|exe| exe.parent().map(Path::to_path_buf));
        let Some(dir) = dir else {
            tracing::warn!("window icon: the game folder is unknown; keeping the default icon");
            return (None, None);
        };
        let config = match read_limited(&dir.join(CONFIG_FILE), MAX_CONFIG_BYTES) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(why) => {
                tracing::warn!("window icon: {why}; keeping the default icon");
                return (None, None);
            }
        };
        let small = load_icon(&config, &dir, SMALL_LOGO, wm::SM_CXSMICON);
        let big = load_icon(&config, &dir, BIG_LOGO, wm::SM_CXICON);
        if small.is_some() || big.is_some() {
            tracing::info!(dir = %dir.display(), "window icon read from the game folder");
        }
        (small, big)
    })
}

/// Whether a window made with these arguments is the title's own window: a
/// top-level one that has a caption or is a popup (the full-screen form).
pub fn is_title_window(parent_is_null: bool, style: u32) -> bool {
    parent_is_null && (style & wm::WS_CAPTION == wm::WS_CAPTION || style & wm::WS_POPUP != 0)
}

/// Names the window and gives it the game's icons, on the class and window.
pub fn apply(hwnd: HWND) {
    if hwnd.is_null() {
        return;
    }
    let mut name: Vec<u16> = victory_gpu::WINDOW_NAME.encode_utf16().collect();
    name.push(0);
    let (small, big) = *icons();
    // SAFETY: a window this thread just created and a NUL-terminated name.
    unsafe {
        wm::SetWindowTextW(hwnd, name.as_ptr());
        for (kind, class_index, icon) in [
            (wm::ICON_SMALL, wm::GCLP_HICONSM, small),
            (wm::ICON_BIG, wm::GCLP_HICON, big),
        ] {
            let Some(icon) = icon else { continue };
            wm::SetClassLongPtrW(hwnd, class_index, icon as isize);
            wm::SendMessageW(hwnd, wm::WM_SETICON, kind as usize, icon as isize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = "<Game><ShellVisuals DefaultDisplayName=\"X\" Square150x150Logo=\"img/big.png\"\n        Square44x44Logo = 'small.png' XSquare44x44Logo=\"no.png\"/></Game>";

    /// A valid 1x1 PNG, written by hand.
    const PNG_1X1: [u8; 70] = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D, 0x49, 0x48, 0x44, 0x52, 0, 0, 0, 1, 0, 0, 0,
        1, 8, 6, 0, 0, 0, 0x1F, 0x15, 0xC4, 0x89, 0, 0, 0, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8,
        0xCF, 0xC0, 0xF0, 0x1F, 0, 0x05, 0, 0x01, 0xFF, 0x89, 0x99, 0x3D, 0x1D, 0, 0, 0, 0, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    #[test]
    fn attributes_are_found_in_either_quotes_and_by_whole_name() {
        assert_eq!(config_attribute(CONFIG, "Square150x150Logo"), Some("img/big.png"));
        assert_eq!(config_attribute(CONFIG, SMALL_LOGO), Some("small.png"));
        assert_eq!(config_attribute(CONFIG, "Missing"), None);
        assert_eq!(config_attribute("Square44x44Logo=oops", SMALL_LOGO), None);
        assert_eq!(config_attribute("Square44x44Logo=\"open", SMALL_LOGO), None);
    }

    #[test]
    fn logo_paths_stay_inside_the_game_folder() {
        let dir = Path::new("game");
        assert_eq!(logo_path(dir, "a.png"), Some(dir.join("a.png")));
        assert_eq!(logo_path(dir, "./img/a.png"), Some(dir.join("img").join("a.png")));
        for bad in ["", " ", "../a.png", "img/../../a.png", "/a.png", "\\a.png", "C:\\a.png", "C:a.png", ".", "a\0b"] {
            assert_eq!(logo_path(dir, bad), None, "{bad:?}");
        }
    }

    #[test]
    fn only_captioned_or_popup_top_level_windows_are_the_title() {
        assert!(is_title_window(true, wm::WS_OVERLAPPEDWINDOW));
        assert!(is_title_window(true, wm::WS_POPUP | wm::WS_VISIBLE));
        assert!(!is_title_window(false, wm::WS_OVERLAPPEDWINDOW));
        assert!(!is_title_window(true, wm::WS_CHILD));
    }

    #[test]
    fn a_missing_logo_is_reported_not_faked() {
        let dir = tempfile::tempdir().unwrap();
        assert!(logo_bytes(CONFIG, dir.path(), BIG_LOGO).is_err());
        assert!(logo_bytes("<Game/>", dir.path(), BIG_LOGO).unwrap_err().contains("no Square150x150Logo"));
        assert!(logo_bytes("Square44x44Logo=\"../x.png\"", dir.path(), SMALL_LOGO).unwrap_err().contains("leaves"));
    }

    #[test]
    fn a_png_beside_the_config_becomes_an_icon_and_garbage_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("small.png"), PNG_1X1).unwrap();
        let bytes = logo_bytes(CONFIG, dir.path(), SMALL_LOGO).unwrap();
        assert_eq!(bytes, PNG_1X1);
        let icon = icon_from_png(&bytes, 16).expect("a valid PNG makes an icon");
        // SAFETY: the icon just created.
        unsafe { wm::DestroyIcon(icon) };
        assert!(icon_from_png(b"not a png at all", 16).is_err());
    }
}
