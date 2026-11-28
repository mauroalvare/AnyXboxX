//! VIctory's shader cache: where it is, and removing it. VIctory keeps the cache in
//! `%LOCALAPPDATA%\victory\cache` (or `VICTORY_CACHE_DIR`); its saves and data live in
//! `victory\data`, which this module never touches.

use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};

/// The folder name the cache must have for the launcher to delete it.
pub const CACHE_NAME: &str = "cache";
const DATA_NAME: &str = "data";

fn victory_subdir(
    name: &str,
    override_dir: Option<&OsStr>,
    local_app_data: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    let non_empty = |value: Option<&OsStr>| value.filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(dir) = non_empty(override_dir) {
        return Some(dir);
    }
    if let Some(base) = non_empty(local_app_data).or_else(|| non_empty(xdg_cache_home)) {
        return Some(base.join("victory").join(name));
    }
    non_empty(home).map(|h| h.join(".cache").join("victory").join(name))
}

/// Where VIctory's cache is, from the environment values it reads (`VICTORY_CACHE_DIR`, then
/// `LOCALAPPDATA`, then `XDG_CACHE_HOME`, then `HOME/.cache`, as victory-core does).
pub fn cache_dir(
    override_dir: Option<&OsStr>,
    local_app_data: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    victory_subdir(
        CACHE_NAME,
        override_dir,
        local_app_data,
        xdg_cache_home,
        home,
    )
}

/// Where VIctory's data root is (`VICTORY_DATA_DIR`, then `LOCALAPPDATA`, `XDG_CACHE_HOME`, `HOME/.cache`).
pub fn data_dir(
    override_dir: Option<&OsStr>,
    local_app_data: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    victory_subdir(
        DATA_NAME,
        override_dir,
        local_app_data,
        xdg_cache_home,
        home,
    )
}

/// The cache of the current environment.
pub fn current_cache_dir() -> Option<PathBuf> {
    cache_dir(
        std::env::var_os("VICTORY_CACHE_DIR").as_deref(),
        std::env::var_os("LOCALAPPDATA").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// The data root of the current environment.
pub fn current_data_dir() -> Option<PathBuf> {
    data_dir(
        std::env::var_os("VICTORY_DATA_DIR").as_deref(),
        std::env::var_os("LOCALAPPDATA").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// Extracts `<TitleId>` (hex `u32`) from `MicrosoftGame.config` text, skipping XML comments; `0` if absent.
pub fn title_id_from_manifest(text: &str) -> u32 {
    let mut clean = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<!--") {
        clean.push_str(&rest[..start]);
        match rest[start + 4..].find("-->") {
            Some(end) => rest = &rest[start + 4 + end + 3..],
            None => {
                rest = "";
                break;
            }
        }
    }
    clean.push_str(rest);
    let mut search = clean.as_str();
    while let Some(open) = search.find("<TitleId") {
        let after_name = &search[open + "<TitleId".len()..];
        let valid_tag = after_name
            .chars()
            .next()
            .is_some_and(|c| c == '>' || c.is_ascii_whitespace());
        if valid_tag
            && let Some(gt) = after_name.find('>')
            && let Some(close) = after_name[gt + 1..].find("</TitleId>")
        {
            let value = after_name[gt + 1..gt + 1 + close].trim();
            return u32::from_str_radix(value, 16).unwrap_or(0);
        }
        search = after_name;
    }
    0
}

/// Reads the title's hex `TitleId` from `<game_dir>/MicrosoftGame.config` (`0` if missing or unset).
pub fn read_title_id(game_dir: &Path) -> u32 {
    std::fs::read_to_string(game_dir.join("MicrosoftGame.config"))
        .map(|text| title_id_from_manifest(&text))
        .unwrap_or(0)
}

/// VIctory's per-title saves subfolder (`<data_root>\<TitleId_8HEX>\saves`).
pub fn title_saves_dir(data_root: &Path, title_id: u32) -> PathBuf {
    data_root.join(format!("{title_id:08X}")).join("saves")
}

/// What `clean_cache` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanOutcome {
    Removed,
    NothingToClean,
}

#[derive(Debug, thiserror::Error)]
pub enum CleanError {
    #[error("refusing to delete '{0}': the launcher only deletes a folder named '{CACHE_NAME}'")]
    NotACache(String),
    #[error("cannot delete '{0}': {1}")]
    Io(String, io::Error),
}

/// Deletes the cache folder `dir` and everything in it. Only a folder whose name is `cache` is
/// deleted; anything else is refused, whatever the environment says.
pub fn clean_cache(dir: &Path) -> Result<CleanOutcome, CleanError> {
    let named_cache = dir.file_name().is_some_and(|n| n == CACHE_NAME);
    if !named_cache {
        return Err(CleanError::NotACache(dir.display().to_string()));
    }
    match std::fs::symlink_metadata(dir) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(CleanOutcome::NothingToClean),
        Err(error) => Err(CleanError::Io(dir.display().to_string(), error)),
        Ok(_) => std::fs::remove_dir_all(dir)
            .map(|()| CleanOutcome::Removed)
            .map_err(|e| CleanError::Io(dir.display().to_string(), e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_wins_over_local_app_data() {
        let dir = cache_dir(
            Some(OsStr::new("D:\\vc")),
            Some(OsStr::new("C:\\Users\\u\\AppData\\Local")),
            None,
            None,
        );
        assert_eq!(dir, Some(PathBuf::from("D:\\vc")));
    }

    #[test]
    fn default_is_local_app_data_victory_cache() {
        let dir = cache_dir(None, Some(OsStr::new("C:\\L")), None, None);
        assert_eq!(
            dir,
            Some(PathBuf::from("C:\\L").join("victory").join("cache"))
        );
    }

    #[test]
    fn falls_back_to_xdg_then_home_and_empty_values_do_not_count() {
        let xdg = cache_dir(
            Some(OsStr::new("")),
            Some(OsStr::new("")),
            Some(OsStr::new("/x")),
            None,
        );
        assert_eq!(xdg, Some(PathBuf::from("/x").join("victory").join("cache")));
        let home = cache_dir(None, None, None, Some(OsStr::new("/h")));
        assert_eq!(
            home,
            Some(
                PathBuf::from("/h")
                    .join(".cache")
                    .join("victory")
                    .join("cache")
            )
        );
        assert_eq!(cache_dir(None, None, None, None), None);
    }

    #[test]
    fn clean_removes_a_folder_named_cache_with_its_content() {
        let root = tempfile::tempdir().expect("tempdir");
        let cache = root.path().join("cache");
        std::fs::create_dir_all(cache.join("shaders")).expect("mkdir");
        std::fs::write(cache.join("shaders").join("x.bin"), b"x").expect("write");
        assert_eq!(clean_cache(&cache).expect("clean"), CleanOutcome::Removed);
        assert!(!cache.exists());
        assert_eq!(
            clean_cache(&cache).expect("clean"),
            CleanOutcome::NothingToClean
        );
    }

    #[test]
    fn clean_refuses_any_folder_not_named_cache() {
        let root = tempfile::tempdir().expect("tempdir");
        let data = root.path().join("data");
        std::fs::create_dir_all(&data).expect("mkdir");
        assert!(matches!(clean_cache(&data), Err(CleanError::NotACache(_))));
        assert!(data.exists(), "a refused folder is left alone");
    }

    #[test]
    fn title_saves_dir_reads_manifest_title_id_and_skips_xml_comments() {
        assert_eq!(
            title_id_from_manifest("<Game><TitleId><!-- comment -->7952987a</TitleId></Game>"),
            0x7952_987A
        );
        assert_eq!(title_id_from_manifest("<Game></Game>"), 0);
        let game = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            game.path().join("MicrosoftGame.config"),
            b"<Game><TitleId>7952987A</TitleId></Game>",
        )
        .expect("write");
        let data = data_dir(None, Some(OsStr::new("C:\\L")), None, None).expect("data");
        assert_eq!(
            title_saves_dir(&data, read_title_id(game.path())),
            PathBuf::from("C:\\L")
                .join("victory")
                .join("data")
                .join("7952987A")
                .join("saves")
        );
    }
}
