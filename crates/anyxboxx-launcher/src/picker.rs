//! Choosing the game folder: the native dialogs (on their own thread, so the window keeps
//! repainting), the check that a folder is the game's, and the search at start-up.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

/// The game's executable; a folder holding it is the game folder.
pub const GAME_EXE: &str = "game_scarlett_master.exe";

/// What a dialog was opened for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    GameFolder,
    RepoFolder,
    Streamline,
    XgLibrary,
}

/// Checks that `dir` is the game folder.
pub fn validate_game_folder(dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    if !dir.join(GAME_EXE).is_file() {
        return Err(format!("{GAME_EXE} not found in {}", dir.display()));
    }
    Ok(())
}

/// The folder a drop means: a folder itself, or the folder of a dropped file.
pub fn folder_of_drop(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().map(Path::to_path_buf).unwrap_or_default()
    }
}

/// Looks for the game folder: the last used one first, then the folder of the launcher and each
/// of its parents.
pub fn detect(launcher_dir: Option<&Path>, last_used: Option<&Path>) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.extend(last_used.map(Path::to_path_buf));
    if let Some(dir) = launcher_dir {
        candidates.extend(dir.ancestors().map(Path::to_path_buf));
    }
    candidates
        .into_iter()
        .find(|dir| validate_game_folder(dir).is_ok())
}

/// The folder of the running launcher.
pub fn launcher_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
}

/// Opens the dialog on a thread; the answer (`None` when cancelled) arrives on the receiver.
pub fn open(kind: Pick, start_in: Option<PathBuf>) -> Receiver<(Pick, Option<PathBuf>)> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new();
        if let Some(dir) = start_in.filter(|d| d.is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        let chosen = match kind {
            Pick::Streamline => dialog
                .set_title("Choose the Streamline bin\\x64 folder")
                .pick_folder(),
            Pick::GameFolder => dialog
                .set_title("Choose the game folder (it holds game_scarlett_master.exe)")
                .pick_folder(),
            Pick::RepoFolder => dialog
                .set_title("Choose your AnyXboxX folder (it holds third_party\\victory)")
                .pick_folder(),
            Pick::XgLibrary => dialog
                .set_title("Choose your xg_xs.dll")
                .add_filter("xg_xs.dll", &["dll"])
                .pick_file(),
        };
        let _ = tx.send((kind, chosen));
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game_folder() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join(GAME_EXE), b"x").expect("write");
        dir
    }

    #[test]
    fn a_folder_with_the_game_exe_is_valid() {
        let dir = game_folder();
        assert!(validate_game_folder(dir.path()).is_ok());
    }

    #[test]
    fn other_folders_are_refused_with_a_reason() {
        let empty = tempfile::tempdir().expect("temp dir");
        let error = validate_game_folder(empty.path()).expect_err("no exe");
        assert!(error.contains(GAME_EXE));
        let missing = empty.path().join("nope");
        assert!(
            validate_game_folder(&missing)
                .expect_err("no folder")
                .contains("not a folder")
        );
    }

    #[test]
    fn a_dropped_file_means_its_folder() {
        let dir = game_folder();
        assert_eq!(folder_of_drop(dir.path()), dir.path());
        assert_eq!(folder_of_drop(&dir.path().join(GAME_EXE)), dir.path());
    }

    #[test]
    fn detection_walks_up_from_the_launcher() {
        let game = game_folder();
        let deep = game.path().join("repo").join("target").join("release");
        std::fs::create_dir_all(&deep).expect("dirs");
        assert_eq!(detect(Some(&deep), None), Some(game.path().to_path_buf()));
    }

    #[test]
    fn the_last_used_folder_wins_and_a_stale_one_is_skipped() {
        let last = game_folder();
        let near = game_folder();
        let run = near.path().join("bin");
        std::fs::create_dir_all(&run).expect("dirs");
        assert_eq!(
            detect(Some(&run), Some(last.path())),
            Some(last.path().to_path_buf())
        );
        let stale = near.path().join("gone");
        assert_eq!(
            detect(Some(&run), Some(&stale)),
            Some(near.path().to_path_buf())
        );
    }

    #[test]
    fn nothing_found_gives_none() {
        let empty = tempfile::tempdir().expect("temp dir");
        assert_eq!(detect(Some(empty.path()), None), None);
        assert_eq!(detect(None, None), None);
    }
}
