//! Finding the AnyXboxX clone (the folder holding `third_party\victory\Cargo.toml`): the folder
//! the player chose in Setup, then `ANYXBOXX_REPO`, then the launcher's own surroundings, then a
//! few usual places. The decisions are pure (they take a predicate), so they are tested without
//! a real clone.

use std::path::{Path, PathBuf};

/// Shown when a chosen folder is not a clone.
pub const NOT_A_CLONE: &str = "This folder is not an AnyXboxX clone: it has no third_party\\victory\\Cargo.toml. Clone https://github.com/LooperSalty/AnyXboxX and pick that folder.";

/// Shown when nothing was found.
pub const NOT_FOUND: &str = "AnyXboxX folder not found. Put anyxboxx-launcher.exe inside your AnyXboxX clone, or choose the clone folder in Setup \u{2192} AnyXboxX folder.";

/// Where the folder in use came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Chosen in Setup (saved in the settings).
    Chosen,
    /// The `ANYXBOXX_REPO` environment variable.
    Environment,
    /// A parent of the launcher or of the current folder.
    Nearby,
    /// One of the usual places.
    Usual,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Chosen => "chosen in this window",
            Source::Environment => "from ANYXBOXX_REPO",
            Source::Nearby => "found next to the launcher",
            Source::Usual => "found in a usual place",
        }
    }
}

/// True when `dir` holds `third_party\victory\Cargo.toml`.
pub fn is_clone(dir: &Path) -> bool {
    dir.join("third_party")
        .join("victory")
        .join("Cargo.toml")
        .is_file()
}

/// Checks a folder the player chose.
pub fn validate(dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    if !is_clone(dir) {
        return Err(NOT_A_CLONE.to_string());
    }
    Ok(())
}

/// The first of `candidates` that `accepts`.
pub fn first_match(
    candidates: impl IntoIterator<Item = PathBuf>,
    accepts: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    candidates.into_iter().find(|dir| accepts(dir))
}

/// The inputs of the search, as plain values.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub chosen: Option<PathBuf>,
    pub environment: Option<PathBuf>,
    /// Parents of the launcher and of the current folder, nearest first.
    pub nearby: Vec<PathBuf>,
    /// The usual places, in order.
    pub usual: Vec<PathBuf>,
}

/// Applies the order: chosen, environment, nearby, usual. Each candidate must pass `accepts`; a
/// stale one is skipped.
pub fn resolve(inputs: &Inputs, accepts: impl Fn(&Path) -> bool) -> Option<(PathBuf, Source)> {
    let steps = [
        (Source::Chosen, inputs.chosen.iter().cloned().collect::<Vec<_>>()),
        (Source::Environment, inputs.environment.iter().cloned().collect()),
        (Source::Nearby, inputs.nearby.clone()),
        (Source::Usual, inputs.usual.clone()),
    ];
    steps
        .into_iter()
        .find_map(|(source, list)| first_match(list, &accepts).map(|dir| (dir, source)))
}

/// The usual places: the launcher's parent folder and its direct neighbours named `AnyXboxX*`,
/// then a few folders under the user profile. `list_dir` lists a folder's sub-folders.
pub fn usual_places(
    exe_dir: Option<&Path>,
    user_profile: Option<&Path>,
    list_dir: impl Fn(&Path) -> Vec<PathBuf>,
) -> Vec<PathBuf> {
    let mut places = Vec::new();
    if let Some(parent) = exe_dir.and_then(Path::parent) {
        places.push(parent.to_path_buf());
        places.extend(named_like_repo(list_dir(parent)));
    }
    if let Some(home) = user_profile {
        places.push(home.join("AnyXboxX"));
        places.push(home.join("Documents").join("AnyXboxX"));
        places.push(home.join("source").join("repos").join("AnyXboxX"));
        places.push(home.join("Desktop").join("AnyXboxX"));
        places.push(home.join("Downloads").join("AnyXboxX"));
        places.extend(named_like_repo(list_dir(&home.join("Downloads"))));
    }
    places
}

/// The entries whose name starts with `AnyXboxX`, any case.
fn named_like_repo(entries: Vec<PathBuf>) -> Vec<PathBuf> {
    entries
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.to_ascii_lowercase().starts_with("anyxboxx"))
        })
        .collect()
}

/// The sub-folders of `dir` (nothing when it cannot be read).
pub fn list_subfolders(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect()
}

/// Reads the machine: the environment, the parents and the usual places.
fn gather(chosen: Option<PathBuf>) -> Inputs {
    let exe_dir = crate::picker::launcher_dir();
    let cwd = std::env::current_dir().ok();
    let environment = std::env::var_os("ANYXBOXX_REPO")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let user_profile = std::env::var_os("USERPROFILE").map(PathBuf::from);
    let nearby = [exe_dir.clone(), cwd]
        .into_iter()
        .flatten()
        .flat_map(|start| start.ancestors().map(Path::to_path_buf).collect::<Vec<_>>())
        .collect();
    Inputs {
        chosen,
        environment,
        nearby,
        usual: usual_places(exe_dir.as_deref(), user_profile.as_deref(), list_subfolders),
    }
}

/// The folder in use and where it came from, reading the machine.
pub fn locate(chosen: Option<PathBuf>) -> Option<(PathBuf, Source)> {
    resolve(&gather(chosen), is_clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clone_at(dir: &Path) {
        let victory = dir.join("third_party").join("victory");
        std::fs::create_dir_all(&victory).expect("dirs");
        std::fs::write(victory.join("Cargo.toml"), b"[package]").expect("write");
    }

    #[test]
    fn a_clone_is_valid_and_other_folders_get_a_clear_reason() {
        let repo = tempfile::tempdir().expect("temp dir");
        clone_at(repo.path());
        assert!(validate(repo.path()).is_ok());
        let empty = tempfile::tempdir().expect("temp dir");
        assert_eq!(validate(empty.path()), Err(NOT_A_CLONE.to_string()));
        let missing = empty.path().join("nope");
        assert!(
            validate(&missing)
                .expect_err("no folder")
                .contains("not a folder")
        );
    }

    #[test]
    fn the_order_is_chosen_then_environment_then_nearby_then_usual() {
        let all = |_: &Path| true;
        let inputs = Inputs {
            chosen: Some("c".into()),
            environment: Some("e".into()),
            nearby: vec!["n".into()],
            usual: vec!["u".into()],
        };
        assert_eq!(resolve(&inputs, all), Some(("c".into(), Source::Chosen)));
        let no_chosen = Inputs {
            chosen: None,
            ..inputs.clone()
        };
        assert_eq!(
            resolve(&no_chosen, all),
            Some(("e".into(), Source::Environment))
        );
        let only_near = Inputs {
            environment: None,
            ..no_chosen
        };
        assert_eq!(resolve(&only_near, all), Some(("n".into(), Source::Nearby)));
        let only_usual = Inputs {
            nearby: vec![],
            ..only_near
        };
        assert_eq!(resolve(&only_usual, all), Some(("u".into(), Source::Usual)));
    }

    #[test]
    fn a_stale_choice_falls_through_to_the_next_source() {
        let inputs = Inputs {
            chosen: Some("gone".into()),
            environment: Some("gone2".into()),
            nearby: vec!["x".into()],
            usual: vec![],
        };
        let only_x = |p: &Path| p == Path::new("x");
        assert_eq!(resolve(&inputs, only_x), Some(("x".into(), Source::Nearby)));
        assert_eq!(resolve(&Inputs::default(), |_| true), None);
        assert_eq!(resolve(&inputs, |_| false), None);
    }

    #[test]
    fn the_usual_places_include_neighbours_and_the_profile_folders() {
        let list = |dir: &Path| -> Vec<PathBuf> {
            if dir == Path::new("D:/Tools") {
                vec!["D:/Tools/anyxboxx-main".into(), "D:/Tools/Other".into()]
            } else if dir == Path::new("C:/Users/p/Downloads") {
                vec![
                    "C:/Users/p/Downloads/AnyXboxX-1.2".into(),
                    "C:/Users/p/Downloads/Music".into(),
                ]
            } else {
                vec![]
            }
        };
        let places = usual_places(
            Some(Path::new("D:/Tools/launcher")),
            Some(Path::new("C:/Users/p")),
            list,
        );
        let has = |p: &str| places.iter().any(|x| x == Path::new(p));
        assert!(has("D:/Tools"));
        assert!(has("D:/Tools/anyxboxx-main"));
        assert!(!has("D:/Tools/Other"));
        assert!(has("C:/Users/p/AnyXboxX"));
        assert!(has("C:/Users/p/Documents/AnyXboxX"));
        assert!(has("C:/Users/p/source/repos/AnyXboxX"));
        assert!(has("C:/Users/p/Desktop/AnyXboxX"));
        assert!(has("C:/Users/p/Downloads/AnyXboxX"));
        assert!(has("C:/Users/p/Downloads/AnyXboxX-1.2"));
        assert!(!has("C:/Users/p/Downloads/Music"));
        assert!(usual_places(None, None, |_| vec![]).is_empty());
    }

    #[test]
    fn the_search_finds_a_real_clone_among_the_usual_places() {
        let root = tempfile::tempdir().expect("temp dir");
        let repo = root.path().join("AnyXboxX-main");
        clone_at(&repo);
        let places = usual_places(Some(&root.path().join("bin")), None, list_subfolders);
        assert_eq!(first_match(places, is_clone), Some(repo));
    }
}
