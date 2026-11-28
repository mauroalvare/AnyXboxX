//! What the window asks for, turned into a launch plan or a refusal. The window keeps text
//! fields; this module reads them, so the rules are tested without any window.

use std::path::{Path, PathBuf};

use anyxboxx::run::XgMode;

use crate::detached;
use crate::env::{self, EnvError, Xg};
use crate::launch::Plan;
use crate::options::LaunchOptions;
use crate::processes::{self, GuardError, ProcessSource};
use crate::profile::Profile;
use crate::warm_state;
use anyxboxx::run::Victory;

/// The window's fields, as typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub game_dir: String,
    /// The AnyXboxX clone chosen in Setup, empty for none (then it is searched for).
    pub repo_dir: String,
    /// The advanced panel: one `KEY=VALUE` per line.
    pub overrides: String,
    /// Path of the user's own `xg_xs.dll`, empty for none.
    pub xg_dll: String,
    pub xg_mode: XgMode,
    /// Worker threads of the translate step; 0 means every core.
    pub jobs: u32,
    /// The advanced panel's override of the one-game guard (for hosts without `tasklist`).
    pub skip_process_check: bool,
    /// The Play page's launch options.
    pub options: LaunchOptions,
    /// Keep the launcher open after the game started (off: it closes once the game is up).
    pub keep_open_after_launch: bool,
    /// "Always re-warm before launch": the shader warm runs even when the marker is current.
    pub always_rewarm: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            game_dir: String::new(),
            repo_dir: String::new(),
            overrides: String::new(),
            xg_dll: String::new(),
            xg_mode: XgMode::Real,
            jobs: 0,
            skip_process_check: false,
            options: LaunchOptions::default(),
            keep_open_after_launch: false,
            always_rewarm: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("choose the game folder first (the one holding game_scarlett_master.exe)")]
    NoGameFolder,
    #[error(transparent)]
    Env(#[from] EnvError),
}

impl Settings {
    /// The game folder as a path, when one was typed.
    pub fn game_path(&self) -> Option<PathBuf> {
        let trimmed = self.game_dir.trim().trim_matches('"');
        (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
    }

    /// The AnyXboxX folder chosen in Setup, when one was saved.
    pub fn repo_path(&self) -> Option<PathBuf> {
        let trimmed = self.repo_dir.trim().trim_matches('"');
        (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
    }

    /// The jobs limit, `None` for every core.
    pub fn jobs_limit(&self) -> Option<usize> {
        (self.jobs > 0).then_some(self.jobs as usize)
    }
}

/// The launch plan for `profile`, with the user's overrides and XG library applied.
pub fn make_plan(
    settings: &Settings,
    repo: &Path,
    victory_exe: Option<&Path>,
    profile: &Profile,
    fg_available: bool,
) -> Result<Plan, SessionError> {
    let game_dir = settings.game_path().ok_or(SessionError::NoGameFolder)?;
    let mut overrides = settings.options.env_pairs();
    let advanced = env::parse_overrides(&settings.overrides)?;
    overrides.extend(settings.options.dlss.env_pairs(fg_available, &advanced));
    overrides.extend(advanced);
    let xg_text = settings.xg_dll.trim().trim_matches('"');
    let xg = (!xg_text.is_empty()).then(|| Xg {
        dll: PathBuf::from(xg_text),
        mode: settings.xg_mode,
    });
    let env = env::build_env(&game_dir, profile, &overrides, xg.as_ref())?;
    let victory = match victory_exe {
        Some(exe) => Victory::prebuilt(exe),
        None => Victory::locate(repo, false),
    };
    let warm = victory.ok().and_then(|victory| {
        let marker = warm_state::marker_path(std::env::var_os("LOCALAPPDATA").as_deref());
        crate::launch::warm_check(
            &victory.exe,
            &env,
            &game_dir,
            settings.always_rewarm,
            settings.options.warm_first,
            marker,
        )
    });
    let game_log = detached::log_dir(std::env::var_os("LOCALAPPDATA").as_deref())
        .map(|dir| detached::game_log_path(&dir, unix_now()));
    Ok(Plan {
        repo: repo.to_path_buf(),
        victory_exe: victory_exe.map(Path::to_path_buf),
        game_dir,
        env,
        jobs: settings.jobs_limit(),
        game_log,
        warm,
    })
}

/// Seconds since the Unix epoch, for the log file name of one launch.
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The one-game guard, unless the user skips it in the advanced panel.
pub fn guard(settings: &Settings, source: &impl ProcessSource) -> Result<(), GuardError> {
    if settings.skip_process_check {
        return Ok(());
    }
    processes::ensure_no_game_running(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::GpuVendor;
    use crate::profile::Profiles;
    use std::io;

    struct Fake(io::Result<Vec<String>>);

    impl ProcessSource for Fake {
        fn image_names(&self) -> io::Result<Vec<String>> {
            match &self.0 {
                Ok(names) => Ok(names.clone()),
                Err(e) => Err(io::Error::new(e.kind(), e.to_string())),
            }
        }
    }

    fn game_folder() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("game_scarlett_master.exe"), b"x").expect("write");
        dir
    }

    fn profile() -> Profile {
        Profiles::embedded()
            .expect("profiles")
            .select(crate::host::Host::Windows, GpuVendor::Nvidia)
            .clone()
    }

    #[test]
    fn plan_needs_a_game_folder() {
        let settings = Settings::default();
        assert!(matches!(
            make_plan(&settings, Path::new("."), None, &profile(), true),
            Err(SessionError::NoGameFolder)
        ));
    }

    #[test]
    fn plan_takes_the_jobs_limit_and_the_profile_env() {
        let dir = game_folder();
        let settings = Settings {
            game_dir: format!("\"{}\"", dir.path().display()),
            jobs: 6,
            ..Settings::default()
        };
        let plan = make_plan(&settings, Path::new("repo"), None, &profile(), true).expect("plan");
        assert_eq!(plan.jobs, Some(6));
        assert_eq!(plan.game_dir, dir.path());
        assert!(plan.env.iter().any(|(k, _)| k == "VICTORY_CARD_WINDOWS_MB"));
    }

    #[test]
    fn zero_jobs_means_every_core() {
        let settings = Settings {
            jobs: 0,
            ..Settings::default()
        };
        assert_eq!(settings.jobs_limit(), None);
    }

    #[test]
    fn overrides_from_the_panel_reach_the_plan() {
        let dir = game_folder();
        let settings = Settings {
            game_dir: dir.path().display().to_string(),
            overrides: "VICTORY_SHADER_OPT=12".to_string(),
            ..Settings::default()
        };
        let plan = make_plan(&settings, Path::new("repo"), None, &profile(), true).expect("plan");
        let value = plan
            .env
            .iter()
            .find(|(k, _)| k == "VICTORY_SHADER_OPT")
            .map(|(_, v)| v.clone());
        assert_eq!(value.as_deref(), Some(std::ffi::OsStr::new("12")));
    }

    #[test]
    fn a_bad_override_refuses_the_plan() {
        let dir = game_folder();
        let settings = Settings {
            game_dir: dir.path().display().to_string(),
            overrides: "NOT_A_VICTORY_KEY=1".to_string(),
            ..Settings::default()
        };
        assert!(matches!(
            make_plan(&settings, Path::new("repo"), None, &profile(), true),
            Err(SessionError::Env(EnvError::Key(1, _)))
        ));
    }

    #[test]
    fn advanced_dlss_settings_win_and_the_final_folder_is_checked() {
        let game = game_folder();
        let sdk = tempfile::tempdir().unwrap();
        let mut settings = Settings {
            game_dir: game.path().display().to_string(),
            overrides: format!(
                "VICTORY_STREAMLINE_DIR={}\nVICTORY_DLSS_MODE=balanced\nVICTORY_UPSCALER=3",
                sdk.path().display()
            ),
            ..Settings::default()
        };
        settings.options.dlss.super_resolution = true;
        let plan = make_plan(&settings, Path::new("."), None, &profile(), true).unwrap();
        let lines = env::describe(&plan.env);
        assert!(lines.contains(&"VICTORY_DLSS_MODE=balanced".into()));
        assert!(lines.contains(&"VICTORY_UPSCALER=3".into()));
        let refusal = crate::streamline::check_launch(&plan.env).unwrap_err();
        assert!(
            refusal.contains("sl.interposer.dll")
                && refusal.contains(&*sdk.path().display().to_string())
        );
        settings.overrides = "VICTORY_DLSS=0".into();
        let plan = make_plan(&settings, Path::new("."), None, &profile(), true).unwrap();
        assert!(crate::streamline::check_launch(&plan.env).is_ok());
        // The profile itself may set the upscaler (the local player build does);
        // what the options add must not.
        let advanced = env::parse_overrides(&settings.overrides).unwrap();
        let added = settings.options.dlss.env_pairs(true, &advanced);
        assert!(!added.iter().any(|(k, _)| k == "VICTORY_UPSCALER"));
        assert!(plan.env.iter().any(|(k, v)| k == "VICTORY_DLSS" && v == "0"));
    }

    #[test]
    fn a_typed_xg_library_that_is_missing_refuses_the_plan() {
        let dir = game_folder();
        let settings = Settings {
            game_dir: dir.path().display().to_string(),
            xg_dll: "Z:/nowhere/xg_xs.dll".to_string(),
            ..Settings::default()
        };
        assert!(make_plan(&settings, Path::new("repo"), None, &profile(), true).is_err());
    }

    #[test]
    fn plan_with_prebuilt_victory_exe_computes_warm_state_without_repo() {
        let dir = game_folder();
        std::fs::write(dir.path().join("shaders.rpf"), b"rpf").expect("write");
        let prebuilt_dir = tempfile::tempdir().expect("tempdir");
        let exe = prebuilt_dir.path().join("victory.exe");
        std::fs::write(&exe, b"MZ").expect("write");
        let settings = Settings {
            game_dir: dir.path().display().to_string(),
            ..Settings::default()
        };
        let plan = make_plan(&settings, Path::new(""), Some(&exe), &profile(), true).expect("plan");
        assert_eq!(plan.victory_exe.as_deref(), Some(exe.as_path()));
        assert!(
            plan.warm.is_some(),
            "warm check should fingerprint the prebuilt victory.exe"
        );
    }

    #[test]
    fn guard_refuses_a_running_game_unless_skipped() {
        let running = Fake(Ok(vec!["victory.exe".to_string()]));
        assert!(guard(&Settings::default(), &running).is_err());
        let skip = Settings {
            skip_process_check: true,
            ..Settings::default()
        };
        assert!(guard(&skip, &running).is_ok());
    }

    #[test]
    fn guard_refuses_when_the_list_cannot_be_read() {
        let broken = Fake(Err(io::Error::other("no tasklist")));
        assert!(matches!(
            guard(&Settings::default(), &broken),
            Err(GuardError::Unknown(_))
        ));
    }
}
