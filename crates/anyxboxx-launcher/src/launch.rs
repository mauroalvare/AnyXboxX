//! The launch steps, built on the shared pieces of `anyxboxx::run` (the same commands as
//! `anyxboxx run`): build VIctory, warm the shader cache when it is not ready (see `warm_state`),
//! then run the game. The environment is the same for the warm and the game.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyxboxx::run::{Env, GAME_EXE, PRECOMPILED_DLLS, RunError, SHADERS, Victory, warm_archives};

use crate::job::Step;
use crate::warm_state;

/// What a launch needs, decided by the window.
#[derive(Debug, Clone)]
pub struct Plan {
    pub repo: PathBuf,
    /// Prebuilt `victory.exe` (`--victory-exe`); when set, skips `cargo build` and needs no repo.
    pub victory_exe: Option<PathBuf>,
    pub game_dir: PathBuf,
    pub env: Env,
    /// Worker threads of the warm step; `None` lets VIctory use every core.
    pub jobs: Option<usize>,
    /// The game's log file: when set, the game starts detached and writes its output there.
    pub game_log: Option<PathBuf>,
    /// The shader cache check of this launch; `None` when it could not be made.
    pub warm: Option<WarmState>,
}

/// Whether the shader cache must be warmed before the game, and where the marker goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmState {
    pub needed: bool,
    pub marker: Option<PathBuf>,
    pub fingerprint: String,
}

fn resolve_victory(plan: &Plan) -> Result<Victory, RunError> {
    match &plan.victory_exe {
        Some(exe) => Victory::prebuilt(exe),
        None => Victory::locate(&plan.repo, false),
    }
}

/// The ordered, deduplicated inputs of `victory warm`: the game folder's own `.rpf` archives,
/// plus `shaders.rpf` (if not already present), `game_scarlett_master.exe` and its two DLLs when
/// `include_precompile` is true.
fn warm_inputs(game_dir: &Path, include_precompile: bool) -> Result<Vec<PathBuf>, RunError> {
    let mut inputs = warm_archives(game_dir)?;
    if include_precompile {
        let shaders = game_dir.join(SHADERS);
        let dlls: Vec<PathBuf> = PRECOMPILED_DLLS.iter().map(|d| game_dir.join(d)).collect();
        for needed in std::iter::once(&shaders).chain(&dlls) {
            if !needed.is_file() {
                return Err(RunError::PrecompileInput(needed.display().to_string()));
            }
        }
        let mut seen: HashSet<String> = inputs
            .iter()
            .filter_map(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .collect();
        for path in [shaders, game_dir.join(GAME_EXE)]
            .into_iter()
            .chain(dlls)
        {
            let key = path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().to_ascii_lowercase());
            if seen.insert(key) {
                inputs.push(path);
            }
        }
    }
    Ok(inputs)
}

/// The steps of a launch, in order. `warm_first` also includes shaders.rpf, the game executable
/// and its DLLs in the single deduplicated warm step (and its smart-warm marker).
///
/// When VIctory is built first, a warm that looked unneeded is still listed: it checks the
/// fingerprint again once the build is over, because a rebuilt `victory.exe` changes it.
pub fn play_steps(plan: &Plan, warm_first: bool) -> Result<Vec<Step>, RunError> {
    let victory = resolve_victory(plan)?;
    anyxboxx::run::game_exe(&plan.game_dir)?;
    let build = build_step(&victory);
    let rebuilds = build.is_some();
    let mut steps: Vec<Step> = build.into_iter().collect();
    if let Some(warm) = plan.warm.as_ref() {
        let step = warm_step(&victory, plan, warm_first, warm.marker.clone())?;
        if warm.needed {
            steps.push(step);
        } else if rebuilds {
            steps.push(step.with_run_if(stale_after_build(
                &victory,
                plan,
                warm_first,
                warm.marker.clone(),
            )?));
        }
    } else if warm_first {
        steps.push(precompile_step(&victory, plan)?);
    }
    let command = victory.game_command(&plan.game_dir, &plan.env);
    steps.push(match &plan.game_log {
        Some(log) => Step::detached("Running the game", command, log.clone()),
        None => Step::new("Running the game", command),
    });
    Ok(steps)
}

/// "Warm now": the full warm of the game's archives (and game binaries when `warm_first` is on),
/// whatever the marker says.
pub fn precompile_steps(plan: &Plan, warm_first: bool) -> Result<Vec<Step>, RunError> {
    let victory = resolve_victory(plan)?;
    anyxboxx::run::game_exe(&plan.game_dir)?;
    let mut steps: Vec<Step> = build_step(&victory).into_iter().collect();
    let marker = plan.warm.as_ref().and_then(|warm| warm.marker.clone());
    steps.push(warm_step(&victory, plan, warm_first, marker)?);
    Ok(steps)
}

/// The fingerprint of `exe`, `env` and `archives` as they are now.
fn fingerprint_now(exe: &Path, env: &Env, archives: &[PathBuf]) -> String {
    let identities = warm_state::archive_identities(archives);
    warm_state::fingerprint(&warm_state::build_id(exe), env, &identities)
}

/// The check of a warm listed after a build: needed when the marker no longer matches.
fn stale_after_build(
    victory: &Victory,
    plan: &Plan,
    include_precompile: bool,
    marker: Option<PathBuf>,
) -> Result<crate::job::RunIf, RunError> {
    let (exe, env) = (victory.exe.clone(), plan.env.clone());
    let inputs = warm_inputs(&plan.game_dir, include_precompile)?;
    Ok(Box::new(move || {
        let stored = marker.as_deref().and_then(warm_state::read_marker);
        warm_state::needs_warm(
            false,
            stored.as_deref(),
            &fingerprint_now(&exe, &env, &inputs),
        )
    }))
}

/// The cargo build of VIctory; `None` for a `victory.exe` built beforehand.
fn build_step(victory: &Victory) -> Option<Step> {
    victory
        .build_command()
        .map(|command| Step::new("Building VIctory", command))
}

/// `victory warm` on the game folder's own `.rpf` archives (plus the game executable and its DLLs
/// when `include_precompile` is true, deduplicated), with the game's environment. After a success
/// the marker gets the fingerprint of that moment (after any build), so the next launch skips the
/// warm while nothing changed.
fn warm_step(
    victory: &Victory,
    plan: &Plan,
    include_precompile: bool,
    marker: Option<PathBuf>,
) -> Result<Step, RunError> {
    let inputs = warm_inputs(&plan.game_dir, include_precompile)?;
    let mut command = victory.warm_command(&inputs, &plan.env);
    if let Some(jobs) = plan.jobs {
        command.arg("--jobs").arg(jobs.to_string());
    }
    let step = Step::new("Warming the shader cache", command);
    Ok(match marker {
        Some(path) => {
            let (exe, env) = (victory.exe.clone(), plan.env.clone());
            step.with_after(Box::new(move || {
                warm_state::write_marker(&path, &fingerprint_now(&exe, &env, &inputs))
                    .map_err(|e| format!("could not write {}: {e}", path.display()))
            }))
        }
        None => step,
    })
}

/// `victory warm` on shaders.rpf, the game and its DLLs (fallback when no warm state exists).
fn precompile_step(victory: &Victory, plan: &Plan) -> Result<Step, RunError> {
    let mut command = victory.precompile_command(&plan.game_dir, &plan.env)?;
    if let Some(jobs) = plan.jobs {
        command.arg("--jobs").arg(jobs.to_string());
    }
    Ok(Step::new("Preloading shaders", command))
}

/// The fingerprint of this launch: the victory.exe identity, the environment, and the warm inputs.
pub fn warm_check(
    victory_exe: &Path,
    env: &Env,
    game_dir: &Path,
    always: bool,
    include_precompile: bool,
    marker: Option<PathBuf>,
) -> Option<WarmState> {
    let inputs = warm_inputs(game_dir, include_precompile).ok()?;
    let fingerprint = fingerprint_now(victory_exe, env, &inputs);
    let stored = marker.as_deref().and_then(warm_state::read_marker);
    let needed = warm_state::needs_warm(always, stored.as_deref(), &fingerprint);
    Some(WarmState {
        needed,
        marker,
        fingerprint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::path::Path;
    use std::process::Command;

    /// The arguments a command receives, for the assertions.
    fn args_of(command: &Command) -> Vec<String> {
        command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    fn has_victory(repo: &Path) -> bool {
        repo.join("third_party")
            .join("victory")
            .join("Cargo.toml")
            .is_file()
    }

    /// A repository with a VIctory manifest and a game folder holding the files translate needs.
    /// The temp folders are returned so they live as long as the test.
    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, Plan) {
        let repo = tempfile::tempdir().expect("tempdir");
        let victory = repo.path().join("third_party").join("victory");
        std::fs::create_dir_all(&victory).expect("mkdir");
        std::fs::write(victory.join("Cargo.toml"), b"[workspace]\n").expect("write");
        let game = tempfile::tempdir().expect("tempdir");
        for name in [
            "game_scarlett_master.exe",
            "shaders.rpf",
            "XboxBvhBuilder.dll",
            "amd_fidelityfx_upscaler_xbox.dll",
        ] {
            std::fs::write(game.path().join(name), b"x").expect("write");
        }
        let plan = Plan {
            repo: repo.path().to_path_buf(),
            victory_exe: None,
            game_dir: game.path().to_path_buf(),
            game_log: None,
            warm: None,
            env: vec![("VICTORY_LATE_WAIT".to_string(), OsString::from("3"))],
            jobs: Some(4),
        };
        (repo, game, plan)
    }

    #[test]
    fn play_runs_build_then_game_without_precompile() {
        let (_repo, _game, plan) = fixture();
        let steps = play_steps(&plan, false).expect("steps");
        let labels: Vec<&str> = steps.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["Building VIctory", "Running the game"]);
        let game_args = args_of(&steps[1].command);
        assert_eq!(game_args.first().map(String::as_str), Some("run"));
        assert!(game_args.iter().any(|a| a == "--execute"));
    }

    /// A warm that looked unneeded before the build is listed, and decides once the build is over:
    /// a marker written for another victory.exe makes it run, a marker of this one skips it.
    #[test]
    fn a_warm_after_a_build_checks_the_fingerprint_of_the_new_exe() {
        let (_repo, _game, mut plan) = fixture();
        let marker_dir = tempfile::tempdir().expect("tempdir");
        let marker = marker_dir.path().join("warm-state.txt");
        plan.warm = Some(WarmState {
            needed: false,
            marker: Some(marker.clone()),
            fingerprint: "before the build".to_string(),
        });
        let exe = Victory::locate(&plan.repo, false).expect("victory").exe;
        let archives = warm_archives(&plan.game_dir).expect("archives");
        let check = |plan: &Plan| {
            let mut steps = play_steps(plan, false).expect("steps");
            let warm = steps
                .iter_mut()
                .find(|s| s.label == "Warming the shader cache")
                .expect("warm listed after the build");
            (warm.run_if.take().expect("decided after the build"))()
        };
        warm_state::write_marker(&marker, "fingerprint of the old victory.exe").expect("marker");
        assert!(check(&plan));
        warm_state::write_marker(&marker, &fingerprint_now(&exe, &plan.env, &archives))
            .expect("marker");
        assert!(!check(&plan));
    }

    #[test]
    fn precompile_step_passes_the_jobs_limit_after_the_inputs() {
        let (_repo, _game, plan) = fixture();
        let steps = precompile_steps(&plan, false).expect("steps");
        let args = args_of(&steps[1].command);
        assert_eq!(args.first().map(String::as_str), Some("warm"));
        let at = args
            .iter()
            .position(|a| a == "--jobs")
            .expect("--jobs given");
        assert_eq!(args.get(at + 1).map(String::as_str), Some("4"));
    }

    #[test]
    fn no_jobs_limit_means_no_flag() {
        let (_repo, _game, mut plan) = fixture();
        plan.jobs = None;
        let steps = precompile_steps(&plan, false).expect("steps");
        assert!(!args_of(&steps[1].command).iter().any(|a| a == "--jobs"));
    }

    #[test]
    fn a_missing_game_exe_refuses_the_launch() {
        let (_repo, _game, mut plan) = fixture();
        plan.game_dir = PathBuf::from("Z:/no/such/game/folder");
        assert!(play_steps(&plan, true).is_err());
    }

    #[test]
    fn a_repository_without_victory_is_an_error() {
        let (_repo, _game, mut plan) = fixture();
        let empty = tempfile::tempdir().expect("tempdir");
        plan.repo = empty.path().to_path_buf();
        assert!(!has_victory(&plan.repo));
        assert!(play_steps(&plan, false).is_err());
    }

    #[test]
    fn prebuilt_victory_exe_skips_build_without_requiring_a_repo() {
        let (_repo, _game, mut plan) = fixture();
        let prebuilt_dir = tempfile::tempdir().expect("tempdir");
        let exe = prebuilt_dir.path().join("victory.exe");
        std::fs::write(&exe, b"MZ").expect("write");
        plan.repo = PathBuf::new();
        plan.victory_exe = Some(exe.clone());

        let play = play_steps(&plan, false).expect("play steps");
        let play_labels: Vec<&str> = play.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(play_labels, vec!["Running the game"]);
        assert_eq!(Path::new(play[0].command.get_program()), exe.as_path());

        let warm = precompile_steps(&plan, false).expect("warm steps");
        let warm_labels: Vec<&str> = warm.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(warm_labels, vec!["Warming the shader cache"]);
        assert_eq!(Path::new(warm[0].command.get_program()), exe.as_path());
    }

    #[test]
    fn a_missing_prebuilt_victory_exe_refuses_the_launch() {
        let (_repo, _game, mut plan) = fixture();
        plan.victory_exe = Some(PathBuf::from("Z:/no/such/victory.exe"));
        assert!(matches!(
            play_steps(&plan, false),
            Err(RunError::NoVictoryExe(_))
        ));
        assert!(matches!(
            precompile_steps(&plan, false),
            Err(RunError::NoVictoryExe(_))
        ));
    }

    #[test]
    fn warm_first_combines_into_one_deduplicated_warm_and_respects_warm_marker() {
        let (_repo, game, mut plan) = fixture();
        std::fs::write(game.path().join("common.rpf"), b"rpf").expect("write");
        let prebuilt_dir = tempfile::tempdir().expect("tempdir");
        let exe = prebuilt_dir.path().join("victory.exe");
        std::fs::write(&exe, b"MZ").expect("write");
        plan.victory_exe = Some(exe.clone());
        let marker_dir = tempfile::tempdir().expect("tempdir");
        let marker = marker_dir.path().join("warm-state.txt");

        // 1. Turning warm_first on changes the fingerprint relative to archives-only warm.
        let archives_only = warm_check(&exe, &plan.env, game.path(), false, false, Some(marker.clone()))
            .expect("archives-only warm check");
        warm_state::write_marker(&marker, &archives_only.fingerprint).expect("write marker");
        let with_binaries = warm_check(&exe, &plan.env, game.path(), false, true, Some(marker.clone()))
            .expect("warm_first warm check");
        assert!(
            with_binaries.needed,
            "enabling warm_first after an archives-only warm must warm the binaries once"
        );

        // 2. When warm is needed and warm_first is on: ONE victory warm step.
        plan.warm = Some(with_binaries);
        let mut steps = play_steps(&plan, true).expect("play steps");
        let warm_idx: Vec<usize> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| args_of(&s.command).first().map(String::as_str) == Some("warm"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            warm_idx.len(),
            1,
            "warm_first must not run two separate victory warm processes"
        );
        // Vice-Tay's `warm` takes no archives (it remakes the pipelines earlier runs noted): the
        // archives and binaries only enter the fingerprint, never the command line.
        let args = args_of(&steps[warm_idx[0]].command);
        assert!(
            !args.iter().any(|a| [".rpf", ".exe", ".dll"].iter().any(|ext| a.ends_with(ext))),
            "Vice-Tay's victory warm refuses file arguments: {args:?}"
        );

        // 3. After the combined warm writes its marker, the next launch skips warm completely.
        (steps[warm_idx[0]].after.take().expect("marker writer"))().expect("write marker");
        let after = warm_check(&exe, &plan.env, game.path(), false, true, Some(marker))
            .expect("warm check after marker write");
        assert!(!after.needed, "marker written by warm_first must match warm_check");
        plan.warm = Some(after);
        let steps_warm = play_steps(&plan, true).expect("play steps when already warm");
        assert!(
            !steps_warm
                .iter()
                .any(|s| args_of(&s.command).first().map(String::as_str) == Some("warm")),
            "warm_first must not re-run victory warm on every launch when cache is already warm"
        );
    }
}
