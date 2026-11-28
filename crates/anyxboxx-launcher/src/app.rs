//! The window's state and the actions behind its buttons. Long work (folder checks, VIctory's
//! translate and game) runs on threads; this module drains their results once per frame.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

use crate::app_check::ManifestState;
use crate::app_launch::Banner;
use crate::cache;
use crate::host::{self, Adapter, GpuVendor, Host};
use crate::job::{self, Event, Step};
use crate::launch;
use crate::launch_progress::{self, Stage, Tracker};
use crate::manifest::Outcome;
use crate::picker::{self, Pick};
use crate::processes::Tasklist;
use crate::profile::{Profile, Profiles};
use crate::progress::{Progress, Summary};
use crate::session::{self, Settings};
use crate::setup::{self, Item};
use crate::splash::Splash;
use crate::store::{self, Stored};
use crate::toast::Toasts;
use crate::verify::Report;

/// Lines kept in the log view.
const LOG_CAPACITY: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Home,
    Setup,
    Files,
    Shaders,
    Play,
}

/// The game folder check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verify {
    NotRun,
    Running,
    Done(Report),
    Failed(String),
}

/// A job running on its thread, whether it is the shader step, its stages and its cancel switch.
pub struct Running {
    pub events: Receiver<Event>,
    pub shaders_only: bool,
    pub stages: Vec<Stage>,
    pub cancel: Arc<AtomicBool>,
}

pub struct App {
    pub repo: Option<PathBuf>,
    /// Where `repo` came from, for the Setup page.
    pub repo_source: Option<crate::repo_dir::Source>,
    /// Prebuilt `victory.exe` passed via `--victory-exe`: skips the cargo build and needs no repo.
    pub victory_exe: Option<PathBuf>,
    pub profiles: Profiles,
    pub host: Host,
    pub gpu: GpuVendor,
    pub adapter_name: Option<String>,
    pub framegen_unavailable: Option<&'static str>,
    pub streamline_check: Option<(String, Result<(), String>)>,
    adapters: Option<Vec<Adapter>>,
    pub setup: Vec<Item>,
    pub settings: Settings,
    pub tab: Tab,
    pub verify: Verify,
    pub(crate) verify_rx: Option<Receiver<Result<Report, String>>>,
    pub(crate) verify_started: Option<std::time::Instant>,
    /// The optional SHA-256 manifest job; it never gates PLAY.
    pub manifest: ManifestState,
    pub(crate) manifest_rx: Option<Receiver<Outcome>>,
    pub running: Option<Running>,
    pub log: VecDeque<String>,
    pub summary: Summary,
    pub notice: Option<String>,
    /// Set when the game is up and the launcher may close (see `keep_open_after_launch`).
    pub close_requested: bool,
    pub confirm_clean: bool,
    pub cache_dir: Option<PathBuf>,
    pub splash: Splash,
    pub toasts: Toasts,
    /// The step the running job is on (`Building VIctory`, `Running the game`...).
    pub step: Option<String>,
    pub drawer_open: bool,
    /// Whether the shader cache folder holds something (looked at every couple of seconds).
    pub cache_ready: bool,
    pub cache_checked_at: f64,
    /// When the page on screen was switched to, in egui seconds (for the page transition).
    pub tab_changed_at: f64,
    pub shown_tab: Tab,
    /// Start the game as soon as the folder check passes (PLAY pressed before it ended).
    pub launch_when_ready: bool,
    /// The launch screen's model: stage, progress, pace.
    pub tracker: Tracker,
    /// The outcome of the last job, shown as a card until dismissed or replaced.
    pub banner: Option<Banner>,
    /// Shows the launch screen without running anything (`--preview-launch`, debug builds only).
    pub preview_launch: bool,
    pub(crate) clock: Instant,
    pick_rx: Option<Receiver<(Pick, Option<PathBuf>)>>,
    settings_path: Option<PathBuf>,
    last_saved: Stored,
    title_id_cache: Option<(PathBuf, u32)>,
}

/// The checklist as the machine is now; `adapters` is what DXGI listed (None when it failed).
fn read_setup(adapters: Option<&[crate::host::Adapter]>) -> Vec<Item> {
    let system_root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    let program_files = std::env::var_os("ProgramFiles(x86)")
        .map_or_else(|| PathBuf::from("C:\\Program Files (x86)"), PathBuf::from);
    let kits = program_files.join("Windows Kits").join("10").join("bin");
    let facts = setup::probe(
        &PathBuf::from(system_root),
        &kits,
        adapters.map(<[_]>::to_vec),
    );
    setup::checklist(&facts)
}

impl App {
    /// The window's first state: the setup checklist, the GPU and the host are read now.
    pub fn new() -> Result<Self, String> {
        let profiles = Profiles::embedded().map_err(|e| e.to_string())?;
        let settings_path = store::current_path();
        let settings = settings_path
            .as_deref()
            .and_then(store::load)
            .map(Stored::into_settings)
            .unwrap_or_default();
        let settings_copy = settings.clone();
        let host = host::detect_host();
        let adapters = host::enumerate_adapters().ok();
        let gpu = adapters
            .as_deref()
            .map_or(GpuVendor::Unknown, host::vendor_of);
        let adapter_name = adapters
            .as_deref()
            .and_then(host::main_adapter)
            .map(|a| a.name.clone());
        let mut app = App {
            repo: None,
            repo_source: None,
            victory_exe: None,
            profiles,
            host,
            gpu,
            framegen_unavailable: crate::framegen::unavailable(
                gpu, adapter_name.as_deref().unwrap_or_default(),
                host == Host::Windows && crate::framegen::detect_hags(),
            ),
            streamline_check: None,
            adapter_name,
            setup: read_setup(adapters.as_deref()),
            adapters,
            settings,
            tab: Tab::Home,
            verify: Verify::NotRun,
            verify_rx: None,
            verify_started: None,
            manifest: ManifestState::NotRun,
            manifest_rx: None,
            running: None,
            log: VecDeque::new(),
            summary: Summary::default(),
            notice: None,
            close_requested: false,
            confirm_clean: false,
            cache_dir: cache::current_cache_dir(),
            splash: Splash::new(),
            toasts: Toasts::default(),
            step: None,
            drawer_open: false,
            cache_ready: false,
            cache_checked_at: 0.0,
            tab_changed_at: 0.0,
            shown_tab: Tab::Home,
            last_saved: Stored::from_settings(&settings_copy),
            launch_when_ready: false,
            tracker: Tracker::default(),
            banner: None,
            preview_launch: false,
            clock: Instant::now(),
            pick_rx: None,
            settings_path,
            title_id_cache: None,
        };
        app.find_game_folder();
        app.refresh_repo();
        if app.settings.game_path().is_some() {
            app.start_verify();
        }
        Ok(app)
    }

    /// Keeps the saved game folder when it still holds the game, else looks for one next to the
    /// launcher and in its parents.
    fn find_game_folder(&mut self) {
        let last = self.settings.game_path();
        let launcher = picker::launcher_dir();
        let found = picker::detect(launcher.as_deref(), last.as_deref());
        if let Some(dir) = found {
            self.settings.game_dir = dir.display().to_string();
        }
    }

    /// Opens the folder dialog on its own thread.
    pub fn pick_game_folder(&mut self) {
        self.open_dialog(Pick::GameFolder);
    }

    /// Opens the file dialog for the xg_xs.dll.
    pub fn pick_xg_library(&mut self) {
        self.open_dialog(Pick::XgLibrary);
    }

    pub(crate) fn open_dialog(&mut self, kind: Pick) {
        if self.pick_rx.is_some() {
            return;
        }
        let start = match kind {
            Pick::Streamline => Some(PathBuf::from(&self.settings.options.dlss.streamline_dir)),
            Pick::GameFolder => self.settings.game_path(),
            Pick::RepoFolder => self.repo.clone(),
            Pick::XgLibrary => PathBuf::from(self.settings.xg_dll.trim())
                .parent()
                .map(PathBuf::from),
        };
        self.pick_rx = Some(picker::open(kind, start));
    }

    /// Uses `dir` as the game folder when it is one (error toast otherwise), then checks it.
    pub fn set_game_folder(&mut self, dir: &std::path::Path) {
        match picker::validate_game_folder(dir) {
            Ok(()) => {
                self.settings.game_dir = dir.display().to_string();
                self.manifest = ManifestState::NotRun;
                self.notice = Some(format!("Game folder saved: {}", dir.display()));
                self.start_verify();
            }
            Err(reason) => self.notice = Some(format!("Cannot use this folder: {reason}")),
        }
    }

    /// Opens the folder dialog for the AnyXboxX clone.
    pub fn pick_repo_folder(&mut self) {
        self.open_dialog(Pick::RepoFolder);
    }

    /// Looks for the AnyXboxX folder again: the saved choice, `ANYXBOXX_REPO`, the launcher's
    /// surroundings, then the usual places.
    pub fn refresh_repo(&mut self) {
        let found = crate::repo_dir::locate(self.settings.repo_path());
        (self.repo, self.repo_source) = match found {
            Some((dir, source)) => (Some(dir), Some(source)),
            None => (None, None),
        };
    }

    /// Uses `dir` as the AnyXboxX folder when it is a clone (error toast otherwise).
    pub fn set_repo_folder(&mut self, dir: &std::path::Path) {
        match crate::repo_dir::validate(dir) {
            Ok(()) => {
                self.settings.repo_dir = dir.display().to_string();
                self.refresh_repo();
                self.notice = Some(format!("AnyXboxX folder saved: {}", dir.display()));
            }
            Err(reason) => self.notice = Some(format!("Cannot use this folder: {reason}")),
        }
    }

    /// Forgets the chosen folder; it is searched for again.
    pub fn clear_repo_folder(&mut self) {
        self.settings.repo_dir.clear();
        self.refresh_repo();
    }

    /// Takes the answer of a dialog, if one came.
    fn poll_dialog(&mut self) {
        let Some(rx) = &self.pick_rx else {
            return;
        };
        let Ok((kind, chosen)) = rx.try_recv() else {
            return;
        };
        self.pick_rx = None;
        let Some(path) = chosen else {
            return;
        };
        match kind {
            Pick::Streamline => self.settings.options.dlss.streamline_dir = path.display().to_string(),
            Pick::GameFolder => self.set_game_folder(&path),
            Pick::RepoFolder => self.set_repo_folder(&path),
            Pick::XgLibrary => self.settings.xg_dll = path.display().to_string(),
        }
    }

    /// A folder (or a file of it) dropped on the window becomes the game folder.
    pub fn handle_drops(&mut self, ctx: &eframe::egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone()));
        if let Some(path) = dropped {
            self.set_game_folder(&picker::folder_of_drop(&path));
        }
    }

    /// The profile for this host and GPU.
    pub fn profile(&self) -> &Profile {
        self.profiles.select(self.host, self.gpu)
    }

    /// Reads the checklist again (after the user installed something).
    pub fn refresh_setup(&mut self) {
        self.setup = read_setup(self.adapters.as_deref());
    }

    /// The repository folder, or an empty path when none was found (the plan is then only shown).
    pub fn repo_or_empty(&self) -> PathBuf {
        self.repo.clone().unwrap_or_default()
    }

    /// Where VIctory keeps the game's saves (`<data>\<TitleId>\saves`), or the data root when no
    /// game folder is set yet.
    pub fn data_dir_label(&mut self) -> String {
        let Some(data) = cache::current_data_dir() else {
            return "unknown".to_string();
        };
        let Some(game_dir) = self.settings.game_path() else {
            return data.display().to_string();
        };
        let title_id = match &self.title_id_cache {
            Some((cached, id)) if *cached == game_dir => *id,
            _ => {
                let id = cache::read_title_id(&game_dir);
                self.title_id_cache = Some((game_dir, id));
                id
            }
        };
        cache::title_saves_dir(&data, title_id)
            .display()
            .to_string()
    }

    pub fn is_busy(&self) -> bool {
        self.running.is_some() || matches!(self.verify, Verify::Running)
    }

    /// The last progress report of the translate, for the bar.
    pub fn progress(&self) -> Option<&Progress> {
        self.summary.last_progress.as_ref()
    }

    /// Starts the play steps, or the shader preloading alone; refused while a game runs.
    pub fn start(&mut self, precompile_only: bool) {
        if self.is_busy() {
            return;
        }
        if let Err(error) = session::guard(&self.settings, &Tasklist) {
            self.notice = Some(error.to_string());
            return;
        }
        let profile = self.profile().clone();
        let repo = match (&self.repo, &self.victory_exe) {
            (Some(repo), _) => repo.clone(),
            (None, Some(_)) => PathBuf::new(),
            (None, None) => {
                self.notice = Some(crate::repo_dir::NOT_FOUND.to_string());
                self.tab = Tab::Setup;
                return;
            }
        };
        let plan = match session::make_plan(
            &self.settings,
            &repo,
            self.victory_exe.as_deref(),
            &profile,
            self.framegen_unavailable.is_none(),
        ) {
            Ok(plan) => plan,
            Err(error) => {
                self.notice = Some(error.to_string());
                return;
            }
        };
        if let Err(reason) = crate::streamline::check_launch(&plan.env) {
            self.notice = Some(reason);
            return;
        }
        if !precompile_only {
            // NVIDIA Smooth Motion lives in the driver's profile, not in VIctory's environment:
            // written now, so the game starts with the switch as the Play page shows it. A failure
            // is reported and the game starts all the same.
            let on = self.settings.options.smooth_motion;
            match crate::smooth_motion::sync(on) {
                Ok(Some(done)) => self.log_line(format!("== {done} ==")),
                Ok(None) => {}
                Err(error) if on => {
                    self.notice = Some(format!("Smooth Motion could not be turned on: {error}"));
                }
                Err(_) => {}
            }
        }
        let steps: Result<Vec<Step>, anyxboxx::run::RunError> = if precompile_only {
            launch::precompile_steps(&plan, self.settings.options.warm_first)
        } else {
            launch::play_steps(&plan, self.settings.options.warm_first)
        };
        match steps {
            Ok(steps) => self.spawn_steps(steps, precompile_only),
            Err(error) => self.notice = Some(error.to_string()),
        }
    }

    fn spawn_steps(&mut self, steps: Vec<Step>, shaders_only: bool) {
        let labels: Vec<String> = steps.iter().map(|s| s.label.clone()).collect();
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        job::spawn_cancellable(steps, tx, Arc::clone(&cancel));
        self.running = Some(Running {
            events: rx,
            shaders_only,
            stages: launch_progress::stages_of(&labels),
            cancel,
        });
        self.step = None;
        self.summary = Summary::default();
        self.tracker = Tracker::start(self.now_secs());
        self.banner = None;
        self.notice = None;
    }

    /// Cleans the shader cache after the user confirmed.
    pub fn clean_cache(&mut self) {
        self.confirm_clean = false;
        let Some(dir) = self.cache_dir.clone() else {
            self.notice = Some("No cache folder: LOCALAPPDATA is not set.".to_string());
            return;
        };
        let cleaned = cache::clean_cache(&dir);
        // With the cache gone, the warm marker would make the next launch skip the warm.
        let forgotten =
            match crate::warm_state::marker_path(std::env::var_os("LOCALAPPDATA").as_deref()) {
                Some(marker) if cleaned.is_ok() => crate::warm_state::forget_marker(&marker)
                    .map_err(|e| format!(" (could not delete {}: {e})", marker.display())),
                _ => Ok(()),
            };
        let mut notice = match cleaned {
            Ok(cache::CleanOutcome::Removed) => format!("Shader cache removed: {}", dir.display()),
            Ok(cache::CleanOutcome::NothingToClean) => "No shader cache to remove.".to_string(),
            Err(error) => error.to_string(),
        };
        if let Err(warning) = forgotten {
            notice.push_str(&warning);
        }
        self.notice = Some(notice);
    }

    /// Drains the results of the threads started earlier.
    pub fn poll(&mut self) {
        self.poll_dialog();
        self.poll_verify();
        self.poll_manifest();
        self.poll_job();
    }

    /// Writes the settings file when something changed since the last write.
    pub fn save_settings_if_changed(&mut self) {
        let now = Stored::from_settings(&self.settings);
        if now == self.last_saved {
            return;
        }
        if let Some(path) = &self.settings_path
            && let Err(error) = store::save(path, &now)
        {
            self.notice = Some(format!("Could not save the settings: {error}"));
        }
        self.last_saved = now;
    }

    pub(crate) fn log_line(&mut self, line: String) {
        self.log.push_back(line);
        while self.log.len() > LOG_CAPACITY {
            self.log.pop_front();
        }
    }
}
