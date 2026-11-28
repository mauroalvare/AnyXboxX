use super::{Mode, plan};
use std::path::PathBuf;

/// What the user asked for, read from the environment once.
#[derive(Clone, Debug)]
pub struct Asked {
    pub dlss: bool,
    pub framegen: bool,
    pub mode: Mode,
    pub preset: u32,
    pub dir: PathBuf,
    /// `VICTORY_DLSS_JITTER` and `VICTORY_DLSS_MV`: sign or scale fixes.
    pub jitter: [f32; 2],
    pub mvec: [f32; 2],
    /// `VICTORY_DLSS_REACTIVE=0`: the reactive mask is not given to DLSS.
    pub reactive: bool,
    /// `VICTORY_STREAMLINE_LOG=<folder>`: Streamline's own log there, verbose.
    pub log_dir: Option<PathBuf>,
    /// `VICTORY_FRAMEGEN_HUDLESS=0`: with frame generation, no colour
    /// without the interface is copied or tagged where the title draws its
    /// interface into the frame (Series X).
    pub hudless: bool,
    /// `VICTORY_STREAMLINE_PRESENT=1`: with DLSS and no frame generation,
    /// our swap chain is Streamline's proxy of it, so that Streamline sees
    /// the presents (its bookkeeping per frame).
    pub present_hook: bool,
}

impl Asked {
    /// The switches; an unknown mode or preset name is an error.
    pub fn from_env() -> Result<Self, String> {
        let var = |name: &str| std::env::var(name).ok();
        let on = |name: &str| var(name).is_some_and(|v| v.trim() == "1");
        let framegen = on("VICTORY_FRAMEGEN");
        let mode = match var("VICTORY_DLSS_MODE") {
            Some(name) => Mode::from_name(&name)
                .ok_or_else(|| format!("VICTORY_DLSS_MODE={name}: not one of dlaa, quality, balanced, performance, ultraperformance"))?,
            // Frame generation alone keeps the scene at the output's size.
            None if framegen && !on("VICTORY_DLSS") => Mode::Dlaa,
            None => Mode::Quality,
        };
        let preset = match var("VICTORY_DLSS_PRESET") {
            Some(name) => plan::preset(&name).ok_or_else(|| format!("VICTORY_DLSS_PRESET={name}: not one of default, e, f, j, k, l, m"))?,
            None => 0,
        };
        Ok(Asked {
            dlss: on("VICTORY_DLSS"),
            framegen,
            mode,
            preset,
            dir: streamline_dir()?,
            jitter: plan::pair(var("VICTORY_DLSS_JITTER").as_deref()),
            mvec: plan::pair(var("VICTORY_DLSS_MV").as_deref()),
            reactive: var("VICTORY_DLSS_REACTIVE").is_none_or(|v| v.trim() != "0"),
            log_dir: var("VICTORY_STREAMLINE_LOG").filter(|v| !v.trim().is_empty() && v.trim() != "0").map(PathBuf::from),
            hudless: var("VICTORY_FRAMEGEN_HUDLESS").is_none_or(|v| v.trim() != "0"),
            present_hook: on("VICTORY_STREAMLINE_PRESENT"),
        })
    }

    pub fn any(&self) -> bool {
        self.dlss || self.framegen
    }
}

/// What [`init`] made available: the upscaling (the title's renderer may be
/// patched to the FidelityFX path for it) and frame generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Available {
    pub dlss: bool,
    pub framegen: bool,
}

fn streamline_dir() -> Result<PathBuf, String> {
    if let Some(dir) = std::env::var_os("VICTORY_STREAMLINE_DIR").filter(|dir| !dir.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    std::env::var_os("LOCALAPPDATA")
        .filter(|dir| !dir.is_empty())
        .map(|dir| PathBuf::from(dir).join("victory/nvidia-sdk/Streamline/bin/x64"))
        .ok_or_else(|| "set VICTORY_STREAMLINE_DIR to the folder containing sl.interposer.dll".into())
}
