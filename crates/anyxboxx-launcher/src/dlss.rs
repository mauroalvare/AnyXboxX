//! Saved Streamline options and their VIctory environment.

use crate::env::Pair;
use serde::{Deserialize, Serialize};

pub const SR: &str = "VICTORY_DLSS";
pub const FG: &str = "VICTORY_FRAMEGEN";
pub const DIR: &str = "VICTORY_STREAMLINE_DIR";
pub const MODES: &[(&str, &str)] = &[
    ("dlaa", "DLAA (native)"),
    ("quality", "Quality"),
    ("balanced", "Balanced"),
    ("performance", "Performance"),
    ("ultraperformance", "Ultra Performance"),
];
pub const PRESETS: &[&str] = &["default", "j", "k", "l", "m"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DlssOptions {
    pub super_resolution: bool,
    pub mode: String,
    pub preset: String,
    pub framegen: bool,
    pub hudless: bool,
    pub streamline_dir: String,
}

impl Default for DlssOptions {
    fn default() -> Self {
        Self {
            super_resolution: false,
            mode: "quality".into(),
            preset: "default".into(),
            framegen: false,
            hudless: true,
            streamline_dir: String::new(),
        }
    }
}

impl DlssOptions {
    /// Advanced switches also select the common FidelityFX path. The caller
    /// applies all advanced pairs last, including any explicit upscaler.
    pub fn env_pairs(&self, fg_available: bool, advanced: &[Pair]) -> Vec<Pair> {
        let switch = |key, fallback| {
            advanced
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map_or(fallback, |(_, v)| v == "1")
        };
        let sr = switch(SR, self.super_resolution);
        let fg = switch(FG, self.framegen && fg_available);
        let mut pairs = vec![
            (SR.into(), u8::from(sr).to_string()),
            (FG.into(), u8::from(fg).to_string()),
        ];
        if sr || fg {
            pairs.push(("VICTORY_UPSCALER".into(), "7".into()));
            pairs.push((
                DIR.into(),
                self.streamline_dir.trim().trim_matches('"').into(),
            ));
        }
        if sr {
            pairs.push(("VICTORY_DLSS_MODE".into(), self.mode.clone()));
            pairs.push(("VICTORY_DLSS_PRESET".into(), self.preset.clone()));
        }
        if fg {
            pairs.push(("VICTORY_THIN".into(), "1".into()));
            pairs.push((
                "VICTORY_FRAMEGEN_HUDLESS".into(),
                u8::from(self.hudless).to_string(),
            ));
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn switches_select_the_fidelityfx_path_and_their_own_settings() {
        for sr in [false, true] {
            for fg in [false, true] {
                for &(mode, _) in MODES {
                    let opts = DlssOptions {
                        super_resolution: sr,
                        framegen: fg,
                        mode: mode.into(),
                        preset: "k".into(),
                        hudless: false,
                        streamline_dir: r"C:\NVIDIA SDK\bin\x64".into(),
                    };
                    let env: HashMap<_, _> = opts.env_pairs(true, &[]).into_iter().collect();
                    let get = |key: &str| env.get(key).map(String::as_str);
                    assert_eq!(get(SR), Some(if sr { "1" } else { "0" }));
                    assert_eq!(get(FG), Some(if fg { "1" } else { "0" }));
                    assert_eq!(get("VICTORY_DLSS_MODE"), sr.then_some(mode));
                    assert_eq!(get("VICTORY_DLSS_PRESET"), sr.then_some("k"));
                    assert_eq!(get("VICTORY_FRAMEGEN_HUDLESS"), fg.then_some("0"));
                    assert_eq!(get("VICTORY_THIN"), fg.then_some("1"));
                    assert_eq!(get("VICTORY_UPSCALER"), (sr || fg).then_some("7"));
                    assert_eq!(get(DIR), (sr || fg).then_some(opts.streamline_dir.as_str()));
                }
            }
        }
    }

    #[test]
    fn a_saved_framegen_preference_is_inactive_on_an_ineligible_host() {
        let options = DlssOptions {
            framegen: true,
            ..Default::default()
        };
        let pairs = options.env_pairs(false, &[]);
        assert!(pairs.contains(&(FG.into(), "0".into())));
        assert!(!pairs.iter().any(|(k, _)| k == "VICTORY_UPSCALER"));
    }
}
