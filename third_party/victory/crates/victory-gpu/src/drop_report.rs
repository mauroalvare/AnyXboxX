//! Optional interval census of dropped GPU operations.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
const MAX_REASONS: usize = 256;
#[derive(Default)]
struct Counts(BTreeMap<String, u64>);
impl Counts {
    fn add(&mut self, key: String) {
        let key = if self.0.contains_key(&key) || self.0.len() < MAX_REASONS { key } else { "other (census full)".to_owned() };
        *self.0.entry(key).or_default() += 1;
    }
    fn take(&mut self) -> String {
        std::mem::take(&mut self.0).into_iter().map(|(key, n)| format!("{n} [{key}]")).collect::<Vec<_>>().join("; ")
    }
}
fn interval() -> Option<Duration> {
    static INTERVAL: OnceLock<Option<Duration>> = OnceLock::new();
    *INTERVAL.get_or_init(|| std::env::var("VICTORY_DROP_REPORT_SECONDS").ok().and_then(|s| s.parse::<u64>().ok()).filter(|&n| n > 0).map(Duration::from_secs))
}
pub(crate) fn enabled() -> bool { interval().is_some() }
pub(crate) fn key(operation: &str, reason: &str, status: &str, hashes: &str) -> String {
    format!("{operation}: {reason}; pipeline={status}; shaders={hashes}")
}
pub(crate) fn record(key: String) {
    let Some(interval) = interval() else { return };
    static STATE: OnceLock<Mutex<(Instant, Counts)>> = OnceLock::new();
    let mut state = STATE.get_or_init(|| Mutex::new((Instant::now(), Counts::default()))).lock().unwrap_or_else(|e| e.into_inner());
    state.1.add(key);
    if state.0.elapsed() >= interval {
        let line = state.1.take();
        state.0 = Instant::now();
        drop(state);
        tracing::info!("dropped GPU operations: {line}");
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_operation_and_pipeline_causes() {
        assert_eq!(key("mesh indirect", "unsupported", "being made", "abc"),
            "mesh indirect: unsupported; pipeline=being made; shaders=abc");
    }
    #[test]
    fn aggregates_and_resets_by_reason_and_shader() {
        let mut counts = Counts::default();
        counts.add("failed; shaders=123".into());
        counts.add("failed; shaders=123".into());
        counts.add("failed; shaders=456".into());
        assert_eq!(counts.take(), "2 [failed; shaders=123]; 1 [failed; shaders=456]");
        assert!(counts.take().is_empty());
    }
    #[test]
    fn limits_cardinality() {
        let mut counts = Counts::default();
        for i in 0..MAX_REASONS + 3 { counts.add(i.to_string()); }
        assert_eq!(counts.0.get("other (census full)"), Some(&3));
    }
}
