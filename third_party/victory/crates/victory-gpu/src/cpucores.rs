//! Per-logical-CPU utilization, from `NtQuerySystemInformation(SystemProcessorPerformanceInformation)`.
//! Shared by the `VICTORY_PANEL` HUD ([`crate::host`]) and the performance overlay
//! ([`crate::overlay`]).

use std::sync::OnceLock;

/// `SystemProcessorPerformanceInformation`.
const SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION: u32 = 8;
/// The most logical processors read (one group).
const MOST_CPUS: usize = 64;

/// Per-logical-CPU utilization tracker: each [`CpuMeter::sample`] gives the load since the one before.
#[derive(Default)]
pub(crate) struct CpuMeter {
    prev: Vec<(i64, i64)>,
    /// The mean of the cores' load, in percent (0 until two samples).
    pub(crate) total_pct: f32,
    /// Each logical CPU's load, in percent (empty until two samples).
    pub(crate) cores_pct: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SysProcPerfInfo {
    idle: i64,
    kernel: i64,
    user: i64,
    dpc: i64,
    interrupt: i64,
    interrupt_count: u32,
    _pad: u32,
}

/// Each CPU's busy percent between two readings of `(idle, kernel + user)` times (kernel time
/// includes idle). Empty when the CPU count changed or there was no reading before.
pub(crate) fn busy_percents(prev: &[(i64, i64)], now: &[(i64, i64)]) -> Vec<f32> {
    if prev.is_empty() || prev.len() != now.len() {
        return Vec::new();
    }
    prev.iter()
        .zip(now)
        .map(|(&(pi, pt), &(ni, nt))| {
            let dt = (nt - pt).max(1);
            let di = (ni - pi).clamp(0, dt);
            ((1.0 - di as f32 / dt as f32) * 100.0).clamp(0.0, 100.0)
        })
        .collect()
}

/// How many cores are busier than `limit` percent.
pub(crate) fn busier_than(cores: &[f32], limit: f32) -> usize {
    cores.iter().filter(|&&p| p > limit).count()
}

type NtQuery = unsafe extern "system" fn(u32, *mut u8, u32, *mut u32) -> i32;

fn nt_query() -> Option<NtQuery> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    static FN: OnceLock<Option<NtQuery>> = OnceLock::new();
    *FN.get_or_init(|| unsafe {
        let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
        if ntdll.is_null() {
            return None;
        }
        let p = GetProcAddress(ntdll, c"NtQuerySystemInformation".as_ptr().cast())?;
        Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, NtQuery>(p))
    })
}

/// The `(idle, busy)` times of each logical CPU now; `None` if the system call fails.
fn read_times() -> Option<Vec<(i64, i64)>> {
    let query = nt_query()?;
    let n = std::thread::available_parallelism().map_or(8, |n| n.get()).min(MOST_CPUS);
    let mut buf = vec![SysProcPerfInfo::default(); n];
    let mut ret = 0u32;
    // SAFETY: a buffer of the size passed; `ret` is the bytes written.
    let status = unsafe {
        query(
            SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION,
            buf.as_mut_ptr().cast(),
            (buf.len() * size_of::<SysProcPerfInfo>()) as u32,
            &mut ret,
        )
    };
    if status < 0 {
        return None;
    }
    let count = (ret as usize / size_of::<SysProcPerfInfo>()).min(buf.len());
    (count > 0).then(|| buf[..count].iter().map(|p| (p.idle, p.kernel + p.user)).collect())
}

impl CpuMeter {
    pub(crate) fn sample(&mut self) {
        let Some(now) = read_times() else { return };
        let cores = busy_percents(&self.prev, &now);
        if !cores.is_empty() {
            self.total_pct = cores.iter().sum::<f32>() / cores.len() as f32;
            self.cores_pct = cores;
        }
        self.prev = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_idle_and_fully_busy_cores() {
        // Core 0: all 100 ticks idle. Core 1: none idle.
        let got = busy_percents(&[(0, 0), (0, 0)], &[(100, 100), (0, 100)]);
        assert_eq!(got, vec![0.0, 100.0]);
    }

    #[test]
    fn half_busy_core() {
        let got = busy_percents(&[(10, 20)], &[(60, 120)]);
        assert!((got[0] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn idle_ahead_of_total_is_clamped_to_zero_load() {
        let got = busy_percents(&[(0, 0)], &[(500, 100)]);
        assert_eq!(got, vec![0.0]);
    }

    #[test]
    fn idle_going_backwards_counts_as_all_busy() {
        let got = busy_percents(&[(100, 0)], &[(50, 100)]);
        assert_eq!(got, vec![100.0]);
    }

    #[test]
    fn no_elapsed_time_does_not_divide_by_zero() {
        let got = busy_percents(&[(5, 5)], &[(5, 5)]);
        assert_eq!(got.len(), 1);
        assert!(got[0].is_finite());
    }

    #[test]
    fn no_previous_reading_or_changed_cpu_count_gives_nothing() {
        assert!(busy_percents(&[], &[(1, 1)]).is_empty());
        assert!(busy_percents(&[(0, 0)], &[(1, 1), (2, 2)]).is_empty());
    }

    #[test]
    fn counts_cores_over_a_limit() {
        assert_eq!(busier_than(&[10.0, 50.0, 50.1, 99.0], 50.0), 2);
        assert_eq!(busier_than(&[], 50.0), 0);
    }
}
