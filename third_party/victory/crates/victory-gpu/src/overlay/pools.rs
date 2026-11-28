//! What the overlay shows about our worker pools: the figures each pool's frame-rate line stores
//! (through [`set_pool`]) and the pure formatting of them. A pool that is off says "off", one with
//! no sample yet (or none with work in it) says "...": nothing is ever made up.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::cpucores::busier_than;

/// Our worker pools.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pool {
    /// List translation (`VICTORY_THIN`).
    Thin,
    /// Landing (`VICTORY_LAND`).
    Land,
    /// The submission thread (`VICTORY_ASYNC_SUBMIT`).
    Submit,
    /// Pipeline workers (`VICTORY_PIPELINE_WORKERS`).
    Pipelines,
}

pub(crate) const POOLS: [Pool; 4] = [Pool::Thin, Pool::Land, Pool::Submit, Pool::Pipelines];

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const SAMPLED: u8 = 2;

/// One pool's last figures: work on all threads and wall time (ms, f64 bits), thread count.
struct Slot {
    state: AtomicU8,
    threads: AtomicU64,
    work: AtomicU64,
    wall: AtomicU64,
}

static SLOTS: [Slot; 4] = [const {
    Slot { state: AtomicU8::new(UNKNOWN), threads: AtomicU64::new(0), work: AtomicU64::new(0), wall: AtomicU64::new(0) }
}; 4];

fn slot(pool: Pool) -> &'static Slot {
    &SLOTS[pool as usize]
}

/// Stores a pool's last second: `work_ms` of work on all its threads (per frame) over `wall_ms`
/// (per frame), with `threads` threads. A `wall_ms` of 0 is a second with nothing to measure.
pub(crate) fn set_pool(pool: Pool, work_ms: f64, wall_ms: f64, threads: usize) {
    let s = slot(pool);
    s.threads.store(threads as u64, Ordering::Relaxed);
    s.work.store(work_ms.to_bits(), Ordering::Relaxed);
    s.wall.store(wall_ms.to_bits(), Ordering::Relaxed);
    s.state.store(SAMPLED, Ordering::Release);
}

/// The pool is not running (its switch is off).
pub(crate) fn set_pool_off(pool: Pool) {
    slot(pool).state.store(OFF, Ordering::Release);
}

/// The pool's size, known before it has a sample.
pub(crate) fn set_pool_threads(pool: Pool, threads: usize) {
    slot(pool).threads.store(threads as u64, Ordering::Relaxed);
}

/// What the overlay knows of a pool.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub(crate) enum PoolState {
    #[default]
    Unknown,
    Off,
    Sampled { work_ms: f64, wall_ms: f64 },
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub(crate) struct PoolFigures {
    pub(crate) state: PoolState,
    pub(crate) threads: usize,
}

pub(crate) fn read(pool: Pool) -> PoolFigures {
    let s = slot(pool);
    let state = match s.state.load(Ordering::Acquire) {
        OFF => PoolState::Off,
        SAMPLED => PoolState::Sampled {
            work_ms: f64::from_bits(s.work.load(Ordering::Relaxed)),
            wall_ms: f64::from_bits(s.wall.load(Ordering::Relaxed)),
        },
        _ => PoolState::Unknown,
    };
    PoolFigures { state, threads: s.threads.load(Ordering::Relaxed) as usize }
}

pub(crate) fn read_all() -> [PoolFigures; 4] {
    POOLS.map(read)
}

/// Work on all threads over wall time: how many threads were busy on average. `None` without
/// wall time or with a figure that is not a number.
pub(crate) fn parallelism(work_ms: f64, wall_ms: f64) -> Option<f64> {
    (wall_ms > 0.0 && work_ms >= 0.0 && work_ms.is_finite() && wall_ms.is_finite()).then(|| work_ms / wall_ms)
}

/// The same as a share of one thread's time, in percent.
pub(crate) fn busy_percent(work_ms: f64, wall_ms: f64) -> Option<f64> {
    parallelism(work_ms, wall_ms).map(|p| p * 100.0)
}

fn factor(f: PoolFigures) -> Option<f64> {
    match f.state {
        PoolState::Sampled { work_ms, wall_ms } => parallelism(work_ms, wall_ms),
        _ => None,
    }
}

fn percent(f: PoolFigures) -> Option<f64> {
    factor(f).map(|p| p * 100.0)
}

/// The row's value for a pool, e.g. `8 thr  x2.6`, `1 thr  97%`, `off`, `...`.
pub(crate) fn pool_value(pool: Pool, f: PoolFigures, making: u64) -> String {
    if pool == Pool::Pipelines {
        return if f.threads == 0 { "...".into() } else { format!("{} wk  {making} making", f.threads) };
    }
    match f.state {
        PoolState::Off => "off".into(),
        PoolState::Unknown => "...".into(),
        PoolState::Sampled { .. } => {
            let figure = match pool {
                Pool::Submit => percent(f).map_or("...".into(), |p| format!("{p:.0}%")),
                _ => factor(f).map_or("...".into(), |p| format!("x{p:.1}")),
            };
            format!("{} thr  {figure}", f.threads)
        }
    }
}

pub(crate) fn pool_label(pool: Pool) -> &'static str {
    match pool {
        Pool::Thin => "Lists",
        Pool::Land => "Landing",
        Pool::Submit => "Submit",
        Pool::Pipelines => "Pipelines",
    }
}

fn log_part(name: &str, pool: Pool, f: PoolFigures) -> String {
    match f.state {
        PoolState::Off => format!("{name} off"),
        PoolState::Unknown => format!("{name} ..."),
        PoolState::Sampled { .. } => match pool {
            Pool::Submit => format!("{name} {}", percent(f).map_or("...".into(), |p| format!("{p:.0}%"))),
            _ => format!("{name} {}/{}", factor(f).map_or("...".into(), |p| format!("x{p:.1}")), f.threads),
        },
    }
}

/// The overlay's 5-second log line part about threads.
pub(crate) fn threads_summary(pools: &[PoolFigures; 4], making: u64, cores: &[f32]) -> String {
    let pipelines = if pools[3].threads == 0 {
        "pipelines ...".to_owned()
    } else {
        format!("pipelines {} ({making} making)", pools[3].threads)
    };
    let busy = if cores.is_empty() { "...".into() } else { format!("{}/{}", busier_than(cores, 50.0), cores.len()) };
    format!(
        "{}, {}, {}, {pipelines}, cores busy>50%: {busy}",
        log_part("lists", Pool::Thin, pools[0]),
        log_part("land", Pool::Land, pools[1]),
        log_part("submit", Pool::Submit, pools[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampled(work_ms: f64, wall_ms: f64, threads: usize) -> PoolFigures {
        PoolFigures { state: PoolState::Sampled { work_ms, wall_ms }, threads }
    }

    #[test]
    fn parallelism_is_work_over_wall() {
        assert_eq!(parallelism(26.0, 10.0), Some(2.6));
        assert_eq!(parallelism(0.0, 10.0), Some(0.0));
    }

    #[test]
    fn no_wall_time_or_bad_numbers_give_no_figure() {
        assert_eq!(parallelism(5.0, 0.0), None);
        assert_eq!(parallelism(5.0, -1.0), None);
        assert_eq!(parallelism(f64::NAN, 10.0), None);
        assert_eq!(parallelism(5.0, f64::INFINITY), None);
        assert_eq!(busy_percent(5.0, 0.0), None);
    }

    #[test]
    fn busy_percent_is_a_share_of_the_wall_time() {
        let p = busy_percent(9.7, 10.0).unwrap();
        assert!((p - 97.0).abs() < 1e-9);
    }

    #[test]
    fn values_for_every_state() {
        assert_eq!(pool_value(Pool::Thin, sampled(26.0, 10.0, 8), 0), "8 thr  x2.6");
        assert_eq!(pool_value(Pool::Land, sampled(36.0, 10.0, 4), 0), "4 thr  x3.6");
        assert_eq!(pool_value(Pool::Submit, sampled(9.7, 10.0, 1), 0), "1 thr  97%");
        assert_eq!(pool_value(Pool::Thin, PoolFigures::default(), 0), "...");
        assert_eq!(pool_value(Pool::Land, PoolFigures { state: PoolState::Off, threads: 4 }, 0), "off");
    }

    #[test]
    fn a_sample_without_wall_time_shows_no_figure() {
        assert_eq!(pool_value(Pool::Thin, sampled(0.0, 0.0, 8), 0), "8 thr  ...");
        assert_eq!(pool_value(Pool::Submit, sampled(0.0, 0.0, 1), 0), "1 thr  ...");
    }

    #[test]
    fn pipelines_show_workers_and_pipelines_being_made() {
        let f = PoolFigures { state: PoolState::Unknown, threads: 12 };
        assert_eq!(pool_value(Pool::Pipelines, f, 3), "12 wk  3 making");
        assert_eq!(pool_value(Pool::Pipelines, PoolFigures::default(), 3), "...");
    }

    #[test]
    fn log_summary_names_every_pool() {
        let pools = [
            sampled(26.0, 10.0, 8),
            sampled(36.0, 10.0, 4),
            sampled(9.7, 10.0, 1),
            PoolFigures { state: PoolState::Unknown, threads: 10 },
        ];
        let cores = [60.0, 10.0, 70.0];
        assert_eq!(
            threads_summary(&pools, 3, &cores),
            "lists x2.6/8, land x3.6/4, submit 97%, pipelines 10 (3 making), cores busy>50%: 2/3"
        );
    }

    #[test]
    fn log_summary_without_samples() {
        let mut pools = [PoolFigures::default(); 4];
        pools[1].state = PoolState::Off;
        assert_eq!(
            threads_summary(&pools, 0, &[]),
            "lists ..., land off, submit ..., pipelines ..., cores busy>50%: ..."
        );
    }

    #[test]
    fn slots_store_and_read_back() {
        set_pool(Pool::Land, 7.2, 2.0, 4);
        assert_eq!(read(Pool::Land), sampled(7.2, 2.0, 4));
        set_pool_off(Pool::Land);
        assert_eq!(read(Pool::Land).state, PoolState::Off);
    }
}
