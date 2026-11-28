use super::{Host, Prelude, host, lazy_prelude};
use std::sync::{Mutex, OnceLock};

const IDLE_LIMIT: usize = 128;

pub(super) struct TimedPrelude {
    prelude: Prelude,
    created: u64,
}

impl TimedPrelude {
    fn new(prelude: Prelude) -> Self {
        Self { prelude, created: super::now_ms() }
    }

    fn record<T>(&mut self, record: impl FnOnce(&Prelude) -> T) -> T {
        self.record_at(super::allocator_trim_seconds(), super::now_ms(), record)
    }

    fn record_at<T>(&mut self, seconds: Option<u64>, now: u64, record: impl FnOnce(&Prelude) -> T) -> T {
        if let Some(seconds) = seconds {
            let mut slot = (self.prelude.0.clone(), self.created);
            if super::refresh_allocator_at(&mut slot, now, seconds) {
                self.prelude.0 = slot.0;
            }
            self.created = slot.1;
        }
        record(&self.prelude)
    }
}

struct IdlePrelude(TimedPrelude);

// SAFETY: D3D12 objects are free-threaded. No list owns an idle prelude.
unsafe impl Send for IdlePrelude {}

pub(super) struct Pool(Mutex<Vec<IdlePrelude>>);

pub(super) static IDLE: Pool = Pool(Mutex::new(Vec::new()));

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_PRELUDE_POOL").is_some_and(|v| v == "1"))
}

impl Pool {
    fn take(&self) -> Option<TimedPrelude> {
        let idle = self.0.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(IdlePrelude(prelude)) = idle {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| tracing::info!("reused a state prelude (VICTORY_PRELUDE_POOL)"));
            Some(prelude)
        } else {
            host().and_then(Host::make_prelude).map(TimedPrelude::new)
        }
    }

    fn give(&self, prelude: TimedPrelude) {
        // Retaining the bounded set avoids recreating command memory on
        // every batch while host RAM pressure persists.
        let mut idle = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if idle.len() < IDLE_LIMIT {
            idle.push(IdlePrelude(prelude));
        }
    }
}

pub(super) enum Slot {
    Owned(OnceLock<Option<Mutex<TimedPrelude>>>),
    Pooled(Mutex<Option<TimedPrelude>>),
}

impl Slot {
    pub(super) fn new(owner: &Host) -> Self {
        let lazy = lazy_prelude();
        if lazy && enabled() {
            return Self::Pooled(Mutex::new(None));
        }
        let owned = OnceLock::new();
        if !lazy {
            let _ = owned.set(owner.make_prelude().map(TimedPrelude::new).map(Mutex::new));
        }
        Self::Owned(owned)
    }

    pub(super) fn record<T>(&self, pool: &Pool, record: impl FnOnce(&Prelude) -> T) -> Option<T> {
        match self {
            Self::Owned(owned) => owned
                .get_or_init(|| host().and_then(Host::make_prelude).map(TimedPrelude::new).map(Mutex::new))
                .as_ref()
                .map(|prelude| prelude.lock().unwrap_or_else(|e| e.into_inner()).record(record)),
            Self::Pooled(held) => {
                let mut held = held.lock().unwrap_or_else(|e| e.into_inner());
                if held.is_none() {
                    *held = pool.take();
                }
                held.as_mut().map(|prelude| prelude.record(record))
            }
        }
    }

    /// Called after the list's previous submission has finished and it has reset.
    pub(super) fn recycle(&self, pool: &Pool) {
        if let Self::Pooled(held) = self {
            let prelude = held.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(prelude) = prelude {
                pool.give(prelude);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Interface;

    #[test]
    fn a_prelude_records_with_its_replaced_allocator() {
        let Some(host) = host() else { return };
        let Some(prelude) = host.make_prelude() else { return };
        let mut prelude = TimedPrelude { prelude, created: 100 };
        let old = prelude.prelude.0.as_raw();
        prelude.record_at(Some(30), 30_100, |(allocator, list)| {
            assert_ne!(allocator.as_raw(), old);
            // SAFETY: the prelude is closed and has never been submitted.
            unsafe {
                allocator.Reset().unwrap();
                list.Reset(allocator, None).unwrap();
                list.Close().unwrap();
            }
        });
        assert_eq!(prelude.created, 30_100);
    }

    #[test]
    fn a_held_prelude_becomes_available_only_after_recycling() {
        if host().is_none() {
            return;
        }
        let pool = Pool(Mutex::new(Vec::new()));
        let first = Slot::Pooled(Mutex::new(None));
        let second = Slot::Pooled(Mutex::new(None));
        let allocator = |(allocator, _): &Prelude| allocator.as_raw() as usize;
        let held = first.record(&pool, allocator).unwrap();
        assert_eq!(first.record(&pool, allocator), Some(held));
        let other = second.record(&pool, allocator).unwrap();
        assert_ne!(other, held);
        first.recycle(&pool);
        let third = Slot::Pooled(Mutex::new(None));
        assert_eq!(third.record(&pool, allocator), Some(held));
    }
}
