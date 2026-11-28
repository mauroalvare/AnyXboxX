use super::present::Presenting;
use super::*;

// Counters since the last frame-rate line: upscalings done by DLSS, failed,
// left to the title; presents with frame generation on and off, frames it
// reported shown; and in all.
pub(super) static EVALUATED: AtomicU64 = AtomicU64::new(0);
pub(super) static FAILED: AtomicU64 = AtomicU64::new(0);
pub(super) static FORWARDED: AtomicU64 = AtomicU64::new(0);
pub(super) static FG_ON: AtomicU64 = AtomicU64::new(0);
pub(super) static FG_OFF: AtomicU64 = AtomicU64::new(0);
pub(super) static FG_SHOWN: AtomicU64 = AtomicU64::new(0);
pub(super) static EVALUATED_ALL: AtomicU64 = AtomicU64::new(0);
pub(super) static FG_STATUS: AtomicU32 = AtomicU32::new(0);
/// Presents with frame generation on, and frames it showed, in all.
pub(super) static FG_ON_ALL: AtomicU64 = AtomicU64::new(0);
pub(super) static FG_SHOWN_ALL: AtomicU64 = AtomicU64::new(0);
// Since the last frame-rate line, microseconds except T_MARKERS: on the presenting thread
// before the present, in the present call
// (and the longest), after it (of them asking frame generation's state); on
// the threads that record upscalings, recording one (of them waiting for our
// lock); recording the copy of the colour without the interface.
static TIMES: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
pub(super) const T_BEFORE: usize = 0;
// Marker calls can take less than a microsecond. Accumulate nanoseconds
// across the title and presentation threads before converting for the log.
const T_MARKERS: usize = 1;
static MARKER_CALLS: AtomicU64 = AtomicU64::new(0);

pub(super) fn marker_spent(since: std::time::Instant) {
    TIMES[T_MARKERS].fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
    MARKER_CALLS.fetch_add(1, Relaxed);
}
const T_PRESENT: usize = 2;
const T_PRESENT_LONGEST: usize = 3;
pub(super) const T_AFTER: usize = 4;
pub(super) const T_STATE: usize = 5;
pub(super) const T_UPSCALE: usize = 6;
pub(super) const T_UPSCALE_LOCK: usize = 7;
pub(super) const T_COPY: usize = 8;

pub(super) fn spent(slot: usize, since: std::time::Instant) {
    TIMES[slot].fetch_add(since.elapsed().as_micros() as u64, Relaxed);
}

/// When the present call of a frame [`present_begins`] began begins: `None`
/// without frame generation (nothing is timed then).
pub(crate) fn present_called(presenting: &Option<Presenting>) -> Option<std::time::Instant> {
    presenting.is_some().then(std::time::Instant::now)
}

/// The present call that began at `called` returned.
pub(crate) fn present_returned(called: Option<std::time::Instant>) {
    if let Some(called) = called {
        let took = called.elapsed().as_micros() as u64;
        TIMES[T_PRESENT].fetch_add(took, Relaxed);
        TIMES[T_PRESENT_LONGEST].fetch_max(took, Relaxed);
    }
}
/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let Some(plan) = PLAN.get() else { return };
    let [done, failed, forwarded, on, off, shown] =
        [&EVALUATED, &FAILED, &FORWARDED, &FG_ON, &FG_OFF, &FG_SHOWN].map(|c| c.swap(0, Relaxed));
    if plan.dlss || forwarded > 0 {
        tracing::info!(
            "DLSS per frame: {:.2} upscalings by DLSS, {:.2} failed, {:.2} left to the title's FidelityFX ({} by DLSS so far)",
            done as f64 / frames,
            failed as f64 / frames,
            forwarded as f64 / frames,
            EVALUATED_ALL.load(Relaxed)
        );
    }
    if plan.framegen {
        tracing::info!(
            "DLSS Frame Generation: {on} presents with it on, {off} off, {shown} frames shown by it (status {:#x}){}",
            FG_STATUS.load(Relaxed),
            if FG_READY.load(Relaxed) && SWAP_CHAIN_PROXIED.load(Relaxed) { "" } else { "; not running (see above)" }
        );
        // Where the time of a frame goes in Streamline's calls.
        let t: Vec<u64> = TIMES.iter().map(|t| t.swap(0, Relaxed)).collect();
        let markers = MARKER_CALLS.swap(0, Relaxed);
        tracing::info!(
            "DLSS Frame Generation latency markers: {markers} calls, {:.3} ms per frame",
            t[T_MARKERS] as f64 / 1_000_000.0 / frames
        );
        if on + off > 0 {
            let ms = |slot: usize| t[slot] as f64 / 1000.0 / frames;
            tracing::info!(
                "DLSS Frame Generation, time per frame: the presenting thread {:.2} ms before the present call, {:.2} ms in it (the longest {:.1} ms), {:.2} ms after it; {:.2} ms asking frame generation's state; an upscaling recorded in {:.2} ms ({:.2} ms of it waiting for the lock it shares with the present); the copy of the colour without the interface recorded in {:.3} ms",
                ms(T_BEFORE),
                ms(T_PRESENT),
                t[T_PRESENT_LONGEST] as f64 / 1000.0,
                ms(T_AFTER),
                ms(T_STATE),
                ms(T_UPSCALE),
                ms(T_UPSCALE_LOCK),
                ms(T_COPY)
            );
        }
        // Every tenth line (about ten seconds): the colour without the
        // interface, where the title draws the interface into the frame.
        static LINES: AtomicU32 = AtomicU32::new(0);
        if hudless_wanted() && LINES.fetch_add(1, Relaxed) % 10 == 9 {
            let [copies, tagged, missing, composites, again, others] = [0, 1, 2, 3, 4, 5].map(|i| HUD_COUNTS[i].swap(0, Relaxed));
            tracing::info!(
                "DLSS Frame Generation, colour without the interface, last ten lines: {copies} copies in front of the title's interface composite, {tagged} presents tagged with one, {missing} presents with frame generation on had none; {composites} interface composite draws into a presented texture ({again} after a copy in their list), {others} draws of other pipelines into one"
            );
        }
    }
}
