use super::*;
use std::collections::VecDeque;

pub(super) fn enable_reflex(functions: &Functions) -> bool {
    let (Some(reflex), Some(_)) = (functions.reflex_set_options, functions.pcl_set_marker) else {
        tracing::warn!(
            "Streamline: sl.reflex and sl.pcl are both required; frame generation stays off"
        );
        return false;
    };
    let options = abi::ReflexOptions {
        base: abi::BaseStructure::new(abi::REFLEX_OPTIONS, 1),
        mode: abi::REFLEX_LOW_LATENCY,
        frame_limit_us: 0,
        use_markers_to_optimize: false,
        virtual_key: 0,
        id_thread: 0,
    };
    // SAFETY: filled options with the version of the loaded function.
    let code = unsafe { reflex(&options) };
    if code != 0 {
        tracing::warn!(
            code,
            result = abi::result_name(code),
            "Reflex refused low latency; frame generation stays off"
        );
        return false;
    }
    tracing::info!("Streamline: Reflex low latency on");
    true
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Simulation,
    RenderSubmit,
    Submitted,
    Present,
    Presented,
}

// An origin can be abandoned without ever reaching PresentX. Forget an old
// unfinished frame rather than attributing any of its markers to a newer one.
const MAX_PENDING: usize = 64;
static PENDING: Mutex<VecDeque<(u64, Phase)>> = Mutex::new(VecDeque::new());

pub(crate) fn mark(frame: impl FnOnce() -> u64, phase: Phase) -> bool {
    advance(tracking_frames(), &PENDING, frame, phase, |frame, kind| {
        let Some((sl, marker)) = SL.get().zip(FUNCTIONS.get().and_then(|f| f.pcl_set_marker))
        else {
            return false;
        };
        let _calls = calls();
        let Some(token) = frame_token(sl, frame as u32) else {
            return false;
        };
        latency_marker(marker, kind, token)
    })
}

pub(super) fn submitted(frame: u64) -> bool {
    lock(&PENDING).contains(&(frame, Phase::Submitted))
}

// Reuse the token used for this present's constants and tags on both
// sides of DXGI Present, even after newer origins have been issued.
pub(super) fn present_marker(
    frame: u64,
    phase: Phase,
    marker: abi::SlPclSetMarker,
    token: *mut abi::FrameToken,
) -> bool {
    advance(
        true,
        &PENDING,
        || frame,
        phase,
        |_, kind| {
            let _calls = calls();
            latency_marker(marker, kind, token)
        },
    )
}

fn latency_marker(marker: abi::SlPclSetMarker, kind: u32, token: *mut abi::FrameToken) -> bool {
    let began = std::time::Instant::now();
    // SAFETY: token belongs to the frame whose phase is advancing.
    let code = unsafe { marker(kind, token) };
    marker_spent(began);
    if code != 0 {
        once!(
            warn,
            kind,
            code,
            result = abi::result_name(code),
            "Streamline refused a Reflex latency marker"
        );
    }
    code == 0
}

fn advance(
    enabled: bool,
    pending: &Mutex<VecDeque<(u64, Phase)>>,
    frame: impl FnOnce() -> u64,
    phase: Phase,
    mut emit: impl FnMut(u64, u32) -> bool,
) -> bool {
    if !enabled {
        return false;
    }
    let frame = frame();
    let mut pending = pending.lock().unwrap_or_else(|e| e.into_inner());
    let index = pending.iter().position(|&(id, _)| id == frame);
    if phase == Phase::Simulation {
        if index.is_some() || !emit(frame, abi::SIMULATION_START) {
            return false;
        }
        if pending.len() == MAX_PENDING {
            pending.pop_front();
        }
        pending.push_back((frame, phase));
        return true;
    }
    let Some(index) = index else { return false };
    let current = pending[index].1;
    let markers: &[u32] = match (current, phase) {
        (Phase::Simulation, Phase::RenderSubmit) => {
            &[abi::SIMULATION_END, abi::RENDER_SUBMIT_START]
        }
        (Phase::RenderSubmit, Phase::Submitted) => &[abi::RENDER_SUBMIT_END],
        (Phase::Submitted, Phase::Present) => &[abi::PRESENT_START],
        (Phase::Present, Phase::Presented) => &[abi::PRESENT_END],
        // The title can submit several lists for a frame.
        (Phase::RenderSubmit, Phase::RenderSubmit) => return true,
        _ => return false,
    };
    let ok = markers.iter().all(|&kind| emit(frame, kind));
    if !ok || phase == Phase::Presented {
        pending.remove(index);
    } else {
        pending[index].1 = phase;
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_plugins_and_accepted_low_latency_options_are_required() {
        unsafe extern "C" fn reflex(options: *const abi::ReflexOptions) -> i32 {
            // SAFETY: enable_reflex supplies the options for this call.
            let options = unsafe { &*options };
            i32::from(
                options.mode != abi::REFLEX_LOW_LATENCY
                    || options.frame_limit_us != 0
                    || options.use_markers_to_optimize,
            )
        }
        unsafe extern "C" fn refuse(_: *const abi::ReflexOptions) -> i32 {
            1
        }
        unsafe extern "C" fn marker(_: u32, _: *const abi::FrameToken) -> i32 {
            0
        }
        let mut functions = Functions {
            pcl_set_marker: Some(marker),
            ..Functions::default()
        };
        assert!(!enable_reflex(&functions));
        functions.reflex_set_options = Some(refuse);
        assert!(!enable_reflex(&functions));
        functions.reflex_set_options = Some(reflex);
        assert!(enable_reflex(&functions));
        functions.pcl_set_marker = None;
        assert!(!enable_reflex(&functions));
    }

    #[test]
    fn queued_frames_keep_their_own_marker_sequence() {
        use Phase::*;
        let pending = Mutex::default();
        let mut log = Vec::new();
        for (frame, phase) in [
            (41, Simulation),
            (41, RenderSubmit),
            (41, RenderSubmit),
            (42, Simulation),
            (42, RenderSubmit),
            (41, Submitted),
            (41, Present),
            (41, Presented),
            (42, Submitted),
            (42, Present),
            (42, Presented),
            (45, Simulation),
            (45, RenderSubmit),
            (45, Submitted),
            (45, Present),
            (45, Presented),
        ] {
            assert!(advance(
                true,
                &pending,
                || frame,
                phase,
                |id, marker| {
                    log.push((id, marker));
                    true
                }
            ));
        }
        for frame in [41, 42, 45] {
            let markers: Vec<_> = log
                .iter()
                .filter_map(|&(id, marker)| (id == frame).then_some(marker))
                .collect();
            assert_eq!(
                markers,
                [
                    abi::SIMULATION_START,
                    abi::SIMULATION_END,
                    abi::RENDER_SUBMIT_START,
                    abi::RENDER_SUBMIT_END,
                    abi::PRESENT_START,
                    abi::PRESENT_END
                ]
            );
        }
        assert!(pending.lock().unwrap().is_empty());
    }

    #[test]
    fn disabled_markers_never_read_the_frame_lock_or_time_a_call() {
        let pending = Mutex::default();
        let _held = pending.lock().unwrap();
        for phase in [
            Phase::Simulation,
            Phase::RenderSubmit,
            Phase::Submitted,
            Phase::Present,
            Phase::Presented,
        ] {
            assert!(!advance(
                false,
                &pending,
                || panic!("read frame"),
                phase,
                |_, _| panic!("timed marker")
            ));
        }
    }

    #[test]
    fn a_failed_or_missing_phase_cannot_enable_generation() {
        let pending = Mutex::default();
        assert!(advance(
            true,
            &pending,
            || 7,
            Phase::Simulation,
            |_, _| true
        ));
        assert!(!advance(
            true,
            &pending,
            || 7,
            Phase::RenderSubmit,
            |_, m| m != abi::SIMULATION_END
        ));
        for frame in [7, 8] {
            for phase in [Phase::Submitted, Phase::Present, Phase::Presented] {
                assert!(!advance(
                    true,
                    &pending,
                    || frame,
                    phase,
                    |_, _| panic!("unexpected marker")
                ));
            }
        }
    }
}
