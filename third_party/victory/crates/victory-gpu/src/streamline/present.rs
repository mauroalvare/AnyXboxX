use super::*;
use crate::frame_rate::FrameGeneration;

/// The planes a present shows (Series S: the scene and the interface over
/// it), the host states they are in, and the back buffer's size.
pub(crate) struct Planes<'a> {
    pub base: (&'a ID3D12Resource, u32),
    pub overlay: Option<(&'a ID3D12Resource, u32)>,
    pub size: [u32; 2],
}

/// What [`present_begins`] began, for [`present_ends`].
pub(crate) struct Presenting {
    token: *mut abi::FrameToken,
    frame: u64,
    // VALID_UNTIL_PRESENT. The title may already have released its handles.
    _inputs: Option<Noted>,
    hudless: Option<ID3D12Resource>,
    tags_accepted: bool,
    separate_plane: bool,
}

/// Before our swap chain's Present of title frame `frame`: with frame
/// generation, the frame's latency markers, constants and tags, and whether
/// frame generation runs on it (it does when an upscaling was submitted
/// since the last present).
pub(crate) fn present_begins(frame: u64, planes: Planes<'_>) -> Option<Presenting> {
    if !FG_READY.load(Acquire) || !SWAP_CHAIN_PROXIED.load(Acquire) {
        return None;
    }
    // The tagged textures' states are those the submissions before this
    // present left: only the line of submissions orders the two.
    if !CHECKING.load(Relaxed) && !crate::queue::line_on() {
        HUDLESS_WANTED.store(false, Relaxed);
        lock(&SUBMITTED).take();
        resizing();
        once!(
            warn,
            "DLSS Frame Generation: the line of submissions did not start; frame generation stays off"
        );
        return None;
    }
    // Where the interface is no plane of its own (Series X): the copy of
    // this texture taken in front of the title's interface composite.
    let hudless = if hudless_wanted() {
        // Consume copies on every present, including a transition to two planes.
        hudless_for(planes.base.0).filter(|_| planes.overlay.is_none())
    } else {
        None
    };
    let (sl, functions, plan) = (SL.get()?, FUNCTIONS.get()?, PLAN.get()?);
    let (set_options, marker) = (functions.dlssg_set_options?, functions.pcl_set_marker?);
    let began = std::time::Instant::now();
    let markers_ok = latency::submitted(frame);
    if !markers_ok {
        once!(warn, frame, "Streamline: present without a complete simulation and render submission; skipping frame generation");
    }
    let mut calls = calls();
    let Some(token) = frame_token(sl, frame as u32) else {
        drop(calls);
        lock(&SUBMITTED).take();
        resizing();
        return None;
    };
    let fresh = lock(&SUBMITTED).take();
    let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
    let on = markers_ok
        && fresh.as_ref().is_some_and(|inputs| {
            tag_inputs(
                sl,
                &plan.asked,
                token,
                &viewport,
                inputs,
                &planes,
                hudless.as_ref(),
            )
        });
    if !on && calls.framegen_on {
        let tags = [
            abi::BUFFER_DEPTH,
            abi::BUFFER_MOTION_VECTORS,
            abi::BUFFER_HUDLESS_COLOR,
            abi::BUFFER_UI_COLOR_AND_ALPHA,
        ]
        .map(|kind| {
            abi::ResourceTag::new(
                std::ptr::null_mut(),
                kind,
                abi::VALID_UNTIL_PRESENT,
                abi::Extent::default(),
            )
        });
        // SAFETY: null resources explicitly clear the previous frame's tags.
        let code = unsafe {
            (sl.set_tag_for_frame)(
                token,
                &viewport,
                tags.as_ptr(),
                tags.len() as u32,
                std::ptr::null_mut(),
            )
        };
        if code != 0 {
            once!(
                warn,
                code,
                "Streamline refused to clear the previous frame's tags"
            );
        }
    }
    let was_on = calls.framegen_on;
    super::options::set_mode(
        frame as u32,
        planes.size,
        &viewport,
        functions,
        set_options,
        &mut calls,
        super::options::framegen_times(),
        on,
    );
    if calls.framegen_on && (!was_on || !calls.fg_count_valid) {
        // GetState counts since its last read. Drain the interval before this
        // Present when FG starts or a previous present/report was unsuccessful.
        calls.fg_count_valid = functions.dlssg_get_state.is_some_and(|get_state| {
            let mut state = abi::DlssgState::default();
            // SAFETY: a live viewport and writable state, no estimate requested.
            unsafe { get_state(&viewport, &mut state, std::ptr::null()) == 0 }
        });
    }
    drop(calls);
    if !latency::present_marker(frame, latency::Phase::Present, marker, token) {
        resizing();
    }
    spent(T_BEFORE, began);
    Some(Presenting {
        token,
        frame,
        _inputs: fresh,
        hudless,
        tags_accepted: on,
        separate_plane: planes.overlay.is_some(),
    })
}

fn tag_inputs(
    sl: &Sl,
    asked: &Asked,
    token: *mut abi::FrameToken,
    viewport: &abi::ViewportHandle,
    inputs: &Noted,
    planes: &Planes<'_>,
    hudless: Option<&ID3D12Resource>,
) -> bool {
    // (Frame generation reads the camera's planes and matrices: the
    // nearer plane is the near one, whichever way the title names them.)
    let camera = inputs.camera.planes_in_order();
    let constants = camera.constants(asked.jitter, asked.mvec);
    // SAFETY: constants for this frame and viewport.
    let constants_code = unsafe { (sl.set_constants)(&constants, token, viewport) };
    if constants_code != 0 {
        once!(
            warn,
            "frame generation: slSetConstants: {} ({constants_code})",
            abi::result_name(constants_code)
        );
        return false;
    }
    let mut resources = vec![
        (inputs.depth.tagged(), abi::BUFFER_DEPTH, camera.render),
        (
            inputs.motion.tagged(),
            abi::BUFFER_MOTION_VECTORS,
            camera.motion_size(),
        ),
    ];
    // Two planes: the scene is the colour without the interface, the
    // interface plane its colour and alpha (premultiplied, as `compose`
    // blends it).
    if let Some((over, over_state)) = planes.overlay {
        resources.push((
            abi::Resource::texture(planes.base.0.as_raw(), planes.base.1),
            abi::BUFFER_HUDLESS_COLOR,
            planes.size,
        ));
        resources.push((
            abi::Resource::texture(over.as_raw(), over_state),
            abi::BUFFER_UI_COLOR_AND_ALPHA,
            planes.size,
        ));
    } else if let Some(copy) = hudless {
        // One plane with the interface drawn into it: the copy taken in
        // front of the interface composite (ours, always a copy's
        // destination; Streamline holds it until the tags of the frame
        // go).
        resources.push((
            abi::Resource::texture(copy.as_raw(), COPY_DEST),
            abi::BUFFER_HUDLESS_COLOR,
            planes.size,
        ));
    }
    let tags: Vec<abi::ResourceTag> = resources
        .iter_mut()
        .map(|(r, kind, size)| {
            abi::ResourceTag::new(
                r,
                *kind,
                abi::VALID_UNTIL_PRESENT,
                abi::Extent::sized(*size),
            )
        })
        .collect();
    // SAFETY: tags of live textures in the states they are in at the
    // present (the line of submissions sends the present after them).
    let code = unsafe {
        (sl.set_tag_for_frame)(
            token,
            viewport,
            tags.as_ptr(),
            tags.len() as u32,
            std::ptr::null_mut(),
        )
    };
    if code != 0 {
        once!(
            warn,
            "frame generation: slSetTagForFrame: {} ({code})",
            abi::result_name(code)
        );
        return false;
    }
    once!(
        info,
        upscale = inputs.id,
        render = ?camera.render,
        output = ?camera.output,
        back_buffer = ?planes.size,
        interface_plane = planes.overlay.is_some(),
        interface_free_copy = hudless.is_some(),
        depth_state = format_args!("{:#x}", resources[0].0.state),
        motion_state = format_args!("{:#x}", resources[1].0.state),
        near = constants.camera_near,
        far = constants.camera_far,
        fov = constants.camera_fov,
        depth_inverted = camera.depth_inverted(),
        jitter = ?constants.jitter_offset,
        mvec_scale = ?constants.mvec_scale,
        "DLSS Frame Generation: first frame with depth and motion vectors tagged"
    );
    true
}

/// After our swap chain's Present: the frame's last marker and what frame generation reports.
pub(crate) fn present_ends(presenting: Option<Presenting>, status: windows::core::HRESULT) -> FrameGeneration {
    let Some(ref presenting) = presenting else {
        return FrameGeneration::Off;
    };
    let (token, frame) = (presenting.token, presenting.frame);
    let Some(functions) = FUNCTIONS.get() else {
        return FrameGeneration::Off;
    };
    let (Some(marker), Some(get_state)) = (functions.pcl_set_marker, functions.dlssg_get_state)
    else {
        return FrameGeneration::Off;
    };
    let began = std::time::Instant::now();
    let marker_ok = latency::present_marker(frame, latency::Phase::Presented, marker, token);
    if status != windows::Win32::Foundation::S_OK {
        calls().fg_count_valid = false;
        if !marker_ok {
            resizing();
        }
        return FrameGeneration::Off;
    }
    let mut generation = FrameGeneration::Off;
    let mut calls = calls();
    if calls.framegen_on {
        FG_ON.fetch_add(1, Relaxed);
        let presents = FG_ON_ALL.fetch_add(1, Relaxed) + 1;
        if matches!(presents, 1 | 240) && !CHECKING.load(Relaxed) && let Some(host) = crate::host::host() {
            host.log_video_memory(if presents == 1 {
                "after the first present with frame generation on"
            } else {
                "after 240 presents with frame generation on"
            });
        }
        if !presenting.separate_plane && hudless_wanted() {
            let counter = if presenting.tags_accepted && presenting.hudless.is_some() {
                HUD_TAGGED_ALL.fetch_add(1, Relaxed);
                HUD_TAGGED
            } else {
                HUD_MISSING
            };
            HUD_COUNTS[counter].fetch_add(1, Relaxed);
        }
        let mut state = abi::DlssgState::default();
        let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
        let asked = std::time::Instant::now();
        // SAFETY: a structure to fill, no options (no estimate asked).
        let code = unsafe { get_state(&viewport, &mut state, std::ptr::null()) };
        spent(T_STATE, asked);
        // A failed read may leave its count in the next report. Use the
        // configured multiplier until both ends of the interval are known.
        generation = FrameGeneration::Unavailable;
        if code == 0 {
            if calls.fg_count_valid {
                generation = FrameGeneration::Reported(state.num_frames_actually_presented);
            }
            FG_SHOWN.fetch_add(u64::from(state.num_frames_actually_presented), Relaxed);
            FG_SHOWN_ALL.fetch_add(u64::from(state.num_frames_actually_presented), Relaxed);
            let before = FG_STATUS.swap(state.status, Relaxed);
            if state.status != before {
                tracing::warn!(
                    status = format_args!("{:#x}", state.status),
                    "DLSS Frame Generation status changed (0 ok, 1 output too small, 2 Reflex not seen, 4 HDR format, 8 constants invalid, 16 back buffer index not asked)"
                );
            }
        } else {
            once!(
                warn,
                code,
                result = abi::result_name(code),
                "Streamline could not report the number of frames shown"
            );
        }
        calls.fg_count_valid = code == 0;
    } else {
        FG_OFF.fetch_add(1, Relaxed);
    }
    drop(calls);
    // The setting above was in effect during Present. Disable it for the
    // following frame if Streamline rejected the final marker.
    if !marker_ok {
        resizing();
    }
    spent(T_AFTER, began);
    generation
}

/// Our swap chain's buffers are about to be made again: frame generation is
/// off until the next present with an upscaling (Streamline asks so).
pub(crate) fn resizing() -> bool {
    if !FG_READY.load(Acquire) || !SWAP_CHAIN_PROXIED.load(Acquire) {
        return true;
    }
    let Some(set_options) = FUNCTIONS.get().and_then(|f| f.dlssg_set_options) else {
        return false;
    };
    let mut calls = calls();
    if calls.framegen_on {
        let options = abi::DlssgOptions::new(abi::DLSSG_OFF);
        let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
        // SAFETY: filled options.
        let code = unsafe { set_options(&viewport, &options) };
        if code != 0 && code != 39 {
            tracing::warn!(
                code,
                "Streamline refused to disable frame generation before resize"
            );
            return false;
        }
        calls.framegen_on = false;
        calls.framegen_times = 0;
    }
    true
}
