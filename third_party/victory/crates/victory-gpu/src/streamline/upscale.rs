use super::*;

/// One upscaling the title asked for: a FidelityFX `ffxDispatch` of an
/// upscale, in its own terms (`victory_hle::modules::fidelityfx` reads it).
#[derive(Clone, Copy, Debug)]
pub struct Upscale {
    /// The FidelityFX context.
    pub context: usize,
    /// The title's command list.
    pub list: *const Object,
    /// Colour, depth, motion vectors, exposure, reactive mask, transparency
    /// and composition mask, output: the title's resource (null if none) and
    /// the FidelityFX state it is in.
    pub resources: [(*const Object, u32); 7],
    pub camera: plan::Camera,
    pub pre_exposure: f32,
    pub frame_ms: f32,
}

// SAFETY: pointers to title objects that live until the list has run (the
// title's part, as on the console); nothing else is shared.
unsafe impl Send for Upscale {}

pub const COLOUR: usize = 0;
pub const DEPTH: usize = 1;
pub const MOTION: usize = 2;
pub const EXPOSURE: usize = 3;
pub const REACTIVE: usize = 4;
pub const OUTPUT: usize = 6;

/// What became of an upscaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Taken {
    /// DLSS does it: the title's own must not run.
    Dlss,
    /// Only noted for frame generation: the title's own runs.
    Noted,
    /// Neither: the title's own runs. Why, for the log.
    No(&'static str),
}
/// Whether the title's FidelityFX calls are to be served (DLSS or frame
/// generation is on): `victory_hle` then puts its own in front of them.
pub fn serving() -> bool {
    DLSS_READY.load(Acquire) || FG_READY.load(Acquire)
}

/// Whether DLSS is to take the title's upscale dispatches (`VICTORY_DLSS=1`
/// and Streamline loaded).
pub fn upscaling() -> bool {
    DLSS_READY.load(Acquire)
}

/// The upscaling `up` asked for on the title's thread: DLSS records it at
/// this place of the title's list instead (or, with frame generation alone,
/// its inputs are noted there for the present); the title's own runs
/// otherwise.
pub fn upscale(up: &Upscale) -> Taken {
    let Some(plan) = PLAN.get() else {
        return Taken::No("Streamline is not loaded");
    };
    let dlss = plan.dlss && DLSS_READY.load(Acquire);
    let framegen = plan.framegen && FG_READY.load(Acquire);
    if !dlss && !framegen {
        return Taken::No("Streamline has no device yet, or its features did not start");
    }
    if [COLOUR, DEPTH, MOTION, OUTPUT].iter().any(|&i| up.resources[i].0.is_null()) {
        return Taken::No("a required texture (colour, depth, motion vectors, output) is missing");
    }
    if up.camera.render.contains(&0) || up.camera.output.contains(&0) {
        return Taken::No("an empty render or output size");
    }
    if framegen && !dlss && ![DEPTH, MOTION].iter().all(|&i| crate::resource::is_resource(up.resources[i].0)) {
        return Taken::No("invalid depth or motion vectors for frame generation");
    }
    if dlss && let Err(why) = validate(up) {
        return Taken::No(why);
    }
    let id = NEXT_ID.fetch_add(1, Relaxed) + 1;
    // Capture the title's frame before thin-list playback can lag behind.
    // id still identifies individual submissions for depth/motion freshness.
    let frame = if framegen { crate::frame::current() } else { id };
    let up = *up;
    let asked = crate::list::external(
        up.list,
        Box::new(move |host| {
            if dlss && !evaluate(host, &up, frame) {
                return false;
            }
            if framegen {
                return note(host, &up, id) || dlss;
            }
            true
        }),
    );
    if !asked {
        return Taken::No("the upscale could not be recorded on the command list");
    }
    if dlss { Taken::Dlss } else { Taken::Noted }
}

/// The title's own upscaling ran for this reason.
pub fn forwarded() {
    FORWARDED.fetch_add(1, Relaxed);
}

/// What an upscaling's textures must be for DLSS, checked on the title's
/// thread (the title's own runs if not): host textures, in states they can
/// be in.
fn validate(up: &Upscale) -> Result<(), &'static str> {
    for (i, &(resource, state)) in up.resources.iter().enumerate() {
        if resource.is_null() || i == 5 {
            continue;
        }
        if !crate::resource::is_resource(resource) {
            return Err("a texture of the upscaling is no resource of ours");
        }
        let Some(copy) = crate::resource::host_copy(resource) else {
            return Err("a texture of the upscaling has no host copy (a buffer?)");
        };
        let wanted = if i == OUTPUT { plan::output_state(state) } else { plan::input_state(state) };
        if crate::host::legal_state(&copy.texture, wanted) != wanted {
            return Err(if i == OUTPUT {
                "the output cannot be written by a shader (no unordered access)"
            } else {
                "an input cannot be read by a shader"
            });
        }
    }
    Ok(())
}

/// DLSS in place of the title's upscaling, recorded into `host` (where the
/// title's list is now). Whether it recorded anything.
fn evaluate(host: &crate::host::HostList, up: &Upscale, id: u64) -> bool {
    let Some(plan) = PLAN.get() else { return false };
    let camera = up.camera;
    let fail = |why: &str| {
        FAILED.fetch_add(1, Relaxed);
        tracing::warn!(id, "DLSS: the upscaling was not recorded: {why} (the output keeps what it had)");
        false
    };
    // The inputs as a compute shader's would be (aliased views, uploads of
    // title-written textures, depth twins), then in the states DLSS is told.
    let used = |i: usize| {
        !up.resources[i].0.is_null()
            && (i != REACTIVE || plan.asked.reactive)
            && (i != EXPOSURE || camera.flags & plan::FFX_AUTO_EXPOSURE == 0)
    };
    let mut textures: Textures = Default::default();
    for i in [COLOUR, DEPTH, MOTION, EXPOSURE, REACTIVE, OUTPUT] {
        if !used(i) {
            continue;
        }
        let (resource, state) = up.resources[i];
        if i != OUTPUT && crate::external::prepare_input(host, resource).is_none() {
            return fail("an input could not be prepared");
        }
        let wanted = if i == OUTPUT { plan::output_state(state) } else { plan::input_state(state) };
        let Some((texture, legal)) = crate::external::host_texture(host, resource, wanted) else {
            return fail("a texture has no host copy");
        };
        if legal != wanted {
            return fail("a texture cannot be in the state DLSS needs");
        }
        textures[i] = Some((texture, legal));
    }
    let recorded = match record(host.list.as_raw(), &textures, &camera, up.pre_exposure, id) {
        Ok(recorded) => recorded,
        Err(why) => return fail(&why),
    };
    crate::external::output_written(up.resources[OUTPUT].0);
    EVALUATED.fetch_add(1, Relaxed);
    let all = EVALUATED_ALL.fetch_add(1, Relaxed) + 1;
    if all == 1 || recorded.changed {
        tracing::info!(
            render = ?camera.render,
            output = ?camera.output,
            mode = plan::sl_mode_name(recorded.mode),
            asked = plan.asked.mode.name(),
            preset = plan.asked.preset,
            optimal = ?recorded.optimal,
            hdr = camera.flags & plan::FFX_HIGH_DYNAMIC_RANGE != 0,
            depth_inverted = camera.depth_inverted(),
            auto_exposure = textures[EXPOSURE].is_none(),
            reactive = textures[REACTIVE].is_some(),
            display_motion_vectors = camera.display_motion_vectors(),
            jitter = ?recorded.jitter,
            mvec_scale = ?recorded.mvec_scale,
            near = camera.near,
            far = camera.far,
            fov = camera.fov_y,
            formats = ?formats(&textures),
            evaluations = all,
            "DLSS: upscaling recorded in place of the title's FidelityFX"
        );
    }
    true
}

/// Colour, depth, motion vectors, exposure, reactive mask, (transparency),
/// output: the host textures DLSS is given and the states they are in.
pub(super) type Textures = [Option<(ID3D12Resource, u32)>; 7];

fn formats(textures: &Textures) -> Vec<i32> {
    textures
        .iter()
        .flatten()
        .map(|(t, _)| {
            // SAFETY: a live texture.
            unsafe { t.GetDesc() }.Format.0
        })
        .collect()
}

/// What [`record`] recorded.
pub(super) struct Recorded {
    pub(super) mode: u32,
    pub(super) optimal: Option<[u32; 2]>,
    /// Another mode or size than the last time.
    changed: bool,
    jitter: [f32; 2],
    mvec_scale: [f32; 2],
}

/// The DLSS evaluation of `textures` recorded into the open command list
/// `list`: the mode whose render size holds the title's, the options, the
/// frame's constants (frame `id` of the upscaling's viewport), the textures
/// as local tags. The textures are left in the states given.
pub(super) fn record(
    list: *mut c_void,
    textures: &Textures,
    camera: &plan::Camera,
    pre_exposure: f32,
    id: u64,
) -> Result<Recorded, String> {
    let (Some(plan), Some(sl), Some(functions)) = (PLAN.get(), SL.get(), FUNCTIONS.get()) else {
        return Err("Streamline is not ready".into());
    };
    let (Some(optimal_of), Some(set_options)) = (functions.dlss_optimal, functions.dlss_set_options) else {
        return Err("no DLSS functions".into());
    };
    // (Timed with frame generation: its present shares this lock and
    // Streamline with the upscaling.)
    let began = plan.framegen.then(std::time::Instant::now);
    let mut calls = calls();
    if let Some(began) = began {
        spent(T_UPSCALE_LOCK, began);
    }
    let output = camera.output;
    let optimal = |calls: &mut Calls, mode: u32| -> Option<[u32; 2]> {
        if let Some(&(_, _, size)) = calls.optimal.iter().find(|(o, m, _)| *o == output && *m == mode) {
            return size;
        }
        let options = abi::DlssOptions::new(mode, output, plan.asked.preset);
        let mut settings = abi::DlssOptimalSettings::default();
        // SAFETY: filled options and a structure to fill.
        let code = unsafe { optimal_of(&options, &mut settings) };
        let size =
            (code == 0 && settings.optimal_render_width > 0).then_some([settings.optimal_render_width, settings.optimal_render_height]);
        calls.optimal.push((output, mode, size));
        size
    };
    let sizes: Vec<(u32, Option<[u32; 2]>)> = plan::MODES.iter().map(|&m| (m, optimal(&mut calls, m))).collect();
    let optimal_of_mode = |m: u32| sizes.iter().find(|s| s.0 == m).and_then(|s| s.1);
    let mode = plan::pick_mode(camera.render, output, optimal_of_mode)
        .ok_or_else(|| format!("no DLSS mode takes a {:?} render for a {output:?} output", camera.render))?;
    let mut options = abi::DlssOptions::new(mode, output, plan.asked.preset);
    options.pre_exposure = if pre_exposure > 0.0 { pre_exposure } else { 1.0 };
    options.color_buffers_hdr = abi::boolean(camera.flags & plan::FFX_HIGH_DYNAMIC_RANGE != 0);
    options.use_auto_exposure = abi::boolean(textures[EXPOSURE].is_none());
    let viewport = abi::ViewportHandle::new(VIEWPORT_DLSS);
    // SAFETY: filled options for our viewport.
    let code = unsafe { set_options(&viewport, &options) };
    if code != 0 {
        return Err(format!("slDLSSSetOptions: {} ({code})", abi::result_name(code)));
    }
    let changed = calls.last != Some((mode, camera.render, output));
    calls.last = Some((mode, camera.render, output));
    let token = frame_token(sl, id as u32).ok_or("no frame token")?;
    let constants = camera.constants(plan.asked.jitter, plan.asked.mvec);
    // SAFETY: constants for this frame and viewport.
    let code = unsafe { (sl.set_constants)(&constants, token, &viewport) };
    if code != 0 {
        return Err(format!("slSetConstants: {} ({code})", abi::result_name(code)));
    }
    let kinds = [
        (COLOUR, abi::BUFFER_SCALING_INPUT_COLOR, camera.render),
        (DEPTH, abi::BUFFER_DEPTH, camera.render),
        (MOTION, abi::BUFFER_MOTION_VECTORS, camera.motion_size()),
        (EXPOSURE, abi::BUFFER_EXPOSURE, [1, 1]),
        (REACTIVE, abi::BUFFER_BIAS_CURRENT_COLOR_HINT, camera.render),
        (OUTPUT, abi::BUFFER_SCALING_OUTPUT_COLOR, output),
    ];
    let mut resources: Vec<(abi::Resource, u32, [u32; 2])> = kinds
        .iter()
        .filter_map(|&(i, kind, size)| textures[i].as_ref().map(|(t, state)| (abi::Resource::texture(t.as_raw(), *state), kind, size)))
        .collect();
    let mut tags: Vec<abi::ResourceTag> = resources
        .iter_mut()
        .map(|(r, kind, size)| abi::ResourceTag::new(r, *kind, abi::VALID_UNTIL_EVALUATE, abi::Extent::sized(*size)))
        .collect();
    let mut inputs: Vec<*const abi::BaseStructure> = vec![std::ptr::from_ref(&viewport.base)];
    inputs.extend(tags.iter_mut().map(|t| std::ptr::from_ref(&t.base)));
    // SAFETY: an open command list; inputs that live through the call.
    let code = unsafe { (sl.evaluate_feature)(abi::FEATURE_DLSS, token, inputs.as_ptr(), inputs.len() as u32, list) };
    if code != 0 {
        return Err(format!("slEvaluateFeature: {} ({code})", abi::result_name(code)));
    }
    if let Some(began) = began {
        spent(T_UPSCALE, began);
    }
    Ok(Recorded { mode, optimal: optimal_of_mode(mode), changed, jitter: constants.jitter_offset, mvec_scale: constants.mvec_scale })
}

/// The title's FidelityFX upscaler `context` is destroyed: what DLSS made
/// for it goes once submitted work has run.
pub fn upscaler_destroyed(context: usize) {
    let (Some(sl), true) = (SL.get(), DLSS_READY.load(Acquire)) else { return };
    let _ = context;
    crate::external::finish_submitted();
    let viewport = abi::ViewportHandle::new(VIEWPORT_DLSS);
    let mut calls = calls();
    // SAFETY: our viewport; nothing that names it is in flight.
    let code = unsafe { (sl.free_resources)(abi::FEATURE_DLSS, &viewport) };
    calls.last = None;
    tracing::info!(code, result = abi::result_name(code), "DLSS: the title's upscaler was destroyed; DLSS let go of its resources");
}
