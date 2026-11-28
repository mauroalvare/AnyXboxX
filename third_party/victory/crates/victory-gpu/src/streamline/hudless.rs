use super::*;

/// The title's interface composite where the interface is no plane of the
/// present (Series X): one triangle that blends the interface target
/// (3840x2160, premultiplied) over the frame's colour, with these vertex and
/// pixel shaders, the frame's last draw into the texture it presents. Seen
/// in every frame listing of 2026-11-26 (stock 3840x2160 output: once a
/// frame, after the scene's composite into the same texture) and bound from
/// frame 3 on in VIctory-NVIDIA's 1440p runs with build 1.0.2.1.
const INTERFACE_COMPOSITE: [&str; 2] = ["0602c2384acbb3b2", "a78d088fa55aa495"];

fn interface_composite<'a>(triangle: bool, stages: impl Iterator<Item = &'a str>) -> bool {
    // HostPipeline::stages lists the vertex shader first, then the pixel shader.
    triangle && stages.eq(INTERFACE_COMPOSITE)
}

/// Whether draws are looked at for the colour without the interface: frame
/// generation runs on our swap chain and `VICTORY_FRAMEGEN_HUDLESS` is not 0.
#[inline]
pub(crate) fn hudless_wanted() -> bool {
    HUDLESS_WANTED.load(Relaxed)
}

pub(super) static HUDLESS_WANTED: AtomicBool = AtomicBool::new(false);

/// The host textures of the last presents (their addresses): the textures
/// the title presents (it has two, in turn).
static SHOWN: [std::sync::atomic::AtomicUsize; 4] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 4];
static SHOWN_NEXT: AtomicU32 = AtomicU32::new(0);

/// A copy of a presented texture taken in front of the interface composite.
#[derive(Clone)]
pub(crate) struct Copied {
    pub(crate) id: u64,
    /// The host texture copied (its address).
    pub(super) target: usize,
    pub(super) texture: ID3D12Resource,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Copied {}

/// Submitted copies, in queue order.
pub(super) static HUD_SUBMITTED: Mutex<Vec<Copied>> = Mutex::new(Vec::new());
static HUD_NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// The two textures the copies go to in turn (a present reads one while the
/// next frame's lists may already be recorded), with their size and format.
struct HudlessTextures {
    size: [u32; 2],
    format: i32,
    textures: [ID3D12Resource; 2],
}

// SAFETY: as above.
unsafe impl Send for HudlessTextures {}

static HUD_TEXTURES: Mutex<Option<HudlessTextures>> = Mutex::new(None);

// Since the last line about it (every ten seconds): copies recorded, presents
// tagged with one, presents with frame generation on that had none, composite
// draws, those after a copy in their list, draws of other pipelines into a
// presented texture.
pub(super) static HUD_COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
const HUD_COPIES: usize = 0;
pub(super) static HUD_TAGGED_ALL: AtomicU64 = AtomicU64::new(0);
pub(super) const HUD_TAGGED: usize = 1;
pub(super) const HUD_MISSING: usize = 2;
const HUD_COMPOSITES: usize = 3;
const HUD_AGAIN: usize = 4;
const HUD_OTHERS: usize = 5;

pub(super) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Before a draw of `pipeline` into `target` alone (no depth), with frame
/// generation: if `target` is a texture the title presents and the draw is
/// the title's interface composite ([`INTERFACE_COMPOSITE`]; `triangle`: one
/// triangle, one instance), the target as it is now, the frame without the
/// interface, is copied into a texture of ours, recorded into `list` in
/// front of the draw; the present after the list's submission tags it as
/// the colour without the interface. One copy a list. Other pipelines that
/// draw into a presented texture are named in the log once each: the
/// interface is in the copy if one of them draws it before the composite.
pub(crate) fn before_target_draw(
    list: &crate::host::HostList,
    target: crate::object::This,
    pipeline: &crate::pipeline::host::HostPipeline,
    triangle: bool,
) {
    let Some(texture) = crate::resource::host_resource(target) else {
        return;
    };
    let shown = texture.as_raw() as usize;
    if !SHOWN.iter().any(|s| s.load(Relaxed) == shown) {
        return;
    }
    let composite = interface_composite(
        triangle,
        pipeline
            .stages
            .iter()
            .map(|stage| stage.shader.hash.as_str()),
    );
    if !composite {
        HUD_COUNTS[HUD_OTHERS].fetch_add(1, Relaxed);
        static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let key = std::ptr::from_ref(pipeline) as usize;
        let mut seen = lock(&SEEN);
        if !seen.contains(&key) && seen.len() < 24 {
            seen.push(key);
            tracing::info!(
                pipeline = %pipeline.name,
                state = %pipeline.state,
                triangle,
                after_the_copy_in_its_list = list.hudless() != 0,
                "DLSS Frame Generation: this pipeline draws into the texture the title presents (the scene's composite does; an interface drawn here before the interface composite would be in the colour without the interface)"
            );
        }
        return;
    }
    HUD_COUNTS[HUD_COMPOSITES].fetch_add(1, Relaxed);
    if list.hudless() != 0 {
        HUD_COUNTS[HUD_AGAIN].fetch_add(1, Relaxed);
        return;
    }
    let began = std::time::Instant::now();
    let id = HUD_NEXT_ID.fetch_add(1, Relaxed) + 1;
    let Some(copy) = hudless_texture(texture, id) else {
        return;
    };
    // The target as a copy's source: the composite's own preparation makes
    // it a render target again.
    const COPY_SOURCE: u32 = 0x800;
    crate::list::ensure_state(list, target, COPY_SOURCE);
    // SAFETY: an open list; the source is in the copy source state in it,
    // ours stays a copy's destination; same size, one level, formats of one
    // family.
    unsafe { list.list.CopyResource(&copy, texture) };
    list.keep(&copy);
    list.keep(texture);
    list.mark_hudless(Copied {
        id,
        target: shown,
        texture: copy,
    });
    HUD_COUNTS[HUD_COPIES].fetch_add(1, Relaxed);
    spent(T_COPY, began);
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    once!(
        info,
        width = desc.Width,
        height = desc.Height,
        format = desc.Format.0,
        pipeline = %pipeline.name,
        state = %pipeline.state,
        "DLSS Frame Generation: the colour without the interface is copied in front of the title's interface composite (first time)"
    );
}

/// One of the two textures the colour without the interface is copied to
/// (copy `id`'s), made like `target` in the swap chain's format; `None` if
/// `target` is no plain 2D texture of a format that is shown.
fn hudless_texture(target: &ID3D12Resource, id: u64) -> Option<ID3D12Resource> {
    // SAFETY: a live texture.
    let desc = unsafe { target.GetDesc() };
    let plain = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.MipLevels == 1
        && desc.SampleDesc.Count == 1;
    let format = crate::host::swap_format(desc.Format).filter(|_| plain);
    let Some(format) = format else {
        once!(
            warn,
            format = desc.Format.0,
            levels = desc.MipLevels,
            samples = desc.SampleDesc.Count,
            "DLSS Frame Generation: the presented texture cannot be copied for the colour without the interface; none is tagged"
        );
        return None;
    };
    let size = [desc.Width as u32, desc.Height];
    let mut textures = lock(&HUD_TEXTURES);
    if textures
        .as_ref()
        .is_none_or(|t| t.size != size || t.format != format.0)
    {
        let device = &crate::host::host()?.device;
        let make = || texture(device, size, format, D3D12_RESOURCE_FLAG_NONE, COPY_DEST);
        match (make(), make()) {
            (Ok(a), Ok(b)) => {
                *textures = Some(HudlessTextures {
                    size,
                    format: format.0,
                    textures: [a, b],
                })
            }
            (Err(e), _) | (_, Err(e)) => {
                once!(warn, error = %e, "DLSS Frame Generation: no texture for the colour without the interface; none is tagged");
                return None;
            }
        }
    }
    textures
        .as_ref()
        .map(|t| t.textures[(id % 2) as usize].clone())
}

/// A host list that recorded copy `id` of the colour without the interface
/// is submitted (in submission order, before the present that follows it).
pub(crate) fn hudless_submitted(copied: Copied) {
    let mut submitted = lock(&HUD_SUBMITTED);
    // A recording can be submitted out of ID order. An earlier copy into
    // the same destination will have been overwritten on the direct queue.
    submitted.retain(|previous| previous.texture.as_raw() != copied.texture.as_raw());
    submitted.push(copied);
}

/// Buffers and displayed-target identities belonged to the previous swap-chain size.
pub(crate) fn hudless_resized() {
    if hudless_wanted() {
        for shown in &SHOWN {
            shown.store(0, Relaxed);
        }
        lock(&HUD_SUBMITTED).clear();
        *lock(&HUD_TEXTURES) = None;
    }
}

/// At a present of host texture `shown`: it is one the title presents
/// ([`SHOWN`]), and the first copy of it submitted since its last present,
/// if any, is the frame's colour without the interface.
pub(super) fn hudless_for(shown: &ID3D12Resource) -> Option<ID3D12Resource> {
    let shown = shown.as_raw() as usize;
    if !SHOWN.iter().any(|s| s.load(Relaxed) == shown) {
        SHOWN[SHOWN_NEXT.fetch_add(1, Relaxed) as usize % SHOWN.len()].store(shown, Relaxed);
    }
    let mut submitted = lock(&HUD_SUBMITTED);
    let first = submitted
        .iter()
        .position(|c| c.target == shown)
        .map(|i| submitted.remove(i));
    submitted.retain(|c| c.target != shown);
    first.map(|c| c.texture)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_builds_single_triangle_ui_composite_matches() {
        let [vs, ps] = INTERFACE_COMPOSITE;
        assert!(interface_composite(true, [vs, ps].into_iter()));
        assert!(!interface_composite(false, [vs, ps].into_iter()));
        assert!(!interface_composite(true, [ps, vs].into_iter()));
        assert!(!interface_composite(true, [vs, "unknown"].into_iter()));
    }
}
