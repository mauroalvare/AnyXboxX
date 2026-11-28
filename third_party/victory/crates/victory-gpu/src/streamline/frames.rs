use super::*;

/// Frame generation: what the present after this list's submission tags,
/// noted on the host list ([`submitted`] then makes it the latest).
pub(super) fn note(host: &crate::host::HostList, up: &Upscale, id: u64) -> bool {
    let (Some(depth), Some(motion)) = (
        crate::resource::host_copy(up.resources[DEPTH].0),
        crate::resource::host_copy(up.resources[MOTION].0),
    ) else {
        once!(
            warn,
            "DLSS Frame Generation: no host depth or motion vectors; this dispatch is left to the title"
        );
        return false;
    };
    host.mark_upscale(Noted {
        id,
        depth: Source::Host(depth),
        motion: Source::Host(motion),
        camera: up.camera,
    });
    true
}

/// The inputs of an upscaling for frame generation.
#[derive(Clone)]
pub(crate) struct Noted {
    pub(super) id: u64,
    pub(super) depth: Source,
    pub(super) motion: Source,
    pub(super) camera: plan::Camera,
}

/// A texture frame generation reads at the present, and the state it is in
/// then: a title texture's host copy (in the state the lists submitted so
/// far left it in), or one of the check's ([`check`]) in a fixed state.
#[derive(Clone)]
pub(super) enum Source {
    Host(Arc<crate::resource::HostTexture>),
    Fixed(ID3D12Resource, u32),
}

impl Source {
    pub(super) fn tagged(&self) -> abi::Resource {
        match self {
            Source::Host(t) => abi::Resource::texture(t.texture.as_raw(), t.state.load(Relaxed)),
            Source::Fixed(t, state) => abi::Resource::texture(t.as_raw(), *state),
        }
    }
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Source {}

/// The most recently submitted upscale, consumed once at the next present.
/// Recording IDs can be out of order when two title threads build lists.
pub(super) static SUBMITTED: Mutex<Option<Noted>> = Mutex::new(None);

pub(crate) fn submitted(noted: Noted) {
    *lock(&SUBMITTED) = Some(noted);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16_FLOAT;

    #[test]
    fn present_consumes_the_last_submission_even_if_it_was_recorded_first() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let input = texture(
            &host.device,
            [16, 16],
            DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_FLAG_NONE,
            0,
        )
        .unwrap();
        let noted = |id| Noted {
            id,
            depth: Source::Fixed(input.clone(), 0),
            motion: Source::Fixed(input.clone(), 0),
            camera: plan::Camera {
                jitter: [0.0; 2],
                motion_scale: [1.0; 2],
                render: [16; 2],
                output: [16; 2],
                near: 0.1,
                far: 1000.0,
                fov_y: 1.0,
                reset: false,
                flags: 0,
            },
        };
        let first = host.new_list().unwrap();
        let second = host.new_list().unwrap();
        first.mark_upscale(noted(1));
        second.mark_upscale(noted(2));
        second.close();
        first.close();
        second.submit_frame_inputs();
        first.submit_frame_inputs();
        assert_eq!(lock(&SUBMITTED).take().unwrap().id, 1);
        assert!(lock(&SUBMITTED).take().is_none());
        second.submit_frame_inputs();
        assert_eq!(lock(&SUBMITTED).take().unwrap().id, 2);
        second.reset();
        second.submit_frame_inputs();
        assert!(lock(&SUBMITTED).take().is_none());
    }
}
