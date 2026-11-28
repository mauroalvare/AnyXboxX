//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, DLSS through
//! NGX, or DLSS through NVIDIA Streamline, `crate::streamline`, say): the
//! host device, a title texture's host copy in the state the library is
//! told, and the host list itself. Two ways in: `record` hands out a
//! `Recording` (FidelityFX and NGX), and `host_texture` / `prepare_input`
//! work on a `HostList` the list translation already holds (Streamline,
//! `crate::list::external`, also with `VICTORY_THIN=1`).
//!
//! The Streamline entry points are ported from DexNom's fork
//! (DexNom/VIctory, `crates/victory-gpu/src/external.rs`, GPL-3.0-or-later).

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::host::HostList;
use crate::object::This;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// The active host list for one synchronous external recording call.
pub struct Recording<'a> {
    host: &'a crate::host::HostList,
    title_list: This,
}

impl Recording<'_> {
    /// The backing and its last state on this list, without recording a barrier.
    pub fn texture_binding(&self, resource: This) -> Option<(usize, Option<u32>)> {
        let copy = crate::resource::host_copy(resource)?;
        Some(self.host.texture_binding(&copy))
    }

    /// A failed external recording must not allocate a missing backing on rollback.
    pub fn existing_texture_binding(&self, resource: This) -> Option<(usize, Option<u32>)> {
        let copy = crate::resource::made_copy(resource)?;
        Some(self.host.texture_binding(&copy))
    }

    /// Move a title texture to the state the provider will see. The list
    /// holds the backing resource through submission.
    pub fn texture(&self, resource: This, state: u32) -> Option<(ID3D12Resource, u32)> {
        host_texture(self.host, resource, state)
    }

    /// Prepare aliases, mip uploads and depth twins like a compute SRV.
    pub fn prepare_input(&self, resource: This) -> Option<()> {
        prepare_input(self.host, resource)
    }

    /// Holds a reference to `object` (what the provider recorded uses, such
    /// as a DLSS feature) until the host list is reset, after its last
    /// submission has run.
    pub fn keep(&self, object: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        self.host.keep_shared(object);
    }

    /// Access flushes any pending host barriers before the provider runs.
    pub fn commands(&self) -> &ID3D12GraphicsCommandList {
        &self.host.list
    }

    /// Record AMD's preparation after the upscaler, on the same host list.
    pub fn prepare_framegen(&self, input: crate::framegen::Inputs<'_>) {
        if let Some(preparation) = crate::framegen::prepare(self.commands(), self.title_list, input) {
            self.host.keep_frame_preparation(preparation);
        }
    }
}

/// A title texture's host copy, moved to the host resource `state` in
/// `list`, and the state it is in: `state` without the bits the host copy
/// cannot take (see `host::legal_state`). What the list records next finds
/// it there and must leave it there. The list holds the texture until it is
/// reset. `None` for a buffer.
pub(crate) fn host_texture(list: &HostList, resource: This, state: u32) -> Option<(ID3D12Resource, u32)> {
    let texture = crate::resource::host_resource(resource)?.clone();
    crate::list::ensure_state(list, resource, state);
    let state = crate::host::legal_state(&texture, state);
    Some((texture, state))
}

/// Prepares a title-authored input with the alias, upload and depth twin
/// rules a compute shader's resource view has in the renderer.
pub(crate) fn prepare_input(list: &HostList, resource: This) -> Option<()> {
    let texture = crate::resource::host_resource(resource)?;
    let aliased = crate::resource::alias_of(resource).is_some_and(|source| crate::texture::alias(list, resource, source));
    if !aliased && crate::texture::title_written(texture) {
        crate::texture::upload_from(list, resource, 0);
    }
    list.twin_use(resource, false, false);
    Some(())
}

/// The producer recorded its output on the host's command list.
pub fn output_written(resource: This) {
    crate::texture::host_written(resource);
}

/// Settle submitted title work before releasing a provider's private GPU resources.
pub fn finish_submitted() {
    crate::queue::before_sync();
    crate::host::settle_for_fence();
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(list: This, f: impl FnOnce(&Recording<'_>) -> R) -> Option<R> {
    if crate::framegen::requested() {
        // Finish the prior presentation before opening an upscaler list.
        // AMD's frame generator reuses its preparation buffers across presents.
        crate::queue::wait_before_framegen();
    }
    let recording = crate::list::thin::for_external(list)?;
    let mut out = None;
    crate::list::with_state(recording, |host, state| {
        let recording = Recording {
            host,
            title_list: list,
        };
        out = Some(f(&recording));
        state.external_compute();
    });
    out
}
