use super::*;

/// Streamline's proxies kept for frame generation: the device that made the
/// direct queue and the queue the swap chain is made on.
struct Proxies {
    _device: ID3D12Device,
    queue: ID3D12CommandQueue,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Proxies {}
// SAFETY: as above.
unsafe impl Sync for Proxies {}

static PROXIES: OnceLock<Proxies> = OnceLock::new();
/// The host device is made: Streamline is given it (`slSetD3DDevice`), the
/// feature functions are fetched, and, for frame generation, the direct
/// queue is made through Streamline's device proxy. The native queue to use
/// for everything, if it was made here.
pub(crate) fn device_made(device: &ID3D12Device) -> Option<ID3D12CommandQueue> {
    let plan = PLAN.get()?;
    let sl = SL.get()?;
    // SAFETY: the native device.
    let code = unsafe { (sl.set_d3d_device)(device.as_raw()) };
    if code != 0 {
        tracing::warn!("Streamline: slSetD3DDevice failed: {} ({code}); DLSS and frame generation stay off", abi::result_name(code));
        return None;
    }
    let function = |feature: u32, name: &CStr| -> Option<*mut c_void> {
        let mut f = std::ptr::null_mut();
        // SAFETY: a NUL-terminated name and an out pointer.
        let code = unsafe { (sl.get_feature_function)(feature, name.as_ptr(), &mut f) };
        if code != 0 || f.is_null() {
            tracing::warn!("Streamline: no {} ({}, {code})", name.to_string_lossy(), abi::result_name(code));
            return None;
        }
        Some(f)
    };
    macro_rules! f {
        ($feature:expr, $name:literal, $t:ty) => {
            // SAFETY: the function of that name has the header's signature.
            function($feature, $name).map(|f| unsafe { std::mem::transmute::<*mut c_void, $t>(f) })
        };
    }
    let mut functions = Functions::default();
    if plan.dlss {
        functions.dlss_optimal = f!(abi::FEATURE_DLSS, c"slDLSSGetOptimalSettings", abi::SlDlssGetOptimalSettings);
        functions.dlss_set_options = f!(abi::FEATURE_DLSS, c"slDLSSSetOptions", abi::SlDlssSetOptions);
        let ready = functions.dlss_optimal.is_some() && functions.dlss_set_options.is_some();
        log_version(sl, abi::FEATURE_DLSS, "DLSS Super Resolution");
        DLSS_READY.store(ready, Release);
    }
    let mut queue = None;
    if plan.framegen {
        functions.dlssg_set_options = f!(abi::FEATURE_DLSS_G, c"slDLSSGSetOptions", abi::SlDlssgSetOptions);
        functions.dlssg_get_state = f!(abi::FEATURE_DLSS_G, c"slDLSSGGetState", abi::SlDlssgGetState);
        functions.reflex_set_options = f!(abi::FEATURE_REFLEX, c"slReflexSetOptions", abi::SlReflexSetOptions);
        functions.pcl_set_marker = f!(abi::FEATURE_PCL, c"slPCLSetMarker", abi::SlPclSetMarker);
        log_version(sl, abi::FEATURE_DLSS_G, "DLSS Frame Generation");
        log_version(sl, abi::FEATURE_REFLEX, "Reflex");
        if functions.dlssg_set_options.is_some() && functions.dlssg_get_state.is_some() && latency::enable_reflex(&functions) {
            queue = proxied_queue(sl, device);
        }
        if queue.is_none() {
            tracing::warn!("DLSS Frame Generation stays off: the device hand-over to Streamline did not complete");
        }
    }
    let _ = FUNCTIONS.set(functions);
    FG_READY.store(queue.is_some(), Release);
    queue
}

fn log_version(sl: &Sl, feature: u32, name: &str) {
    let mut version = abi::FeatureVersion {
        base: abi::BaseStructure::new(abi::FEATURE_VERSION, 1),
        version_sl: abi::Version::default(),
        version_ngx: abi::Version::default(),
    };
    // SAFETY: a structure to fill.
    let code = unsafe { (sl.get_feature_version)(feature, &mut version) };
    if code == 0 {
        tracing::info!(streamline = %version.version_sl, ngx = %version.version_ngx, "Streamline: {name} loaded");
    } else {
        tracing::warn!("Streamline: {name} has no version: {} ({code})", abi::result_name(code));
    }
}

/// The direct queue, made through Streamline's proxy of `device` (frame
/// generation hooks queue creation); the native queue is returned for our
/// use and the proxy kept for the swap chain.
fn proxied_queue(sl: &Sl, device: &ID3D12Device) -> Option<ID3D12CommandQueue> {
    let mut raw = device.as_raw();
    // SAFETY: an owned reference of ours; on success `raw` is a new proxy
    // that holds its own reference to the device.
    let code = unsafe { (sl.upgrade_interface)(&mut raw) };
    if code != 0 || raw.is_null() || raw == device.as_raw() {
        tracing::warn!("Streamline: the device was not upgraded: {} ({code})", abi::result_name(code));
        return None;
    }
    // SAFETY: the proxy's one reference, now ours.
    let proxy_device = unsafe { ID3D12Device::from_raw(raw) };
    let desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
    // SAFETY: plain queue creation through the proxy.
    let proxy_queue: ID3D12CommandQueue = match unsafe { proxy_device.CreateCommandQueue(&desc) } {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(error = %e, "Streamline: no queue from the device proxy");
            return None;
        }
    };
    let mut native = std::ptr::null_mut();
    // SAFETY: a proxy queue; the native one comes back with a reference.
    let code = unsafe { (sl.get_native_interface)(proxy_queue.as_raw(), &mut native) };
    if code != 0 || native.is_null() {
        tracing::warn!("Streamline: no native queue behind the proxy: {} ({code})", abi::result_name(code));
        return None;
    }
    // SAFETY: the reference slGetNativeInterface added.
    let native = unsafe { ID3D12CommandQueue::from_raw(native) };
    tracing::info!("Streamline: the direct queue was made through the device proxy (frame generation)");
    let _ = PROXIES.set(Proxies { _device: proxy_device, queue: proxy_queue });
    Some(native)
}

/// For frame generation: Streamline's proxy of `factory` and the queue
/// proxy to make the swap chain with; `None` to make it as before.
pub(crate) fn swap_chain_factory(factory: &IDXGIFactory2) -> Option<(IDXGIFactory2, ID3D12CommandQueue)> {
    if !FG_READY.load(Acquire) {
        return None;
    }
    let sl = SL.get()?;
    let queue = PROXIES.get()?.queue.clone();
    let mut raw = factory.as_raw();
    // SAFETY: as for the device: a new proxy with its own reference.
    let code = unsafe { (sl.upgrade_interface)(&mut raw) };
    if code != 0 || raw.is_null() || raw == factory.as_raw() {
        tracing::warn!("Streamline: the DXGI factory was not upgraded: {} ({code})", abi::result_name(code));
        FG_READY.store(false, Release);
        return None;
    }
    // SAFETY: the proxy's reference, now ours.
    let proxy = unsafe { IDXGIFactory2::from_raw(raw) };
    Some((proxy, queue))
}

/// The swap chain was made through the proxy (or not, after all: then
/// frame generation stays off).
pub(crate) fn swap_chain_made(proxied: bool, error: Option<&windows::core::Error>) {
    SWAP_CHAIN_PROXIED.store(proxied, Release);
    HUDLESS_WANTED.store(proxied && FG_READY.load(Acquire) && PLAN.get().is_some_and(|p| p.asked.hudless), Release);
    if proxied {
        tracing::info!("Streamline: our swap chain is Streamline's proxy; DLSS Frame Generation can run on its presents");
    } else {
        FG_READY.store(false, Release);
        tracing::warn!(
            error = error.map(|e| e.to_string()).unwrap_or_default(),
            "Streamline: the swap chain could not be made through the proxy; made as before, frame generation off"
        );
    }
}

/// `VICTORY_STREAMLINE_PRESENT=1`, DLSS without frame generation: Streamline's
/// proxy of `swap_chain`, whose Present, ResizeBuffers and GetBuffer run
/// Streamline's hooks first (its common plugin frees what it retired and
/// keeps its video memory budget at each present; without it, it logs
/// "presentCommon() was not observed" after 60 upscalings and frees nothing
/// it retired). `swap_chain` itself if the switch is off, frame generation
/// owns the swap chain, or the upgrade fails.
pub(crate) fn present_hooked(swap_chain: IDXGISwapChain3) -> IDXGISwapChain3 {
    let (Some(plan), Some(sl)) = (PLAN.get(), SL.get()) else { return swap_chain };
    if !plan.asked.present_hook || !plan.dlss || !DLSS_READY.load(Acquire) || SWAP_CHAIN_PROXIED.load(Acquire) {
        return swap_chain;
    }
    let mut raw = swap_chain.as_raw();
    // SAFETY: an owned reference of ours; on success `raw` is a new proxy
    // that holds its own reference to the swap chain.
    let code = unsafe { (sl.upgrade_interface)(&mut raw) };
    if code != 0 || raw.is_null() || raw == swap_chain.as_raw() {
        tracing::warn!(
            "Streamline: the swap chain was not upgraded: {} ({code}); presents stay unseen by Streamline",
            abi::result_name(code)
        );
        return swap_chain;
    }
    tracing::info!("Streamline: our swap chain is Streamline's proxy of it (VICTORY_STREAMLINE_PRESENT=1): it sees every present");
    PRESENT_HOOKED.store(true, Release);
    // SAFETY: the proxy's one reference, now ours; it implements the swap
    // chain interfaces up to IDXGISwapChain4 and holds the swap chain.
    unsafe { IDXGISwapChain3::from_raw(raw) }
}

pub(super) const COPY_DEST: u32 = 0x400;

/// A host texture for the HUD-free copy or the standalone GPU check.
pub(super) fn texture(
    device: &ID3D12Device,
    size: [u32; 2],
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
    state: u32,
) -> windows::core::Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: u64::from(size[0]),
        Height: size[1],
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: plain texture creation.
    unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATES(state as i32), None, &mut made)? };
    made.ok_or_else(windows::core::Error::empty)
}
