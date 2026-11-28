use super::upscale::{Textures, record};
use super::*;

/// Exercise Streamline without loading a title. Release the synthetic GPU
/// resources and the swap chain before shutting down the SDK on every exit.
pub fn check(output: [u32; 2], device: bool, present: bool) -> Result<(), String> {
    let mut asked = Asked::from_env()?;
    asked.dlss = true;
    asked.framegen = !asked.present_hook;
    let available = load_and_init(&asked, false).map_err(|why| format!("Streamline at {}: {why}", asked.dir.display()))?;
    println!("DLSS Super Resolution: {}. Frame Generation with Reflex: {}.", available.dlss, available.framegen);
    let result = if device || present {
        match crate::host::host() {
            Some(host) => {
                CHECKING.store(true, Relaxed);
                let hidden = present && inactive_desktop();
                let result = exercise(host, &asked, output, present && !hidden).and_then(|()| {
                    if hidden {
                        Err("cannot measure displayed frames on an inactive Windows desktop; run --present on the visible desktop".into())
                    } else {
                        Ok(())
                    }
                });
                if let Err(why) = &result {
                    eprintln!("Streamline check: {why}");
                }
                tracing::info!("Streamline check: disabling frame generation");
                resizing();
                tracing::info!("Streamline check: releasing the swap chain");
                host.drop_presenter();
                *lock(&SUBMITTED) = None;
                lock(&HUD_SUBMITTED).clear();
                result
            }
            None => Err("no host D3D12 device".into()),
        }
    } else if available.dlss || available.framegen {
        Ok(())
    } else {
        Err("no requested feature is supported".into())
    };
    shutdown()?;
    result
}

/// Standalone diagnostic only. With an offline driver we observed NGX waiting
/// indefinitely in UninitializeTelemetry after all GPU work had completed.
/// Keep slShutdown on the initializing thread. A timeout is a failed check.
fn shutdown() -> Result<(), String> {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::Duration;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};

    let (finished, wait) = mpsc::channel();
    let watchdog = std::thread::Builder::new()
        .name("streamline-check-shutdown".into())
        .spawn(move || {
            if let Err(RecvTimeoutError::Timeout) = wait.recv_timeout(Duration::from_secs(30)) {
                eprintln!("Streamline check failed: slShutdown did not return within 30 seconds; ending the diagnostic");
                // A blocked driver may also hang in DLL_PROCESS_DETACH on exit.
                // SAFETY: only this standalone probe, after its GPU work finished.
                unsafe { TerminateProcess(GetCurrentProcess(), 1) };
            }
        })
        .map_err(|e| format!("cannot monitor Streamline shutdown: {e}"))?;
    tracing::info!("Streamline check: shutting down the SDK");
    // SAFETY: this is the only Streamline session and its GPU work finished.
    let code = unsafe { (SL.get().expect("initialized Streamline").shutdown)() };
    let _ = finished.send(());
    let _ = watchdog.join();
    println!("Streamline shut down: {} ({code})", abi::result_name(code));
    if code == 0 { Ok(()) } else { Err(format!("slShutdown failed ({code})")) }
}

fn exercise(host: &crate::host::Host, asked: &Asked, output: [u32; 2], present: bool) -> Result<(), String> {
    let framegen = FG_READY.load(Acquire);
    if !DLSS_READY.load(Acquire) {
        return Err("DLSS device hand-over failed".into());
    }
    let render = asked.mode.render_size(output);
    let camera = plan::Camera {
        jitter: [0.0, 0.0],
        motion_scale: [render[0] as f32, render[1] as f32],
        render,
        output,
        // (The far plane first, as the title names them.)
        near: 10_000.0,
        far: 0.15,
        fov_y: 1.0,
        reset: true,
        flags: plan::FFX_HIGH_DYNAMIC_RANGE | plan::FFX_DEPTH_INVERTED | plan::FFX_AUTO_EXPOSURE,
    };
    let device = &host.device;
    let made = (|| -> windows::core::Result<_> {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let depth = texture(device, render, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, SHADER_READ)?;
        let motion = texture(device, render, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_NONE, SHADER_READ)?;
        let colour = texture(device, render, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, SHADER_READ)?;
        let out = texture(device, output, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, UNORDERED)?;
        let base = texture(device, output, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 0)?;
        Ok((depth, motion, colour, out, base))
    })();
    let (depth, motion, colour, out, base) = match made {
        Ok(made) => made,
        Err(e) => {
            return Err(format!("stand-in textures not made: {e}"));
        }
    };
    {
        // (Known once the first evaluation asked for them.)
        let sizes = || -> String {
            let calls = calls();
            let sizes: Vec<String> = calls
                .optimal
                .iter()
                .filter(|(o, _, _)| *o == output)
                .map(|&(_, m, s)| format!("{} {:?}", plan::sl_mode_name(m), s))
                .collect();
            if sizes.is_empty() { "?".into() } else { sizes.join(", ") }
        };
        let mut textures: Textures = Default::default();
        textures[COLOUR] = Some((colour, SHADER_READ));
        textures[DEPTH] = Some((depth.clone(), SHADER_READ));
        textures[MOTION] = Some((motion.clone(), SHADER_READ));
        textures[OUTPUT] = Some((out, UNORDERED));
        // (`VICTORY_STREAMLINE_PRESENT=1`: DLSS alone, with a present through
        // Streamline's proxy of our swap chain after each upscaling, more of
        // them than Streamline waits for a present.)
        let hooked = present && asked.present_hook && !framegen;
        for round in 0..if hooked { 120 } else { 3 } {
            if hooked && round > 0 {
                host.present_planes(&base, 0, None, round, None).ok_or("synthetic present failed")?;
            }
            let id = NEXT_ID.fetch_add(1, Relaxed) + 1;
            let list = match host.new_list() {
                Ok(list) => list,
                Err(e) => {
                    return Err(format!("no command list: {e}"));
                }
            };
            let recorded = record(list.list.as_raw(), &textures, &camera, 1.0, id);
            list.close();
            host.execute_sync_internal(&list);
            // SAFETY: a plain query.
            let removed = unsafe { device.GetDeviceRemovedReason() };
            match (recorded, removed) {
                (Ok(r), Ok(())) if round == 0 => println!(
                    "DLSS ran on the GPU: {render:?} -> {output:?} as {} (optimal {:?}); render sizes for {output:?}: {}",
                    plan::sl_mode_name(r.mode),
                    r.optimal,
                    sizes()
                ),
                (Ok(_), Ok(())) => {}
                (Err(why), _) => {
                    return Err(format!("DLSS did not run: {why}"));
                }
                (_, Err(e)) => {
                    return Err(format!("the device was removed after DLSS ran: {e}"));
                }
            }
        }
        if hooked && !PRESENT_HOOKED.load(Relaxed) {
            return Err("DLSS swap chain proxy unavailable".into());
        }
        println!(
            "{}",
            if hooked {
                format!(
                    "DLSS ran 120 times on stand-in textures with a present after each; our swap chain is {}Streamline's proxy of it (VICTORY_STREAMLINE_PRESENT=1); the device is fine",
                    if PRESENT_HOOKED.load(Relaxed) { "" } else { "NOT " }
                )
            } else {
                "DLSS ran three times on stand-in textures; the device is fine".into()
            }
        );
    }
    if present && framegen {
        let (on, shown) = (FG_ON_ALL.load(Relaxed), FG_SHOWN_ALL.load(Relaxed));
        // A stand-in for the colour without the interface, tagged as the
        // copies in front of the title's interface composite are: ours, a
        // copy's destination, of the back buffer's size and format.
        let copy = {
            use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
            texture(device, output, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, COPY_DEST).ok()
        };
        let tagged = HUD_TAGGED_ALL.load(Relaxed);
        for frame in 1..=180u64 {
            latency::mark(|| frame, latency::Phase::Simulation);
            latency::mark(|| frame, latency::Phase::RenderSubmit);
            let id = NEXT_ID.fetch_add(1, Relaxed) + 1;
            *SUBMITTED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Noted {
                id,
                depth: Source::Fixed(depth.clone(), SHADER_READ),
                motion: Source::Fixed(motion.clone(), SHADER_READ),
                camera,
            });
            if let Some(copy) = &copy {
                lock(&HUD_SUBMITTED).push(Copied { id: frame, target: base.as_raw() as usize, texture: copy.clone() });
            }
            latency::mark(|| frame, latency::Phase::Submitted);
            host.present_planes(&base, 0, None, frame, None).ok_or("synthetic present failed")?;
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        crate::host::settle_for_fence();
        println!(
            "frame generation: 180 presents, {} with it on, {} frames shown by it, {} presents tagged with a colour without the interface, status {:#x} (0 is fine){}",
            FG_ON_ALL.load(Relaxed) - on,
            FG_SHOWN_ALL.load(Relaxed) - shown,
            HUD_TAGGED_ALL.load(Relaxed) - tagged,
            FG_STATUS.load(Relaxed),
            if SWAP_CHAIN_PROXIED.load(Relaxed) { "" } else { "; the swap chain is NOT Streamline's proxy" }
        );
        if FG_STATUS.load(Relaxed) != 0
            || FG_ON_ALL.load(Relaxed) - on != 180
            || FG_SHOWN_ALL.load(Relaxed) - shown <= 180
            || (asked.hudless && HUD_TAGGED_ALL.load(Relaxed) - tagged != 180)
        {
            return Err("synthetic frame generation check failed (see counts above)".into());
        }
        // SAFETY: a plain query.
        if let Err(e) = unsafe { device.GetDeviceRemovedReason() } {
            return Err(format!("the device was removed while presenting: {e}"));
        }
    } else if present && !asked.present_hook {
        return Err("frame generation is not ready".into());
    }
    Ok(())
}
const SHADER_READ: u32 = 0xc0;
const UNORDERED: u32 = 0x8;

/// DXGI reports occlusion on an inactive desktop. The driver may wait for
/// scan-out during shutdown if a frame-generation swap chain was made there.
fn inactive_desktop() -> bool {
    use windows_sys::Win32::System::StationsAndDesktops::{GetThreadDesktop, GetUserObjectInformationW, UOI_IO};
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    let mut receives_input = 0i32;
    // SAFETY: borrowed desktop of this thread, BOOL-sized out parameter.
    unsafe {
        let desktop = GetThreadDesktop(GetCurrentThreadId());
        GetUserObjectInformationW(
            desktop,
            UOI_IO,
            (&raw mut receives_input).cast(),
            size_of_val(&receives_input) as u32,
            std::ptr::null_mut(),
        ) != 0
            && receives_input == 0
    }
}
