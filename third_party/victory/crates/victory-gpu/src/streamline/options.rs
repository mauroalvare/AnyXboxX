use super::*;

fn limit_generated(wanted: u32, driver_max: u32) -> u32 {
    wanted.saturating_sub(1).clamp(1, driver_max.max(1))
}

pub(super) fn set_mode(
    frame: u32,
    size: [u32; 2],
    viewport: &abi::ViewportHandle,
    functions: &Functions,
    set_options: abi::SlDlssgSetOptions,
    calls: &mut Calls,
    wanted: u32,
    on: bool,
) {
    let on = on && wanted != 0;
    let times = if on { wanted } else { 0 };
    if on == calls.framegen_on && times == calls.framegen_times {
        return;
    }
    let mut options = abi::DlssgOptions::new(if on { abi::DLSSG_ON } else { abi::DLSSG_OFF });
    // `VICTORY_FRAMEGEN_X=2..6`: frames shown for each one rendered (2 unless set), no more
    // than this card and driver generate (the state says how many: 5 on an RTX 50).
    let mut generated = wanted.max(2) - 1;
    if on
        && generated > 1
        && let Some(get_state) = functions.dlssg_get_state
    {
        let mut state = abi::DlssgState::default();
        // SAFETY: a structure to fill, no options.
        let code = unsafe { get_state(viewport, &mut state, std::ptr::null()) };
        let most = if code == 0 {
            state.num_frames_to_generate_max.max(1)
        } else {
            tracing::warn!(
                code,
                "Streamline did not report its frame generation limit; requesting 2x"
            );
            1
        };
        let supported = limit_generated(wanted, most);
        if generated != supported {
            tracing::warn!(
                asked = generated + 1,
                given = most + 1,
                "DLSS Frame Generation: more frames asked than this card generates"
            );
            generated = supported;
        }
    }
    options.num_frames_to_generate = generated;
    options.flags = abi::DLSSG_RETAIN_RESOURCES_WHEN_OFF;
    options.num_back_buffers = crate::host::BUFFERS;
    options.color_width = size[0];
    options.color_height = size[1];
    options.on_error_callback = Some(present_error);
    // SAFETY: filled options for viewport 0, on the presenting thread.
    let code = unsafe { set_options(viewport, &options) };

    // 39, "warning: out of VRAM": the options are set all the same, with the card's memory
    // past its budget. Taken as set: asked again at every present (1,828 times in one
    // run, after the menu's 3X on a full 12 GB card) it showed no generated frame at all.
    if code == 39 && on {
        tracing::warn!(
            times = generated + 1,
            "DLSS Frame Generation is on with the card's memory past its budget: frames may come unevenly (the launcher leaves it room when it is on from the start)"
        );
    }
    if code == 0 || code == 39 {
        tracing::info!(
            frame,
            on,
            times = generated + 1,
            code,
            "DLSS Frame Generation switched"
        );
        calls.framegen_on = on;
        // What was asked (the driver may give fewer): not set again until that changes.
        calls.framegen_times = times;
    } else {
        once!(
            warn,
            frame,
            on,
            code,
            result = abi::result_name(code),
            "Streamline refused the frame generation options"
        );
    }
}

unsafe extern "C" fn present_error(error: *const i32) {
    // SAFETY: the error for the call.
    let code = unsafe { error.as_ref() }.copied().unwrap_or(0);
    if code == windows::Win32::Foundation::DXGI_STATUS_OCCLUDED.0 {
        once!(warn, "DLSS Frame Generation: DXGI reports the output is occluded; generated frames cannot be displayed");
        return;
    }
    once!(
        warn,
        hresult = format_args!("{:#x}", code as u32),
        "DLSS Frame Generation: a present failed"
    );
}

/// Frames shown for each one rendered with frame generation on, as asked at the start:
/// `VICTORY_FRAMEGEN_X=2..6` (2 unless set); 0 leaves generation off.
pub(super) fn framegen_times() -> u32 {
    static TIMES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TIMES.get_or_init(|| {
        std::env::var("VICTORY_FRAMEGEN_X")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map_or(2, |n| if n == 0 { 0 } else { n.clamp(2, 6) })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rtx_40_adds_at_most_one_image() {
        assert_eq!(limit_generated(2, 1), 1);
        assert_eq!(limit_generated(6, 1), 1);
        assert_eq!(limit_generated(6, 0), 1);
        assert_eq!(limit_generated(4, 5), 3);
    }

    #[test]
    fn rejected_options_do_not_count_as_enabled_and_a_menu_turns_fg_off() {
        unsafe extern "C" fn refused(
            _: *const abi::ViewportHandle,
            _: *const abi::DlssgOptions,
        ) -> i32 {
            7
        }
        unsafe extern "C" fn accepted(
            _: *const abi::ViewportHandle,
            options: *const abi::DlssgOptions,
        ) -> i32 {
            // SAFETY: set_mode invokes this with a live options structure.
            let options = unsafe { &*options };
            if options.num_frames_to_generate == 1
                && options.num_back_buffers == crate::host::BUFFERS
                && [options.color_width, options.color_height] == [1920, 1080]
            {
                0
            } else {
                7
            }
        }
        let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
        let mut calls = Calls::default();
        let functions = Functions::default();
        set_mode(
            1,
            [1920, 1080],
            &viewport,
            &functions,
            refused,
            &mut calls,
            2,
            true,
        );
        assert!(!calls.framegen_on);
        set_mode(
            2,
            [1920, 1080],
            &viewport,
            &functions,
            accepted,
            &mut calls,
            2,
            true,
        );
        assert!(calls.framegen_on);
        set_mode(
            3,
            [1920, 1080],
            &viewport,
            &functions,
            accepted,
            &mut calls,
            2,
            false,
        );
        assert!(!calls.framegen_on);
    }
}
