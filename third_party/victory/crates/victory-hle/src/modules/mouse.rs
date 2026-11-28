//! The mouse as the stand-in gamepad reads it (ported from the Neptune base).
//!
//! The right button is the left trigger (aim), the left button the right trigger (fire), and the
//! movement is the right stick (-1 to 1, up positive). The pointer is the user's own and gives the
//! pad nothing until a click on the picture (or F9) takes it; from then the movement since the last
//! reading is a stick, smoothed over a few readings, and the pointer is hidden and put back at the
//! picture's middle after each reading. F9 lets it go, and so does the window losing focus (or not
//! being in front): the cursor is never touched then, and it is not taken again before the next
//! click on the picture.
//!
//! Switch: `VICTORY_MOUSE=<scale>` multiplies the movement (default 1; `0` leaves the buttons
//! working but gives the stick nothing).

/// How much of the previous stick value is kept at each reading.
const KEEP: f32 = 0.8;
/// Stick per pixel of movement: about 8 pixels a reading (1000 a second) is the stick at its end.
const GAIN: f32 = 0.025;

/// What the stand-in pad takes from the mouse.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mouse {
    pub left: bool,
    pub right: bool,
    pub x: f32,
    pub y: f32,
}

/// The stick value after one more reading: the old one decayed, plus the pixels moved.
pub fn smooth(previous: f32, pixels: i32, scale: f32) -> f32 {
    previous * KEEP + pixels as f32 * GAIN * scale
}

/// A reading from the smoothed stick (screen y grows downwards, the stick's up is positive).
pub fn reading(left: bool, right: bool, x: f32, y_down: f32) -> Mouse {
    Mouse {
        left,
        right,
        x: x.clamp(-1.0, 1.0),
        y: (-y_down).clamp(-1.0, 1.0),
    }
}

/// The host's mouse; nothing where there is none, or while the window is not in front.
pub fn read(focused: bool) -> Mouse {
    #[cfg(windows)]
    {
        host::read(focused)
    }
    #[cfg(not(windows))]
    {
        let _ = focused;
        Mouse::default()
    }
}

#[cfg(windows)]
mod host {
    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering::Relaxed};
    use std::sync::{LazyLock, Mutex};

    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallWindowProcW, GWLP_WNDPROC, GetCursorPos, GetForegroundWindow, GetWindowInfo,
        GetWindowLongPtrW, SetCursor, SetCursorPos, SetWindowLongPtrW, WINDOWINFO, WNDPROC,
    };

    use super::{Mouse, reading, smooth};

    /// The pointer is taken: the window shows no cursor over its picture.
    static TAKEN: AtomicBool = AtomicBool::new(false);
    /// The window whose procedure `hidden` stands before, and that procedure.
    static BEFORE: [AtomicIsize; 2] = [AtomicIsize::new(0), AtomicIsize::new(0)];

    /// Stands before the window's own procedure: no cursor over the picture while the pointer is
    /// taken.
    unsafe extern "system" fn hidden(hwnd: HWND, message: u32, w: WPARAM, l: LPARAM) -> LRESULT {
        const WM_SETCURSOR: u32 = 0x20;
        const HTCLIENT: isize = 1;
        // SAFETY: no cursor is a valid cursor; `BEFORE[1]` is the procedure the window had,
        // stored before ours was set.
        unsafe {
            if message == WM_SETCURSOR && TAKEN.load(Relaxed) && (l & 0xffff) == HTCLIENT {
                SetCursor(std::ptr::null_mut());
                return 1;
            }
            let own: WNDPROC = std::mem::transmute::<isize, WNDPROC>(BEFORE[1].load(Relaxed));
            CallWindowProcW(own, hwnd, message, w, l)
        }
    }

    struct Held {
        taken: bool,
        f9: bool,
        /// The first reading after it is taken: where the pointer was says nothing.
        fresh: bool,
        /// The left button at the last reading.
        left: bool,
        /// The click that took the pointer is still held: no trigger pull.
        click: bool,
        x: f32,
        y: f32,
    }

    impl Held {
        const fn new() -> Held {
            Held { taken: false, f9: false, fresh: true, left: false, click: false, x: 0.0, y: 0.0 }
        }
        fn release(&mut self) {
            (self.taken, self.fresh, self.x, self.y) = (false, true, 0.0, 0.0);
            (self.left, self.click) = (false, false);
        }
    }

    static HELD: Mutex<Held> = Mutex::new(Held::new());
    static SCALE: LazyLock<f32> = LazyLock::new(|| {
        std::env::var("VICTORY_MOUSE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1.0)
    });

    pub(super) fn read(focused: bool) -> Mouse {
        let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
        if !focused {
            held.release();
            TAKEN.store(false, Relaxed);
            return Mouse::default();
        }
        // SAFETY: plain queries, a pointer move inside our own window, and our procedure set
        // before that window's own, which it calls.
        unsafe {
            let f9 = GetAsyncKeyState(0x78) < 0;
            if f9 && !held.f9 {
                (held.taken, held.fresh, held.x, held.y) = (!held.taken, true, 0.0, 0.0);
            }
            held.f9 = f9;
            let hwnd = GetForegroundWindow();
            let mut info: WINDOWINFO = std::mem::zeroed();
            info.cbSize = size_of::<WINDOWINFO>() as u32;
            let mut at = POINT { x: 0, y: 0 };
            if GetWindowInfo(hwnd, &mut info) == 0 || GetCursorPos(&mut at) == 0 {
                TAKEN.store(false, Relaxed);
                return Mouse::default();
            }
            let picture = info.rcClient;
            let (left, right) = (GetAsyncKeyState(0x01) < 0, GetAsyncKeyState(0x02) < 0);
            if !held.taken {
                TAKEN.store(false, Relaxed);
                let over = at.x >= picture.left
                    && at.x < picture.right
                    && at.y >= picture.top
                    && at.y < picture.bottom;
                let clicked = left && !held.left && over;
                held.left = left;
                if !clicked {
                    return Mouse::default();
                }
                (held.taken, held.fresh, held.click, held.x, held.y) = (true, true, true, 0.0, 0.0);
            }
            held.left = left;
            held.click &= left;
            let left = left && !held.click;
            if !TAKEN.swap(true, Relaxed) && BEFORE[0].load(Relaxed) == 0 {
                let own = GetWindowLongPtrW(hwnd, GWLP_WNDPROC);
                if own != 0 {
                    BEFORE[1].store(own, Relaxed);
                    BEFORE[0].store(hwnd as isize, Relaxed);
                    SetWindowLongPtrW(hwnd, GWLP_WNDPROC, hidden as *const () as usize as isize);
                }
            }
            let middle = ((picture.left + picture.right) / 2, (picture.top + picture.bottom) / 2);
            let (dx, dy) = if held.fresh { (0, 0) } else { (at.x - middle.0, at.y - middle.1) };
            held.fresh = false;
            SetCursorPos(middle.0, middle.1);
            held.x = smooth(held.x, dx, *SCALE);
            held.y = smooth(held.y, dy, *SCALE);
            reading(left, right, held.x, held.y)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movement_builds_up_and_decays() {
        let a = smooth(0.0, 8, 1.0);
        assert!((a - 0.2).abs() < 1e-6);
        assert!((smooth(a, 0, 1.0) - 0.16).abs() < 1e-6);
        assert_eq!(smooth(0.0, 8, 0.0), 0.0);
        assert!((smooth(0.0, 8, 2.0) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn reading_clamps_and_puts_up_positive() {
        let m = reading(true, false, 3.0, 0.5);
        assert_eq!((m.left, m.right, m.x, m.y), (true, false, 1.0, -0.5));
        let m = reading(false, true, -3.0, -3.0);
        assert_eq!((m.x, m.y), (-1.0, 1.0));
    }
}
