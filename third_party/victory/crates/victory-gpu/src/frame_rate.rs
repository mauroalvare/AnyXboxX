use std::time::Duration;

/// The mode requested from Streamline generates one extra frame per rendered frame.
pub(crate) const FG_MULTIPLIER: u32 = 2;

#[derive(Clone, Copy)]
pub(crate) enum FrameGeneration {
    Off,
    Reported(u32),
    Unavailable,
}

/// One meter interval, containing only presents accepted by the swap chain.
#[derive(Default)]
pub(crate) struct FrameCounts {
    pub rendered: u32,
    shown: u64,
    active: bool,
    estimated: bool,
}

impl FrameCounts {
    pub fn record(&mut self, fg: FrameGeneration) {
        self.rendered += 1;
        self.active |= !matches!(fg, FrameGeneration::Off);
        self.shown += u64::from(match fg {
            FrameGeneration::Off => 1,
            FrameGeneration::Reported(shown) => shown,
            FrameGeneration::Unavailable => {
                self.estimated = true;
                FG_MULTIPLIER
            }
        });
    }

    pub fn rates(&self, elapsed: Duration) -> FrameRate {
        FrameRate {
            rendered: f64::from(self.rendered) / elapsed.as_secs_f64(),
            shown: self.shown as f64 / elapsed.as_secs_f64(),
            active: self.active,
            estimated: self.estimated,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct FrameRate {
    pub rendered: f64,
    pub shown: f64,
    pub active: bool,
    estimated: bool,
}

impl FrameRate {
    pub fn rendered_detail(self, precision: usize) -> String {
        format!("({:.precision$} rendered, FG x{FG_MULTIPLIER})", self.rendered)
    }

    pub fn shown_label(self) -> &'static str {
        if self.estimated { "shown est." } else { "shown" }
    }

    pub fn unit(self, fps: &str, precision: usize) -> String {
        if self.active {
            format!("{fps} {} {}", self.shown_label(), self.rendered_detail(precision))
        } else {
            fps.to_owned()
        }
    }

    pub fn caption(self) -> String {
        format!("{:.0} {}", self.shown, self.unit("fps", 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_uses_the_count_reported_over_the_render_interval() {
        let mut counts = FrameCounts::default();
        let elapsed = Duration::from_secs(2);
        for _ in 0..34 {
            counts.record(FrameGeneration::Off);
        }
        assert_eq!(counts.rates(elapsed).caption(), "17 fps");

        let mut counts = FrameCounts::default();
        for _ in 0..34 {
            counts.record(FrameGeneration::Reported(2));
        }
        assert_eq!(counts.rates(elapsed).caption(), "34 fps shown (17 rendered, FG x2)");
        counts.record(FrameGeneration::Reported(0));
        counts.record(FrameGeneration::Reported(1));
        counts.record(FrameGeneration::Off);
        let mixed = counts.rates(elapsed);
        assert_eq!(mixed.shown, 35.0);
        assert_eq!(mixed.rendered, 18.5);
        counts.record(FrameGeneration::Unavailable);
        assert_eq!(counts.rates(elapsed).caption(), "36 fps shown est. (19 rendered, FG x2)");
    }
}
