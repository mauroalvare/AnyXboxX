use std::path::Path;

use eframe::egui::{self, Ui};

use crate::app::App;
use crate::dlss::{MODES, PRESETS};
use crate::picker::Pick;
use crate::{streamline, theme, tokens};

pub fn card(app: &mut App, ui: &mut Ui) {
    theme::card(ui, |ui| {
        ui.label(tokens::subtitle("NVIDIA DLSS"));
        ui.add_space(tokens::S);
        let options = &mut app.settings.options.dlss;
        ui.checkbox(&mut options.super_resolution, "DLSS Super Resolution");
        ui.add_enabled_ui(options.super_resolution, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Mode");
                for &(mode, label) in MODES {
                    if ui.selectable_label(options.mode == mode, label).clicked() {
                        options.mode = mode.into();
                    }
                }
            });
            egui::ComboBox::from_label("DLSS model preset")
                .selected_text(&options.preset)
                .show_ui(ui, |ui| {
                    for &preset in PRESETS {
                        ui.selectable_value(&mut options.preset, preset.into(), preset);
                    }
                });
        });
        ui.add_space(tokens::S);
        let available = app.framegen_unavailable.is_none();
        let mut enabled = available && options.framegen;
        if ui
            .add_enabled(
                available,
                egui::Checkbox::new(&mut enabled, "Frame Generation 2x"),
            )
            .changed()
        {
            options.framegen = enabled;
        }
        if let Some(reason) = app.framegen_unavailable {
            ui.label(tokens::caption(reason));
        }
        ui.add_enabled(
            enabled,
            egui::Checkbox::new(
                &mut options.hudless,
                "Exclude the HUD from generated motion",
            ),
        );
        ui.add_space(tokens::S);
        ui.label("Streamline folder");
        let browse = ui
            .horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut options.streamline_dir)
                        .hint_text(r"C:\NVIDIA\Streamline\bin\x64"),
                );
                ui.button("Browse…").clicked()
            })
            .inner;
        let path = options.streamline_dir.trim().trim_matches('"');
        if !path.is_empty() || enabled || options.super_resolution {
            let refresh = ui.small_button("Recheck folder").clicked();
            if refresh
                || app
                    .streamline_check
                    .as_ref()
                    .is_none_or(|(last, _)| last != path)
            {
                app.streamline_check = Some((path.into(), streamline::check(Path::new(path))));
            }
            if let Some((_, result)) = &app.streamline_check {
                match result {
                    Ok(()) => ui.label("sl.interposer.dll is signed by NVIDIA Corporation."),
                    Err(reason) => ui.colored_label(theme::WARN, reason),
                };
            }
        }
        if browse {
            app.open_dialog(Pick::Streamline);
        }
        if ui
            .button("Télécharger les fichiers NVIDIA")
            .on_hover_text("Open NVIDIA’s official Streamline releases in your browser")
            .clicked()
        {
            ui.ctx()
                .open_url(egui::OpenUrl::new_tab(streamline::RELEASES));
        }
    });
}
