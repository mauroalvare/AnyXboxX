//! The "AnyXboxX folder" card on the Setup page: which clone the launcher uses, where it came
//! from, and a button to choose another one.

use eframe::egui::Ui;

use crate::app::App;
use crate::{theme, tokens, widgets};

pub fn section(app: &mut App, ui: &mut Ui) {
    theme::card(ui, |ui| {
        ui.label(tokens::subtitle("AnyXboxX folder"));
        ui.add_space(tokens::XS);
        match (&app.repo, app.repo_source) {
            (Some(dir), Some(source)) => {
                ui.label(tokens::body(dir.display().to_string()));
                ui.label(tokens::caption(format!("Source: {}.", source.label())));
            }
            _ => {
                ui.label(tokens::body("Not found.").color(theme::BAD));
                ui.label(tokens::caption(
                    "Put anyxboxx-launcher.exe inside your AnyXboxX clone, or choose the clone \
                     folder here.",
                ));
            }
        }
        ui.label(tokens::caption(
            "The clone is the folder holding third_party\\victory. The launcher builds and \
             starts VIctory from it.",
        ));
        ui.add_space(tokens::S);
        ui.horizontal(|ui| {
            if widgets::secondary_button(ui, true, "Browse...").clicked() {
                app.pick_repo_folder();
            }
            let chosen = app.settings.repo_path().is_some();
            if chosen && widgets::secondary_button(ui, true, "Search again").clicked() {
                app.clear_repo_folder();
            }
        });
    });
}
