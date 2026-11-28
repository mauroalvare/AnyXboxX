//! The optional file manifest on the Game folder page: a button, then a progress bar with speed
//! and time left, a Cancel button, and the result.

use eframe::egui::{self, Ui};

use crate::app::App;
use crate::app_check::ManifestState;
use crate::manifest::manifest_text;
use crate::{bar, theme, tokens, widgets};

pub fn section(app: &mut App, ui: &mut Ui) {
    ui.label(tokens::subtitle("File manifest (optional)"));
    ui.add_space(tokens::XS);
    ui.label(tokens::muted(
        "Only to check that a copy of the game is intact: you do not need it to play. It reads \
             every byte (about 20 minutes for 130 GB, then remembered) to compute the SHA-256 of \
             the required files and the .rpf archives.",
    ));
    ui.add_space(tokens::M);
    match app.manifest.clone() {
        ManifestState::Running(progress) => {
            bar::gradient_bar(ui, Some(progress.fraction()), &progress.describe());
            ui.add_space(tokens::S);
            if widgets::secondary_button(ui, true, "Cancel").clicked() {
                app.cancel_manifest();
            }
        }
        ManifestState::Done(entries) => {
            let text = manifest_text(&entries);
            ui.collapsing("Show the manifest", |ui| {
                let mut shown = text.clone();
                ui.add(
                    egui::TextEdit::multiline(&mut shown)
                        .desired_rows(6)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace),
                );
                if ui.button("Copy manifest").clicked() {
                    ui.ctx().copy_text(text.clone());
                }
            });
            start_button(app, ui, "Compute again");
        }
        ManifestState::Failed(error) => {
            ui.label(tokens::body(format!("The manifest failed: {error}")).color(theme::BAD));
            start_button(app, ui, "Try again");
        }
        ManifestState::Cancelled => {
            ui.label("Cancelled. Files already hashed are remembered.");
            start_button(app, ui, "Compute file manifest");
        }
        ManifestState::NotRun => start_button(app, ui, "Compute file manifest"),
    }
}

fn start_button(app: &mut App, ui: &mut Ui, label: &str) {
    if widgets::secondary_button(ui, true, label).clicked() {
        app.start_manifest();
    }
}
