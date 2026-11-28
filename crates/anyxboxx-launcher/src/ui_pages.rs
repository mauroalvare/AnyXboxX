//! The four pages of the window: setup checklist, game folder, shader preloading, play.

use eframe::egui::{self, vec2};

use crate::app::{App, Verify};
use crate::setup::State;
use crate::verify::{self, Problem};
use crate::widgets::{self, Status};
use crate::{bar, launch_progress, session, theme, tokens, ui_options, ui_states};
use anyxboxx::run::XgMode;

/// A page's title and the sentence under it.
pub fn header(ui: &mut egui::Ui, title: &str, description: &str) {
    ui.label(tokens::title(title));
    ui.add_space(tokens::XS);
    ui.label(tokens::muted(description));
    ui.add_space(tokens::L);
}

/// Page 1: the checklist, with official links only.
pub fn setup(app: &mut App, ui: &mut egui::Ui) {
    header(
        ui,
        "Setup",
        "What this PC needs. A check means the launcher found it; an exclamation mark means it cannot tell.",
    );
    crate::ui_repo::section(app, ui);
    ui.add_space(tokens::S);
    for item in &app.setup {
        theme::card(ui, |ui| {
            let (status, text) = match &item.state {
                State::Ok(text) => (Status::Good, format!("Found: {text}")),
                State::Missing(text) => (Status::Bad, format!("Missing: {text}")),
                State::Unknown(text) => (Status::Warn, format!("Unknown: {text}")),
            };
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(vec2(28.0, 28.0), egui::Sense::hover());
                widgets::status_icon(ui.painter(), rect.center(), 12.0, status);
                ui.add_space(tokens::S);
                ui.vertical(|ui| {
                    ui.label(tokens::subtitle(item.title));
                    ui.label(tokens::body(text).color(status.color()));
                    ui.label(tokens::caption(format!("Needed for: {}", item.needed_for)));
                    if let Some((label, url)) = item.link {
                        ui.hyperlink_to(label, url);
                    }
                });
            });
        });
        ui.add_space(tokens::S);
    }
    ui.add_space(tokens::S);
    theme::card(ui, |ui| {
        ui.label(tokens::subtitle("This PC"));
        ui.add_space(tokens::XS);
        ui.label(tokens::body(format!(
            "Host: {:?}. GPU: {} ({}).",
            app.host,
            app.gpu.label(),
            app.adapter_name.as_deref().unwrap_or("no adapter found")
        )));
        ui.label(tokens::body(format!(
            "Profile: {} ({:?}).",
            app.profile().name,
            app.profile().status
        )));
        ui.add_space(tokens::S);
        if widgets::secondary_button(ui, true, "Check again").clicked() {
            app.refresh_setup();
        }
    });
}

/// Page 2: the game folder, its check, and the manifest.
pub fn files(app: &mut App, ui: &mut egui::Ui) {
    header(
        ui,
        "Game folder",
        "The folder holding game_scarlett_master.exe, outside this repository. Only its own files are read.",
    );
    if app.settings.game_path().is_none() && !app.is_busy() {
        theme::card(ui, |ui| ui_states::empty_folder(app, ui));
        return;
    }
    theme::card(ui, |ui| {
        ui.label(tokens::subtitle("Folder"));
        ui.add_space(tokens::S);
        ui.horizontal(|ui| {
            let room = (ui.available_width() - 260.0).max(160.0);
            ui.add(egui::TextEdit::singleline(&mut app.settings.game_dir).desired_width(room));
            if widgets::secondary_button(ui, true, "Browse...").clicked() {
                app.pick_game_folder();
            }
            let idle = !app.is_busy();
            if widgets::glow_button(ui, idle, "Check folder", vec2(0.0, 36.0), false).clicked() {
                app.start_verify();
            }
        });
        ui.label(tokens::caption(
            "You can also drop the folder on this window.",
        ));
        ui.add_space(tokens::M);
        folder_state(app, ui);
    });
    ui.add_space(tokens::M);
    theme::card(ui, |ui| crate::ui_manifest::section(app, ui));
}

/// The result of the folder check, as one of the page states.
fn folder_state(app: &mut App, ui: &mut egui::Ui) {
    match app.verify.clone() {
        Verify::NotRun => {
            ui.label(tokens::muted(
                "Not checked yet. Check folder reads the file headers.",
            ));
        }
        Verify::Running => ui_states::loading(ui, "Checking the files (a few seconds)..."),
        Verify::Failed(error) => {
            let log = app.log_text();
            ui_states::outcome_card(ui, Status::Bad, "Cannot check the folder", &error, |ui| {
                if widgets::secondary_button(ui, true, "Copy log").clicked() {
                    ui.ctx().copy_text(log);
                }
            });
        }
        Verify::Done(report) if report.is_ok() => ui_states::notice_card(
            ui,
            Status::Good,
            "Folder OK",
            "Every required file is present and its headers are sound.",
        ),
        Verify::Done(report) => {
            let lines: Vec<String> = report
                .problems
                .iter()
                .map(|(name, problem)| {
                    let text = match problem {
                        Problem::Missing => "missing".to_string(),
                        Problem::Corrupted(reason) => format!("corrupted: {reason}"),
                        Problem::Unreadable(reason) => format!("unreadable: {reason}"),
                    };
                    format!("{name}: {text}")
                })
                .collect();
            let message = lines.join("\n");
            ui_states::outcome_card(ui, Status::Bad, "Files to fix", &message, |ui| {
                if widgets::secondary_button(ui, true, "Copy the list").clicked() {
                    ui.ctx().copy_text(message.clone());
                }
            });
            ui.add_space(tokens::S);
            ui.collapsing("Required files", |ui| {
                for (name, reason) in verify::required_files() {
                    ui.label(tokens::caption(format!("{name}: {reason}")));
                }
            });
        }
    }
}

/// Page 3: shader preloading (VIctory's translate into the cache), with its progress.
pub fn shaders(app: &mut App, ui: &mut egui::Ui) {
    header(
        ui,
        "Shader preloading",
        "Translates shaders.rpf and the game's two DLLs into VIctory's cache, so the first launch does not wait.",
    );
    theme::card(ui, |ui| {
        let (status, text) = if app.cache_ready {
            (Status::Good, "The cache holds shaders")
        } else {
            (Status::Warn, "The cache is empty")
        };
        widgets::chip(
            ui,
            status,
            text,
            "The cache depends on the settings: change them, and it is filled again.",
        );
        ui.add_space(tokens::M);
        ui.label(tokens::body(
            "Several minutes the first time. The cache depends on the settings: change them, and the cache is filled again.",
        ));
        ui.add_space(tokens::M);
        ui.horizontal(|ui| {
            ui.label(tokens::body("Worker threads (0 = every core)"));
            ui.add(egui::DragValue::new(&mut app.settings.jobs).range(0..=64));
        });
        ui.add_space(tokens::M);
        let ready = matches!(&app.verify, Verify::Done(report) if report.is_ok());
        if !ready {
            ui.label(tokens::body("Check the game folder first (page 2).").color(theme::WARN));
            ui.add_space(tokens::S);
        }
        let enabled = ready && !app.is_busy();
        if widgets::glow_button(ui, enabled, "Warm now", vec2(160.0, 40.0), false).clicked() {
            app.start(true);
        }
        if let Some(progress) = app.progress() {
            ui.add_space(tokens::M);
            bar::gradient_bar(
                ui,
                Some(progress.fraction()),
                &launch_progress::count_line(progress),
            );
        }
    });
}

/// Page 4: the launch, the profile, the XG library, the advanced panel and the cache.
pub fn play(app: &mut App, ui: &mut egui::Ui) {
    ui_states::banner(app, ui);
    header(
        ui,
        "Play",
        "How the game starts. The choices are kept for the next launch.",
    );
    ui_options::cards(app, ui);
    ui.add_space(tokens::M);
    crate::ui_dlss::card(app, ui);
    ui.add_space(tokens::M);
    theme::card(ui, |ui| crate::ui_play::controls(app, ui));
}

/// The confirmation window of the cache clean.
pub fn confirm_clean(app: &mut App, ctx: &egui::Context) {
    if !app.confirm_clean {
        return;
    }
    let target = app
        .cache_dir
        .as_ref()
        .map_or("unknown folder".to_string(), |d| d.display().to_string());
    let mut close = false;
    egui::Window::new("Clean the shader cache?")
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(tokens::body(format!("This deletes {target}.")));
            ui.label(tokens::muted(
                "The next launches translate the shaders again (minutes). Saves are not touched.",
            ));
            ui.add_space(tokens::S);
            ui.horizontal(|ui| {
                if widgets::glow_button(ui, true, "Delete the cache", vec2(0.0, 36.0), false)
                    .clicked()
                {
                    app.clean_cache();
                }
                if widgets::secondary_button(ui, true, "Cancel").clicked() {
                    close = true;
                }
            });
        });
    if close {
        app.confirm_clean = false;
    }
}

/// The XG library row of the launch card.
pub fn xg_row(app: &mut App, ui: &mut egui::Ui) {
    ui.label(tokens::body("XG library (your own xg_xs.dll, optional)"));
    ui.horizontal(|ui| {
        let room = (ui.available_width() - 120.0).max(160.0);
        ui.add(egui::TextEdit::singleline(&mut app.settings.xg_dll).desired_width(room));
        if widgets::secondary_button(ui, true, "Browse...").clicked() {
            app.pick_xg_library();
        }
    });
    ui.horizontal(|ui| {
        ui.radio_value(&mut app.settings.xg_mode, XgMode::Real, "real");
        ui.radio_value(&mut app.settings.xg_mode, XgMode::Compare, "compare");
    });
}

/// The plan of the launch: whether it can be made, for the advanced panel.
pub fn plan_lines(app: &App) -> Option<Vec<String>> {
    let profile = app.profile().clone();
    session::make_plan(
        &app.settings,
        &app.repo_or_empty(),
        app.victory_exe.as_deref(),
        &profile,
        app.framegen_unavailable.is_none(),
    )
    .ok()
    .map(|plan| crate::env::describe(&plan.env))
}
