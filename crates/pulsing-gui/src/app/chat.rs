use eframe::egui;

use super::WorkspaceApp;
use crate::settings::{provider_available, ChatMode, MODEL_PRESETS};
use crate::state::{ChatMessage, MessageKind};

pub fn render(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    let chat = app.sessions().active_chat().clone();
    let busy = chat.busy;

    ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
        render_composer(app, ui, busy);
        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .max_height(ui.available_height())
            .show(ui, |ui| {
                if chat.messages.is_empty() && !busy {
                    render_empty_state(app, ui);
                } else {
                    for msg in &chat.messages {
                        render_message(ui, msg);
                    }
                }
            });
    });
}

fn render_message(ui: &mut egui::Ui, msg: &ChatMessage) {
    match &msg.kind {
        MessageKind::User(text) => {
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    ui.label(egui::RichText::new("You").weak());
                    egui::Frame::group(ui.style())
                        .fill(ui.visuals().widgets.active.bg_fill)
                        .show(ui, |ui| {
                            ui.label(text);
                        });
                });
            });
            ui.add_space(8.0);
        }
        MessageKind::Assistant { body, streaming } => {
            if body.is_empty() && !streaming {
                return;
            }
            let display = if *streaming && body.is_empty() {
                "▍".to_string()
            } else if *streaming {
                format!("{body}▍")
            } else {
                body.clone()
            };
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Agent").weak());
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.label(display);
                });
            });
            ui.add_space(8.0);
        }
        MessageKind::Tool {
            name,
            running,
            ok,
            detail,
        } => {
            let symbol = if *running {
                "…"
            } else if *ok {
                "✓"
            } else {
                "✗"
            };
            let summary = if detail.is_empty() {
                name.clone()
            } else {
                format!("{name} — {detail}")
            };
            ui.label(
                egui::RichText::new(format!("{symbol} {summary}"))
                    .small()
                    .weak(),
            );
            ui.add_space(4.0);
        }
        MessageKind::Error(text) => {
            ui.colored_label(egui::Color32::RED, format!("Error: {text}"));
            ui.add_space(8.0);
        }
    }
}

fn render_empty_state(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    ui.vertical_centered(|ui| {
        ui.add_space(48.0);
        ui.heading("What are we building?");
        ui.label("Ask about your project, edit files, or run agent workflows.");
        ui.add_space(16.0);
        for suggestion in [
            "Explain this codebase",
            "Find bugs in recent changes",
            "Add a new feature",
        ] {
            if ui.button(suggestion).clicked() {
                *app.input_text_mut() = suggestion.to_string();
            }
        }
    });
}

fn render_composer(app: &mut WorkspaceApp, ui: &mut egui::Ui, busy: bool) {
    let can_send = !busy && !app.input_text().trim().is_empty();

    ui.horizontal(|ui| {
        egui::ComboBox::from_label("Mode")
            .selected_text(app.chat_mode().label())
            .show_ui(ui, |ui| {
                for mode in ChatMode::ALL {
                    if ui
                        .selectable_label(app.chat_mode() == mode, mode.label())
                        .clicked()
                    {
                        app.pick_mode(mode);
                    }
                }
            });

        let model_label = short_model_label(&app.agent().model);
        egui::ComboBox::from_label("Model")
            .selected_text(&model_label)
            .show_ui(ui, |ui| {
                let mut last_section = "";
                for preset in MODEL_PRESETS {
                    if preset.section != last_section {
                        ui.separator();
                        ui.label(egui::RichText::new(preset.section).weak());
                        last_section = preset.section;
                    }
                    let available = provider_available(preset.provider);
                    let checked = app.agent().provider == preset.provider
                        && app.agent().model == preset.model;
                    if ui
                        .add_enabled(available, egui::SelectableLabel::new(checked, preset.label))
                        .clicked()
                    {
                        app.pick_model(preset.provider, preset.model);
                    }
                }
            });

        if busy {
            if ui.button("Stop").clicked() {
                app.stop_agent();
            }
        } else if ui
            .add_enabled(can_send, egui::Button::new("Send"))
            .clicked()
        {
            app.send_message();
        }
    });

    ui.label(egui::RichText::new(app.chat_mode().hint()).small().weak());

    let response = ui.add(
        egui::TextEdit::multiline(app.input_text_mut())
            .desired_rows(3)
            .hint_text("Ask anything about your project…"),
    );
    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift)
    {
        app.send_message();
    }

    ui.label(
        egui::RichText::new("Enter to send · Shift+Enter for newline")
            .small()
            .weak(),
    );
}

fn short_model_label(model: &str) -> String {
    let m = model.trim();
    if m.len() <= 22 {
        return m.to_string();
    }
    format!("{}…", m.chars().take(21).collect::<String>())
}
