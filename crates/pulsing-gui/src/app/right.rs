use eframe::egui;

use super::WorkspaceApp;
use crate::model::WorkspaceAction;

pub fn render(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    let busy = app.workspace().runtime.local_busy;
    let active = app.sessions().active_id();

    ui.heading("Sessions");
    ui.horizontal(|ui| {
        let status = if busy { "Running" } else { "Idle" };
        ui.colored_label(
            if busy {
                egui::Color32::LIGHT_GREEN
            } else {
                egui::Color32::GRAY
            },
            status,
        );
    });
    ui.separator();

    let sessions: Vec<_> = app
        .sessions()
        .sessions()
        .iter()
        .map(|s| (s.id, s.chat.session_title()))
        .collect();

    for (id, title) in sessions {
        let is_active = id == active;
        if ui
            .selectable_label(is_active, format!("🤖 {title}"))
            .clicked()
        {
            app.dispatch_action(WorkspaceAction::FocusSession(id));
        }
    }

    ui.add_space(8.0);
    if ui
        .add_enabled(!busy, egui::Button::new("+ New chat"))
        .clicked()
    {
        app.dispatch_action(WorkspaceAction::NewSession);
    }

    ui.separator();
    ui.label(egui::RichText::new("Cluster").weak());
    ui.add_enabled(false, egui::Button::new("Remote agents (Soon)"));
}
