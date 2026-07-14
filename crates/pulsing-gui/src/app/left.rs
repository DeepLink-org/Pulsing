use eframe::egui;

use super::{LeftTab, WorkspaceApp};
use crate::model::{FileTreeNode, WorkspaceAction};

pub fn render(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        for (tab, label) in [
            (LeftTab::Explorer, "Files"),
            (LeftTab::Revisions, "History"),
            (LeftTab::Workflows, "Workflows"),
        ] {
            if ui.selectable_label(app.left_tab() == tab, label).clicked() {
                app.set_left_tab(tab);
            }
        }
    });
    ui.separator();

    match app.left_tab() {
        LeftTab::Explorer => render_explorer(app, ui),
        LeftTab::Revisions => render_revisions(app, ui),
        LeftTab::Workflows => render_workflows(app, ui),
    }
}

fn render_explorer(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        if ui.button("Open").clicked() {
            if let Some(rel) = app.workspace().selected_file.clone() {
                app.dispatch_action(WorkspaceAction::OpenFile(rel));
            }
        }
        if ui.button("Refresh").clicked() {
            app.dispatch_action(WorkspaceAction::RefreshExplorer);
        }
    });
    if let Some(rel) = &app.workspace().selected_file {
        ui.label(
            egui::RichText::new(rel.display().to_string())
                .small()
                .weak(),
        );
    }
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        let tree = app.file_tree().to_vec();
        for node in &tree {
            render_tree_node(app, ui, node, 0);
        }
    });
}

fn render_tree_node(app: &mut WorkspaceApp, ui: &mut egui::Ui, node: &FileTreeNode, depth: usize) {
    let indent = depth as f32 * 14.0;
    ui.horizontal(|ui| {
        ui.add_space(indent);
        if node.is_dir {
            let icon = if node.expanded { "📂" } else { "📁" };
            if ui
                .selectable_label(false, format!("{icon} {}", node.label))
                .clicked()
            {
                app.on_tree_click(&node.id, true);
            }
        } else if ui
            .selectable_label(false, format!("📄 {}", node.label))
            .clicked()
        {
            app.on_tree_click(&node.id, false);
        }
    });
    if node.is_dir && node.expanded {
        for child in &node.children {
            render_tree_node(app, ui, child, depth + 1);
        }
    }
}

fn render_revisions(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    if ui.button("Refresh").clicked() {
        app.dispatch_action(WorkspaceAction::RefreshRevisions);
    }
    ui.separator();

    let snapshot = app.workspace().revisions.clone();
    if snapshot.revisions.is_empty() {
        ui.label("No checkpoints yet");
        ui.label(
            egui::RichText::new("Use `pulsing checkpoint` or the agent /checkpoint command.")
                .small()
                .weak(),
        );
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        for rev in &snapshot.revisions {
            let is_head = snapshot.head.as_deref() == Some(rev.id.as_str());
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    if is_head {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "HEAD");
                    }
                    ui.strong(&rev.id);
                    ui.label(format!("{} files", rev.file_count));
                });
                ui.label(egui::RichText::new(&rev.message).small().weak());
            });
        }
    });
}

fn render_workflows(app: &mut WorkspaceApp, ui: &mut egui::Ui) {
    if ui.button("Refresh").clicked() {
        app.dispatch_action(WorkspaceAction::RefreshWorkflows);
    }
    ui.separator();

    let scripts = app.workspace().workflow_scripts.clone();
    if scripts.is_empty() {
        ui.label("No workflow scripts");
        ui.label(
            egui::RichText::new(
                "Add `.py` files under `.pulsing/workflows/` (run `pulsing init`).",
            )
            .small()
            .weak(),
        );
        return;
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        for path in &scripts {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            ui.group(|ui| {
                ui.strong(name);
                ui.label(egui::RichText::new("Python workflow").small().weak());
            });
        }
    });
}
