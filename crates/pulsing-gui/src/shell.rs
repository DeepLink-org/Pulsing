use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use gpui::{div, AppContext, Context, Entity, IntoElement, ParentElement, Render, Styled, Window};
use gpui_component::dock::{DockArea, DockPlacement, PanelView};
use pulsing_forge::InteractiveConfig;

use crate::dock::{configure_dock, WorkspacePanels};
use crate::model::{CenterTabState, SessionId, SessionStore, WorkspaceAction, WorkspaceModel};
use crate::panels::center::FilePreviewPanel;
use crate::panels::{
    AgentsPanel, ChatPanel, ClusterPanel, ExplorerPanel, RevisionsPanel, RuntimeSummaryPanel,
    WorkflowsPanel,
};
use crate::ui::notify;

pub struct WorkspaceShell {
    pub dock_area: Entity<DockArea>,
    pub workspace: Entity<WorkspaceModel>,
    pub sessions: Entity<SessionStore>,
    center_tabs: CenterTabState,
    file_panels: HashMap<PathBuf, Entity<FilePreviewPanel>>,
    pending_actions: Vec<WorkspaceAction>,
}

impl WorkspaceShell {
    pub fn new(agent: InteractiveConfig, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let workspace = cx.new(|_cx| WorkspaceModel::new(agent.cwd.clone()));
        let sessions = cx.new(|_cx| SessionStore::new());
        let dock_area = cx.new(|cx| DockArea::new("workspace", None, window, cx));
        let weak_dock = dock_area.downgrade();
        let weak_shell = cx.entity().downgrade();

        let chat =
            cx.new(|cx| ChatPanel::new(agent, workspace.clone(), sessions.clone(), window, cx));
        let explorer = cx.new(|cx| ExplorerPanel::new(workspace.clone(), weak_shell.clone(), cx));
        let revisions = cx.new(|cx| RevisionsPanel::new(workspace.clone(), weak_shell.clone(), cx));
        let workflows = cx.new(|cx| WorkflowsPanel::new(workspace.clone(), weak_shell.clone(), cx));
        let agents = cx.new(|cx| {
            AgentsPanel::new(workspace.clone(), sessions.clone(), weak_shell.clone(), cx)
        });
        let runtime = cx.new(|cx| RuntimeSummaryPanel::new(workspace.clone(), cx));
        let cluster = cx.new(ClusterPanel::new);

        let panels = WorkspacePanels {
            explorer,
            revisions,
            workflows,
            chat,
            agents,
            runtime,
            cluster,
        };

        dock_area.update(cx, |dock, cx| {
            configure_dock(dock, weak_dock, &panels, window, cx);
        });

        Self {
            dock_area,
            workspace,
            sessions,
            center_tabs: CenterTabState::default(),
            file_panels: HashMap::new(),
            pending_actions: Vec::new(),
        }
    }

    pub fn dispatch(&mut self, action: WorkspaceAction, cx: &mut Context<Self>) {
        self.pending_actions.push(action);
        cx.notify();
    }

    fn process_action(
        &mut self,
        action: WorkspaceAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match action {
            WorkspaceAction::OpenFile(rel) => self.open_file(rel, window, cx),
            WorkspaceAction::CloseFile(rel) => self.unregister_file(&rel, cx),
            WorkspaceAction::RefreshExplorer => {
                // Explorer handles its own refresh; shell just nudges workspace file count.
                cx.notify();
            }
            WorkspaceAction::NewSession => self.new_session(window, cx),
            WorkspaceAction::FocusSession(id) => self.focus_session(id, window, cx),
            WorkspaceAction::RefreshRevisions => {
                self.workspace.update(cx, |model, cx| {
                    model.refresh_revisions();
                    cx.notify();
                });
                notify::success(window, "Revisions refreshed", "Checkpoint list updated", cx);
            }
            WorkspaceAction::RefreshWorkflows => {
                self.workspace.update(cx, |model, cx| {
                    model.refresh_workflows();
                    cx.notify();
                });
                notify::success(window, "Workflows refreshed", "Script list updated", cx);
            }
        }
    }

    fn new_session(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.sessions.read(cx).active_chat().busy {
            return;
        }
        self.sessions.update(cx, |store, cx| {
            store.new_session();
            cx.notify();
        });
        notify::info(window, "New chat", "Started a new session", cx);
        cx.notify();
    }

    fn focus_session(&mut self, id: SessionId, window: &mut Window, cx: &mut Context<Self>) {
        self.sessions.update(cx, |store, cx| {
            store.focus(id);
            cx.notify();
        });
        let title = self.sessions.read(cx).session_title(id);
        notify::info(window, "Session", title, cx);
        cx.notify();
    }

    fn open_file(&mut self, rel: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        if self.center_tabs.is_open(&rel) {
            if let Some(panel) = self.file_panels.get(&rel).cloned() {
                panel.focus_handle(cx).focus(window);
            }
            notify::info(window, "Already open", rel.display().to_string(), cx);
            return;
        }

        let abs = self.workspace.read(cx).layout.root.join(&rel);
        let weak_shell = cx.entity().downgrade();
        let panel = cx.new(|cx| FilePreviewPanel::new(rel.clone(), abs, weak_shell, window, cx));

        if panel.read(cx).error.is_some() {
            let err = panel
                .read(cx)
                .error
                .clone()
                .unwrap_or_else(|| "Preview failed".into());
            notify::error(window, "Cannot open file", err, cx);
            return;
        }

        let arc: Arc<dyn PanelView> = Arc::new(panel.clone());
        self.dock_area.update(cx, |dock, cx| {
            dock.add_panel(arc, DockPlacement::Center, None, window, cx);
        });
        self.center_tabs.register(rel.clone());
        self.file_panels.insert(rel.clone(), panel);
        notify::success(window, "Opened", rel.display().to_string(), cx);
    }

    fn unregister_file(&mut self, rel: &PathBuf, cx: &mut Context<Self>) {
        self.center_tabs.unregister(rel);
        self.file_panels.remove(rel);
        cx.notify();
    }
}

impl Render for WorkspaceShell {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let actions: Vec<_> = self.pending_actions.drain(..).collect();
        for action in actions {
            self.process_action(action, window, cx);
        }
        div().size_full().child(self.dock_area.clone())
    }
}
