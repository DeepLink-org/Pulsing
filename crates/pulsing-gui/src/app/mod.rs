mod chat;
mod left;
mod right;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui;
use pulsing_forge::InteractiveConfig;

use crate::controller::ForgeController;
use crate::model::{
    build_file_tree, count_files, FileTreeNode, SessionId, SessionStore, WorkspaceAction,
    WorkspaceModel,
};
use crate::settings::{build_agent_config, ChatMode};
use crate::state::{ChatMessage, MessageKind};

const MAX_PREVIEW_BYTES: usize = 512 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeftTab {
    Explorer,
    Revisions,
    Workflows,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CenterTab {
    Chat,
    File(usize),
}

struct OpenFile {
    rel: PathBuf,
    content: String,
    error: Option<String>,
}

pub struct WorkspaceApp {
    agent: InteractiveConfig,
    workspace: WorkspaceModel,
    sessions: SessionStore,
    chat_mode: ChatMode,

    left_tab: LeftTab,
    center_tab: CenterTab,
    file_tree: Vec<FileTreeNode>,
    open_files: Vec<OpenFile>,
    last_file_click: Option<(String, Instant)>,

    input_text: String,
    forge: ForgeController,
    toast: Option<(String, Instant)>,
}

impl WorkspaceApp {
    fn new(agent: InteractiveConfig) -> Self {
        let workspace = WorkspaceModel::new(agent.cwd.clone());
        let sessions =
            SessionStore::new(agent.provider.clone(), agent.model.clone(), ChatMode::Agent);
        let mut app = Self {
            agent,
            workspace,
            sessions,
            chat_mode: ChatMode::Agent,
            left_tab: LeftTab::Explorer,
            center_tab: CenterTab::Chat,
            file_tree: Vec::new(),
            open_files: Vec::new(),
            last_file_click: None,
            input_text: String::new(),
            forge: ForgeController::start(),
            toast: None,
        };
        app.rebuild_tree();
        app
    }

    fn rebuild_tree(&mut self) {
        self.file_tree = build_file_tree(&self.workspace.layout, &self.file_tree);
        self.workspace.runtime.file_count = count_files(&self.file_tree);
    }

    fn dispatch(&mut self, action: WorkspaceAction) {
        match action {
            WorkspaceAction::OpenFile(rel) => self.open_file(rel),
            WorkspaceAction::CloseFile(rel) => self.close_file(&rel),
            WorkspaceAction::RefreshExplorer => self.rebuild_tree(),
            WorkspaceAction::NewSession => self.new_session(),
            WorkspaceAction::FocusSession(id) => {
                self.sessions.focus(id);
                let (provider, model, mode) = self.sessions.active_settings();
                self.agent.provider = provider.to_string();
                self.agent.model = model.to_string();
                self.chat_mode = mode;
                self.center_tab = CenterTab::Chat;
                self.toast = Some((
                    format!("Session: {}", self.sessions.session_title(id)),
                    Instant::now(),
                ));
            }
            WorkspaceAction::RefreshRevisions => {
                self.workspace.refresh_revisions();
                self.toast = Some(("Revisions refreshed".into(), Instant::now()));
            }
            WorkspaceAction::RefreshWorkflows => {
                self.workspace.refresh_workflows();
                self.toast = Some(("Workflows refreshed".into(), Instant::now()));
            }
        }
        self.sync_runtime();
    }

    fn new_session(&mut self) {
        self.sessions.new_session(
            self.agent.provider.clone(),
            self.agent.model.clone(),
            self.chat_mode,
        );
        self.center_tab = CenterTab::Chat;
        self.toast = Some(("New chat started".into(), Instant::now()));
        self.sync_runtime();
    }

    fn open_file(&mut self, rel: PathBuf) {
        if let Some(ix) = self.open_files.iter().position(|f| f.rel == rel) {
            self.center_tab = CenterTab::File(ix);
            return;
        }
        let abs = self.workspace.layout.root.join(&rel);
        let (content, error) = load_preview(&rel, &abs);
        let preview_err = error.clone();
        self.open_files.push(OpenFile {
            rel: rel.clone(),
            content,
            error,
        });
        self.center_tab = CenterTab::File(self.open_files.len() - 1);
        if preview_err.is_none() {
            self.toast = Some((format!("Opened {}", rel.display()), Instant::now()));
        }
    }

    fn close_file(&mut self, rel: &PathBuf) {
        if let Some(ix) = self.open_files.iter().position(|f| &f.rel == rel) {
            self.open_files.remove(ix);
            self.center_tab = match self.center_tab {
                CenterTab::File(i) if i == ix => CenterTab::Chat,
                CenterTab::File(i) if i > ix => CenterTab::File(i - 1),
                other => other,
            };
        }
    }

    fn sync_runtime(&mut self) {
        let chat = self.sessions.active_chat();
        self.workspace.set_busy(chat.busy);
        self.workspace.set_session_title(chat.session_title());
    }

    fn poll_agent_events(&mut self) {
        let mut changed = false;
        while let Some(event) = self.forge.try_recv() {
            let session_id = SessionId(event.gui_session_id);
            if let Some(chat) = self.sessions.chat_mut(session_id) {
                chat.apply(event.event);
                changed = true;
            } else if event.gui_session_id == 0 {
                self.toast = Some(("Forge controller failed".into(), Instant::now()));
            }
        }
        if changed {
            self.sync_runtime();
        }
    }

    fn try_send(&mut self) {
        let text = self.input_text.trim().to_string();
        if text.is_empty() || self.sessions.active_chat().busy {
            return;
        }
        self.input_text.clear();

        let chat = self.sessions.active_chat_mut();
        chat.messages.push(ChatMessage {
            kind: MessageKind::User(text.clone()),
        });
        chat.messages.push(ChatMessage {
            kind: MessageKind::Assistant {
                body: String::new(),
                streaming: true,
            },
        });
        chat.busy = true;
        self.sync_runtime();

        let session_id = self.sessions.active_id();
        self.forge.start_turn(
            session_id.0,
            build_agent_config(
                &self.agent.cwd,
                &self.agent.provider,
                &self.agent.model,
                self.chat_mode,
            ),
            text,
        );
    }

    fn stop_generation(&mut self) {
        if !self.sessions.active_chat().busy {
            return;
        }
        let session_id = self.sessions.active_id();
        self.forge.cancel_turn(session_id.0);
        self.toast = Some(("Cancellation requested".into(), Instant::now()));
    }

    fn set_model(&mut self, provider: &str, model: &str) {
        if self.sessions.active_chat().busy {
            return;
        }
        if !self.sessions.active_chat().messages.is_empty() {
            self.agent.provider = provider.into();
            self.agent.model = model.into();
            self.sessions.new_session(
                self.agent.provider.clone(),
                self.agent.model.clone(),
                self.chat_mode,
            );
            self.toast = Some(("Model change started a new chat".into(), Instant::now()));
            self.sync_runtime();
            return;
        }
        self.agent.provider = provider.into();
        self.agent.model = model.into();
        self.sessions.set_active_settings(
            self.agent.provider.clone(),
            self.agent.model.clone(),
            self.chat_mode,
        );
    }

    fn set_chat_mode(&mut self, mode: ChatMode) {
        if self.sessions.active_chat().busy {
            return;
        }
        if !self.sessions.active_chat().messages.is_empty() {
            self.chat_mode = mode;
            self.sessions.new_session(
                self.agent.provider.clone(),
                self.agent.model.clone(),
                self.chat_mode,
            );
            self.toast = Some(("Mode change started a new chat".into(), Instant::now()));
            self.sync_runtime();
            return;
        }
        self.chat_mode = mode;
        self.sessions.set_active_settings(
            self.agent.provider.clone(),
            self.agent.model.clone(),
            self.chat_mode,
        );
    }

    fn on_file_click(&mut self, id: &str, is_dir: bool) {
        if is_dir {
            toggle_dir(&mut self.file_tree, id);
            self.rebuild_tree();
            return;
        }
        let now = Instant::now();
        let double = self.last_file_click.as_ref().is_some_and(|(last, t)| {
            last == id && now.duration_since(*t) < Duration::from_millis(400)
        });
        self.last_file_click = Some((id.to_string(), now));
        self.workspace.set_selected_file(Some(PathBuf::from(id)));
        if double {
            self.dispatch(WorkspaceAction::OpenFile(PathBuf::from(id)));
        }
    }
}

pub fn run(agent: InteractiveConfig) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Pulsing"),
        ..Default::default()
    };
    eframe::run_native(
        "Pulsing",
        options,
        Box::new(move |_cc| Ok(Box::new(WorkspaceApp::new(agent)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

impl eframe::App for WorkspaceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_agent_events();
        if self.sessions.any_busy() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }

        if let Some((msg, t0)) = &self.toast {
            if t0.elapsed() > Duration::from_secs(3) {
                self.toast = None;
            } else {
                egui::TopBottomPanel::top("toast").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(msg);
                    });
                });
            }
        }

        let workspace_name = self
            .workspace
            .layout
            .root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.workspace.layout.root.display().to_string());
        let busy = self.workspace.runtime.local_busy;
        let session_title = self.workspace.runtime.session_title.clone();

        egui::TopBottomPanel::top("titlebar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Pulsing");
                ui.separator();
                ui.label(&workspace_name);
                ui.separator();
                let status = if busy {
                    "● Agent running"
                } else {
                    "● Ready"
                };
                ui.colored_label(
                    if busy {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::GRAY
                    },
                    status,
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&session_title);
                });
            });
        });

        egui::TopBottomPanel::bottom("runtime").show(ctx, |ui| {
            let r = &self.workspace.runtime;
            ui.horizontal(|ui| {
                ui.label(if busy { "Running" } else { "Ready" });
                ui.separator();
                ui.label(format!("files {}", r.file_count));
                ui.label(format!("revs {}", r.revision_count));
                ui.label(format!("workflows {}", r.workflow_count));
                ui.label("cluster · phase 3");
            });
        });

        egui::SidePanel::left("left")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                left::render(self, ui);
            });

        egui::SidePanel::right("agents")
            .resizable(true)
            .default_width(220.0)
            .show(ctx, |ui| {
                right::render(self, ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_center(ui);
        });
    }
}

impl WorkspaceApp {
    fn render_center(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.center_tab == CenterTab::Chat, "Chat")
                .clicked()
            {
                self.center_tab = CenterTab::Chat;
            }
            let tabs: Vec<_> = self
                .open_files
                .iter()
                .enumerate()
                .map(|(ix, file)| {
                    let name = file
                        .rel
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| file.rel.display().to_string());
                    (ix, name, file.rel.clone())
                })
                .collect();
            for (ix, name, rel) in tabs {
                let tab = CenterTab::File(ix);
                let selected = self.center_tab == tab;
                ui.horizontal(|ui| {
                    if ui.selectable_label(selected, &name).clicked() {
                        self.center_tab = tab;
                    }
                    if ui.small_button("×").clicked() {
                        self.dispatch(WorkspaceAction::CloseFile(rel));
                    }
                });
            }
        });
        ui.separator();

        match self.center_tab {
            CenterTab::Chat => chat::render(self, ui),
            CenterTab::File(ix) => {
                if let Some(file) = self.open_files.get(ix) {
                    if let Some(err) = &file.error {
                        ui.colored_label(egui::Color32::RED, format!("Preview failed: {err}"));
                    } else {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.add(egui::Label::new(&file.content).wrap());
                        });
                    }
                }
            }
        }
    }
}

fn toggle_dir(nodes: &mut [FileTreeNode], id: &str) -> bool {
    for node in nodes {
        if node.id == id && node.is_dir {
            node.expanded = !node.expanded;
            return true;
        }
        if toggle_dir(&mut node.children, id) {
            return true;
        }
    }
    false
}

fn load_preview(rel_path: &Path, abs_path: &Path) -> (String, Option<String>) {
    let meta = match std::fs::metadata(abs_path) {
        Ok(m) => m,
        Err(err) => return (String::new(), Some(err.to_string())),
    };
    if !meta.is_file() {
        return (String::new(), Some("Not a file".into()));
    }
    if meta.len() as usize > MAX_PREVIEW_BYTES {
        return (
            String::new(),
            Some(format!(
                "File too large to preview ({} KB max)",
                MAX_PREVIEW_BYTES / 1024
            )),
        );
    }
    let bytes = match std::fs::read(abs_path) {
        Ok(b) => b,
        Err(err) => return (String::new(), Some(err.to_string())),
    };
    if bytes.iter().take(8192).any(|b| *b == 0) {
        return (
            String::new(),
            Some("Binary file cannot be previewed".into()),
        );
    }
    let content = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return (String::new(), Some("Invalid UTF-8 text file".into())),
    };
    let text = if is_markdown(rel_path) {
        content
    } else {
        wrap_code_block(language_for_path(rel_path), &content)
    };
    (text, None)
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("md" | "markdown")
    )
}

fn language_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("toml") => "toml",
        Some("json") => "json",
        Some("yaml" | "yml") => "yaml",
        Some("js") => "javascript",
        Some("ts") => "typescript",
        Some("sh" | "bash" | "zsh") => "bash",
        _ => "text",
    }
}

fn wrap_code_block(lang: &str, code: &str) -> String {
    let fence = if code.contains("```") { "````" } else { "```" };
    format!("{fence}{lang}\n{code}\n{fence}")
}

// Re-export helpers for submodules
impl WorkspaceApp {
    pub(crate) fn agent(&self) -> &InteractiveConfig {
        &self.agent
    }
    pub(crate) fn workspace(&self) -> &WorkspaceModel {
        &self.workspace
    }
    pub(crate) fn sessions(&self) -> &SessionStore {
        &self.sessions
    }
    pub(crate) fn chat_mode(&self) -> ChatMode {
        self.chat_mode
    }
    pub(crate) fn left_tab(&self) -> LeftTab {
        self.left_tab
    }
    pub(crate) fn set_left_tab(&mut self, tab: LeftTab) {
        self.left_tab = tab;
    }
    pub(crate) fn file_tree(&self) -> &[FileTreeNode] {
        &self.file_tree
    }
    pub(crate) fn input_text(&self) -> &str {
        &self.input_text
    }
    pub(crate) fn input_text_mut(&mut self) -> &mut String {
        &mut self.input_text
    }
    pub(crate) fn dispatch_action(&mut self, action: WorkspaceAction) {
        self.dispatch(action);
    }
    pub(crate) fn on_tree_click(&mut self, id: &str, is_dir: bool) {
        self.on_file_click(id, is_dir);
    }
    pub(crate) fn send_message(&mut self) {
        self.try_send();
    }
    pub(crate) fn stop_agent(&mut self) {
        self.stop_generation();
    }
    pub(crate) fn pick_model(&mut self, provider: &str, model: &str) {
        self.set_model(provider, model);
    }
    pub(crate) fn pick_mode(&mut self, mode: ChatMode) {
        self.set_chat_mode(mode);
    }
}
