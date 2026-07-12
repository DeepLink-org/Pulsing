use std::path::{Path, PathBuf};

use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, IntoElement, ParentElement, Render,
    SharedString, Styled, WeakEntity, Window,
};
use gpui_component::{
    alert::Alert,
    dock::{Panel, PanelEvent},
    text::TextView,
    v_flex,
};

use crate::dock::layout::PANEL_FILE_PREVIEW;
use crate::model::WorkspaceAction;
use crate::shell::WorkspaceShell;

const MAX_PREVIEW_BYTES: usize = 512 * 1024;

pub struct FilePreviewPanel {
    focus_handle: FocusHandle,
    rel_path: PathBuf,
    preview_id: SharedString,
    file_name: SharedString,
    markdown: SharedString,
    pub(crate) error: Option<String>,
    shell: WeakEntity<WorkspaceShell>,
}

impl FilePreviewPanel {
    pub fn new(
        rel_path: PathBuf,
        abs_path: PathBuf,
        shell: WeakEntity<WorkspaceShell>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let file_name: SharedString = rel_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| rel_path.display().to_string())
            .into();

        let preview_id: SharedString = format!("preview-{}", rel_path.display()).into();
        let (markdown, error) = load_preview(&rel_path, &abs_path);

        let _ = TextView::markdown(preview_id.clone(), markdown.clone(), window, cx);

        Self {
            focus_handle: cx.focus_handle(),
            rel_path,
            preview_id,
            file_name,
            markdown,
            error,
            shell,
        }
    }
}

fn load_preview(rel_path: &Path, abs_path: &Path) -> (SharedString, Option<String>) {
    let meta = match std::fs::metadata(abs_path) {
        Ok(m) => m,
        Err(err) => return (SharedString::default(), Some(err.to_string())),
    };
    if !meta.is_file() {
        return (SharedString::default(), Some("Not a file".into()));
    }
    if meta.len() as usize > MAX_PREVIEW_BYTES {
        return (
            SharedString::default(),
            Some(format!(
                "File too large to preview ({} KB max)",
                MAX_PREVIEW_BYTES / 1024
            )),
        );
    }

    let bytes = match std::fs::read(abs_path) {
        Ok(b) => b,
        Err(err) => return (SharedString::default(), Some(err.to_string())),
    };
    if bytes.iter().take(8192).any(|b| *b == 0) {
        return (
            SharedString::default(),
            Some("Binary file cannot be previewed".into()),
        );
    }
    let content = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            return (
                SharedString::default(),
                Some("Invalid UTF-8 text file".into()),
            )
        }
    };

    let md = if is_markdown(rel_path) {
        content
    } else {
        wrap_code_block(language_for_path(rel_path), &content)
    };
    (md.into(), None)
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
        Some("tsx") => "tsx",
        Some("jsx") => "javascript",
        Some("go") => "go",
        Some("sh" | "bash" | "zsh") => "bash",
        Some("sql") => "sql",
        Some("html") => "html",
        Some("css") => "css",
        Some("md" | "markdown") => "markdown",
        _ => "text",
    }
}

fn wrap_code_block(lang: &str, code: &str) -> String {
    let fence = if code.contains("```") { "````" } else { "```" };
    format!("{fence}{lang}\n{code}\n{fence}")
}

impl Panel for FilePreviewPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_FILE_PREVIEW
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.file_name.clone()
    }

    fn tab_name(&self, _: &App) -> Option<SharedString> {
        Some(self.file_name.clone())
    }

    fn closable(&self, _: &App) -> bool {
        true
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }

    fn on_removed(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let rel = self.rel_path.clone();
        let shell = self.shell.clone();
        shell
            .update(cx, |app, cx| {
                app.dispatch(WorkspaceAction::CloseFile(rel), cx);
            })
            .ok();
    }
}

impl EventEmitter<PanelEvent> for FilePreviewPanel {}
impl Focusable for FilePreviewPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for FilePreviewPanel {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some(err) = &self.error {
            return v_flex()
                .size_full()
                .p_4()
                .child(Alert::error("file-preview-error", err.clone()).title("Preview failed"));
        }

        v_flex().size_full().overflow_hidden().child(
            TextView::markdown(self.preview_id.clone(), self.markdown.clone(), window, cx)
                .scrollable(true),
        )
    }
}
