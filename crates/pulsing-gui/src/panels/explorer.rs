use gpui::{
    div, prelude::FluentBuilder as _, px, App, AppContext, Context, Entity, EventEmitter,
    FocusHandle, Focusable, IntoElement, ParentElement, Render, SharedString, Styled, WeakEntity,
    Window,
};
use gpui_component::{
    breadcrumb::{Breadcrumb, BreadcrumbItem},
    button::{Button, ButtonVariants},
    dock::{Panel, PanelControl, PanelEvent},
    h_flex,
    list::ListItem,
    tree::{tree, TreeEntry, TreeItem, TreeState},
    ActiveTheme, Sizable,
};

use crate::dock::layout::PANEL_EXPLORER;
use crate::model::{build_file_tree, count_files, WorkspaceAction, WorkspaceModel};
use crate::shell::WorkspaceShell;
use crate::ui::icons::{glyph_xs, sym, tree_chevron, tree_entry_icon};
use crate::ui::notify;
use crate::ui::style::{sidebar_panel, PanelSide};

use std::path::PathBuf;
use std::time::{Duration, Instant};

const DOUBLE_CLICK_WINDOW: Duration = Duration::from_millis(400);

pub struct ExplorerPanel {
    focus_handle: FocusHandle,
    workspace: Entity<WorkspaceModel>,
    shell: WeakEntity<WorkspaceShell>,
    tree_state: Entity<TreeState>,
    file_tree: Vec<TreeItem>,
    last_click: Option<(usize, Instant)>,
}

impl ExplorerPanel {
    pub fn new(
        workspace: Entity<WorkspaceModel>,
        shell: WeakEntity<WorkspaceShell>,
        cx: &mut Context<Self>,
    ) -> Self {
        let tree_state = cx.new(|cx| TreeState::new(cx));

        let mut panel = Self {
            focus_handle: cx.focus_handle(),
            workspace,
            shell,
            tree_state: tree_state.clone(),
            file_tree: Vec::new(),
            last_click: None,
        };
        panel.rebuild_tree(cx);

        cx.observe(&tree_state, |this, _, cx| {
            this.on_tree_changed(cx);
        })
        .detach();

        panel
    }

    fn dispatch(&self, action: WorkspaceAction, cx: &mut Context<Self>) {
        self.shell
            .update(cx, |shell, cx| shell.dispatch(action, cx))
            .ok();
    }

    fn on_tree_changed(&mut self, cx: &mut Context<Self>) {
        if let Some(entry) = self.tree_state.read(cx).selected_entry().cloned() {
            let ix = self.tree_state.read(cx).selected_index().unwrap_or(0);
            if !entry.is_folder() {
                let now = Instant::now();
                let double_click = self.last_click.is_some_and(|(last_ix, t)| {
                    last_ix == ix && now.duration_since(t) < DOUBLE_CLICK_WINDOW
                });
                self.last_click = Some((ix, now));

                if double_click {
                    let rel = PathBuf::from(entry.item().id.to_string());
                    self.dispatch(WorkspaceAction::OpenFile(rel), cx);
                    return;
                }
            } else {
                self.last_click = None;
            }
        }
        self.sync_selection(cx);
    }

    fn rebuild_tree(&mut self, cx: &mut Context<Self>) {
        let layout = self.workspace.read(cx).layout.clone();
        self.file_tree = build_file_tree(&layout, &self.file_tree);
        let file_count = count_files(&self.file_tree);

        self.tree_state.update(cx, |state, cx| {
            state.set_items(self.file_tree.clone(), cx);
        });

        self.workspace.update(cx, |model, cx| {
            model.runtime.file_count = file_count;
            cx.notify();
        });
        cx.notify();
    }

    fn refresh(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.rebuild_tree(cx);
        let file_count = self.workspace.read(cx).runtime.file_count;
        notify::success(
            window,
            "Workspace refreshed",
            format!("{file_count} files"),
            cx,
        );
    }

    fn sync_selection(&mut self, cx: &mut Context<Self>) {
        let selected = self.tree_state.read(cx).selected_entry().and_then(|entry| {
            if entry.is_folder() {
                None
            } else {
                Some(PathBuf::from(entry.item().id.to_string()))
            }
        });

        if self.workspace.read(cx).selected_file == selected {
            return;
        }

        self.workspace.update(cx, |model, cx| {
            model.set_selected_file(selected);
            cx.notify();
        });
        cx.notify();
    }

    fn open_selected(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(rel) = self.workspace.read(cx).selected_file.clone() else {
            notify::warning(
                window,
                "No file selected",
                "Select a file in the tree first",
                cx,
            );
            return;
        };
        self.dispatch(WorkspaceAction::OpenFile(rel), cx);
    }
}

impl Panel for ExplorerPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_EXPLORER
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Explorer"
    }

    fn tab_name(&self, _: &App) -> Option<SharedString> {
        Some("Files".into())
    }

    fn zoomable(&self, _: &App) -> Option<PanelControl> {
        Some(PanelControl::Toolbar)
    }

    fn toolbar_buttons(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Vec<Button>> {
        Some(vec![
            Button::new("explorer-open")
                .ghost()
                .small()
                .label(sym::OPEN)
                .tooltip("Open selected file")
                .on_click(cx.listener(|this, _, window, cx| this.open_selected(window, cx))),
            Button::new("explorer-refresh")
                .ghost()
                .small()
                .label(sym::REFRESH)
                .tooltip("Refresh workspace")
                .on_click(cx.listener(|this, _, window, cx| this.refresh(window, cx))),
        ])
    }

    fn title_suffix(&mut self, _: &mut Window, cx: &mut Context<Self>) -> Option<impl IntoElement> {
        let path = self.workspace.read(cx).selected_file.clone()?;
        Some(file_breadcrumb(&path))
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

impl EventEmitter<PanelEvent> for ExplorerPanel {}
impl Focusable for ExplorerPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ExplorerPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(
            sidebar_panel(cx, PanelSide::Left)
                .child(tree(&self.tree_state, |ix, entry, selected, _, cx| {
                    tree_row(ix, entry, selected, cx)
                })),
        )
    }
}

fn file_breadcrumb(path: &std::path::Path) -> Breadcrumb {
    use std::path::Component;

    let mut breadcrumb = Breadcrumb::new();
    for component in path.components() {
        let label = match component {
            Component::Normal(s) => s.to_string_lossy().into_owned(),
            Component::RootDir => continue,
            other => other.as_os_str().to_string_lossy().into_owned(),
        };
        breadcrumb = breadcrumb.child(BreadcrumbItem::new(label));
    }
    breadcrumb
}

fn tree_row(ix: usize, entry: &TreeEntry, selected: bool, cx: &mut App) -> ListItem {
    let item = entry.item();
    let is_folder = entry.is_folder();
    let expanded = entry.is_expanded();

    let chevron = tree_chevron(is_folder, expanded);
    let file_icon = tree_entry_icon(is_folder, expanded);

    ListItem::new(ix)
        .py_1()
        .mx_2()
        .rounded_md()
        .when(selected, |row| {
            row.bg(cx.theme().sidebar_accent)
                .border_1()
                .border_color(cx.theme().sidebar_border)
                .shadow_xs()
        })
        .pl(px(6.0 + entry.depth() as f32 * 14.0))
        .selected(selected)
        .child(
            h_flex()
                .gap_1p5()
                .items_center()
                .child(glyph_xs(chevron, cx.theme().muted_foreground))
                .child(glyph_xs(
                    file_icon,
                    if is_folder {
                        cx.theme().primary
                    } else {
                        cx.theme().muted_foreground
                    },
                ))
                .child(
                    div()
                        .text_sm()
                        .text_color(if selected {
                            cx.theme().foreground
                        } else {
                            cx.theme().sidebar_foreground
                        })
                        .child(item.label.clone()),
                ),
        )
}
