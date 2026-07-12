use gpui::{
    div, prelude::FluentBuilder as _, px, App, Context, Entity, EventEmitter, FocusHandle,
    Focusable, IntoElement, ParentElement, Render, SharedString, Styled, WeakEntity, Window,
};
use gpui_component::{
    button::{Button, ButtonVariants},
    dock::{Panel, PanelControl, PanelEvent},
    h_flex,
    label::Label,
    list::ListItem,
    v_flex, ActiveTheme, Sizable, StyledExt,
};

use crate::dock::layout::PANEL_WORKFLOWS;
use crate::model::{WorkspaceAction, WorkspaceModel};
use crate::panels::common::empty_list_hint;
use crate::shell::WorkspaceShell;
use crate::ui::icons::{icon_badge, sym};

pub struct WorkflowsPanel {
    focus_handle: FocusHandle,
    workspace: Entity<WorkspaceModel>,
    shell: WeakEntity<WorkspaceShell>,
}

impl WorkflowsPanel {
    pub fn new(
        workspace: Entity<WorkspaceModel>,
        shell: WeakEntity<WorkspaceShell>,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&workspace, |_, _, cx| cx.notify()).detach();
        Self {
            focus_handle: cx.focus_handle(),
            workspace,
            shell,
        }
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        self.shell
            .update(cx, |shell, cx| {
                shell.dispatch(WorkspaceAction::RefreshWorkflows, cx)
            })
            .ok();
    }
}

impl Panel for WorkflowsPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_WORKFLOWS
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Workflows"
    }

    fn tab_name(&self, _: &App) -> Option<SharedString> {
        Some("Workflows".into())
    }

    fn zoomable(&self, _: &App) -> Option<PanelControl> {
        Some(PanelControl::Toolbar)
    }

    fn toolbar_buttons(&mut self, _: &mut Window, cx: &mut Context<Self>) -> Option<Vec<Button>> {
        Some(vec![Button::new("workflows-refresh")
            .ghost()
            .small()
            .label(sym::REFRESH)
            .tooltip("Refresh workflow scripts")
            .on_click(cx.listener(|this, _, _, cx| this.refresh(cx)))])
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

impl EventEmitter<PanelEvent> for WorkflowsPanel {}
impl Focusable for WorkflowsPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for WorkflowsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let scripts = self.workspace.read(cx).workflow_scripts.clone();

        v_flex()
            .size_full()
            .p_2()
            .gap_1()
            .when(scripts.is_empty(), |col| {
                col.child(empty_list_hint(
                    cx,
                    "No workflow scripts",
                    "Add `.py` files under `.pulsing/workflows/` (run `pulsing init`).",
                ))
            })
            .when(!scripts.is_empty(), |col| {
                col.children(scripts.iter().enumerate().map(|(ix, path)| {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.display().to_string());
                    ListItem::new(ix).child(
                        div()
                            .w_full()
                            .px_3()
                            .py_2()
                            .rounded_lg()
                            .border_1()
                            .border_color(cx.theme().border)
                            .bg(cx.theme().popover)
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .child(icon_badge(cx, sym::FILE, 28.0))
                                    .child(
                                        v_flex()
                                            .gap(px(1.0))
                                            .child(div().text_sm().font_semibold().child(name))
                                            .child(
                                                Label::new("Python workflow")
                                                    .text_xs()
                                                    .text_color(cx.theme().muted_foreground),
                                            ),
                                    ),
                            ),
                    )
                }))
            })
    }
}
