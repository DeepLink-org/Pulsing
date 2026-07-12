use gpui::{
    div, prelude::FluentBuilder as _, App, Context, Entity, EventEmitter, FocusHandle, Focusable,
    IntoElement, ParentElement, Render, SharedString, Styled, WeakEntity, Window,
};
use gpui_component::{
    button::{Button, ButtonVariants},
    dock::{Panel, PanelControl, PanelEvent},
    h_flex,
    label::Label,
    list::ListItem,
    tag::Tag,
    v_flex, ActiveTheme, Sizable, StyledExt,
};

use crate::dock::layout::PANEL_REVISIONS;
use crate::model::{WorkspaceAction, WorkspaceModel};
use crate::panels::common::empty_list_hint;
use crate::shell::WorkspaceShell;
use crate::ui::icons::sym;

pub struct RevisionsPanel {
    focus_handle: FocusHandle,
    workspace: Entity<WorkspaceModel>,
    shell: WeakEntity<WorkspaceShell>,
}

impl RevisionsPanel {
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
                shell.dispatch(WorkspaceAction::RefreshRevisions, cx)
            })
            .ok();
    }
}

impl Panel for RevisionsPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_REVISIONS
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Revisions"
    }

    fn tab_name(&self, _: &App) -> Option<SharedString> {
        Some("History".into())
    }

    fn zoomable(&self, _: &App) -> Option<PanelControl> {
        Some(PanelControl::Toolbar)
    }

    fn toolbar_buttons(&mut self, _: &mut Window, cx: &mut Context<Self>) -> Option<Vec<Button>> {
        Some(vec![Button::new("revisions-refresh")
            .ghost()
            .small()
            .label(sym::REFRESH)
            .tooltip("Refresh checkpoints")
            .on_click(cx.listener(|this, _, _, cx| this.refresh(cx)))])
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

impl EventEmitter<PanelEvent> for RevisionsPanel {}
impl Focusable for RevisionsPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for RevisionsPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let snapshot = self.workspace.read(cx).revisions.clone();

        v_flex()
            .size_full()
            .p_2()
            .gap_1()
            .when(snapshot.revisions.is_empty(), |col| {
                col.child(empty_list_hint(
                    cx,
                    "No checkpoints yet",
                    "Use `pulsing checkpoint` or the agent /checkpoint command.",
                ))
            })
            .when(!snapshot.revisions.is_empty(), |col| {
                col.children(snapshot.revisions.iter().enumerate().map(|(ix, rev)| {
                    let is_head = snapshot.head.as_deref() == Some(rev.id.as_str());
                    ListItem::new(ix).child(
                        div()
                            .w_full()
                            .px_3()
                            .py_2()
                            .rounded_lg()
                            .when(is_head, |row| {
                                row.border_1().border_color(cx.theme().primary)
                            })
                            .when(!is_head, |row| {
                                row.border_1().border_color(cx.theme().border)
                            })
                            .bg(cx.theme().popover)
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_center()
                                    .when(is_head, |row| {
                                        row.child(Tag::success().small().outline().child("HEAD"))
                                    })
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_semibold()
                                            .text_color(cx.theme().foreground)
                                            .child(rev.id.clone()),
                                    )
                                    .child(
                                        Label::new(format!("{} files", rev.file_count))
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground),
                                    )
                                    .child(
                                        Label::new(rev.message.clone())
                                            .text_xs()
                                            .text_color(cx.theme().muted_foreground)
                                            .flex_grow()
                                            .overflow_hidden(),
                                    ),
                            ),
                    )
                }))
            })
    }
}
