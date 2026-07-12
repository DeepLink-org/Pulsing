use gpui::{
    div, prelude::FluentBuilder as _, px, App, Context, Entity, EventEmitter, FocusHandle,
    Focusable, IntoElement, ParentElement, Render, SharedString, Styled, Window,
};
use gpui_component::{
    dock::{Panel, PanelEvent},
    h_flex,
    label::Label,
    spinner::Spinner,
    tag::Tag,
    ActiveTheme, Sizable,
};

use crate::dock::layout::PANEL_RUNTIME;
use crate::model::WorkspaceModel;
use crate::ui::style::status_dot;

pub struct RuntimeSummaryPanel {
    focus_handle: FocusHandle,
    workspace: Entity<WorkspaceModel>,
}

impl RuntimeSummaryPanel {
    pub fn new(workspace: Entity<WorkspaceModel>, cx: &mut Context<Self>) -> Self {
        cx.observe(&workspace, |_, _, cx| cx.notify()).detach();
        Self {
            focus_handle: cx.focus_handle(),
            workspace,
        }
    }
}

impl Panel for RuntimeSummaryPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_RUNTIME
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Runtime"
    }

    fn tab_name(&self, _: &App) -> Option<SharedString> {
        Some("Summary".into())
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

impl EventEmitter<PanelEvent> for RuntimeSummaryPanel {}
impl Focusable for RuntimeSummaryPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for RuntimeSummaryPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let runtime = self.workspace.read(cx).runtime.clone();
        let busy = runtime.local_busy;

        div()
            .size_full()
            .border_t_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().popover)
            .shadow_md()
            .child(
                h_flex()
                    .size_full()
                    .px_4()
                    .items_center()
                    .gap_3()
                    .child(status_dot(busy, cx))
                    .when(busy, |row| row.child(Spinner::new().small()))
                    .child(
                        Label::new(if busy { "Agent running" } else { "Ready" })
                            .text_xs()
                            .text_color(cx.theme().foreground),
                    )
                    .child(div().w(px(1.0)).h(px(14.0)).bg(cx.theme().border))
                    .child(metric_tag("files", runtime.file_count))
                    .child(metric_tag("revs", runtime.revision_count))
                    .child(metric_tag("workflows", runtime.workflow_count))
                    .child(
                        Tag::secondary()
                            .small()
                            .outline()
                            .child("cluster · phase 3"),
                    )
                    .child(
                        Label::new(runtime.session_title.clone())
                            .text_xs()
                            .text_color(cx.theme().muted_foreground)
                            .ml_auto()
                            .overflow_hidden(),
                    ),
            )
    }
}

fn metric_tag(label: &'static str, count: usize) -> Tag {
    Tag::secondary()
        .small()
        .outline()
        .child(format!("{label} {count}"))
}
