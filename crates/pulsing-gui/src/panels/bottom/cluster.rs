use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, IntoElement, Render, SharedString, Window,
};
use gpui_component::dock::{Panel, PanelEvent};

use crate::dock::layout::PANEL_CLUSTER;
use crate::panels::common::phase_placeholder;

pub struct ClusterPanel {
    focus_handle: FocusHandle,
}

impl ClusterPanel {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
        }
    }
}

impl Panel for ClusterPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_CLUSTER
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Cluster"
    }

    fn tab_name(&self, _: &App) -> Option<SharedString> {
        Some("Cluster".into())
    }
}

impl EventEmitter<PanelEvent> for ClusterPanel {}
impl Focusable for ClusterPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ClusterPanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        phase_placeholder(
            "cluster-phase3",
            "Cluster runtime",
            "Node and actor status will appear here in Phase 3.",
        )
    }
}
