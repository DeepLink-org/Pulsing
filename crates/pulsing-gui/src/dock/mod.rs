use std::sync::Arc;

use gpui::{Context, Entity, WeakEntity, Window};
use gpui_component::dock::{DockArea, DockItem};

use crate::panels::{
    AgentsPanel, ChatPanel, ClusterPanel, ExplorerPanel, RevisionsPanel, RuntimeSummaryPanel,
    WorkflowsPanel,
};

use layout::{bottom_height, left_width, right_width};

pub mod layout;

pub struct WorkspacePanels {
    pub explorer: Entity<ExplorerPanel>,
    pub revisions: Entity<RevisionsPanel>,
    pub workflows: Entity<WorkflowsPanel>,
    pub chat: Entity<ChatPanel>,
    pub agents: Entity<AgentsPanel>,
    pub runtime: Entity<RuntimeSummaryPanel>,
    pub cluster: Entity<ClusterPanel>,
}

pub fn configure_dock(
    dock: &mut DockArea,
    weak: WeakEntity<DockArea>,
    panels: &WorkspacePanels,
    window: &mut Window,
    cx: &mut Context<DockArea>,
) {
    let center = DockItem::tabs(vec![Arc::new(panels.chat.clone())], &weak, window, cx);
    dock.set_center(center, window, cx);

    let left = DockItem::tabs(
        vec![
            Arc::new(panels.explorer.clone()),
            Arc::new(panels.revisions.clone()),
            Arc::new(panels.workflows.clone()),
        ],
        &weak,
        window,
        cx,
    );
    dock.set_left_dock(left, Some(left_width()), true, window, cx);

    let right = DockItem::panel(Arc::new(panels.agents.clone()));
    dock.set_right_dock(right, Some(right_width()), true, window, cx);

    let bottom = DockItem::tabs(
        vec![
            Arc::new(panels.runtime.clone()),
            Arc::new(panels.cluster.clone()),
        ],
        &weak,
        window,
        cx,
    );
    dock.set_bottom_dock(bottom, Some(bottom_height()), true, window, cx);
}
