use gpui::px;

pub const LEFT_DOCK_WIDTH: f32 = 272.0;
pub const RIGHT_DOCK_WIDTH: f32 = 268.0;
pub const BOTTOM_DOCK_HEIGHT: f32 = 36.0;

pub const PANEL_EXPLORER: &str = "explorer";
pub const PANEL_REVISIONS: &str = "revisions";
pub const PANEL_WORKFLOWS: &str = "workflows";
pub const PANEL_CHAT: &str = "chat";
pub const PANEL_FILE_PREVIEW: &str = "file-preview";
pub const PANEL_AGENTS: &str = "agents";
pub const PANEL_RUNTIME: &str = "runtime";
pub const PANEL_CLUSTER: &str = "cluster";

pub fn left_width() -> gpui::Pixels {
    px(LEFT_DOCK_WIDTH)
}

pub fn right_width() -> gpui::Pixels {
    px(RIGHT_DOCK_WIDTH)
}

pub fn bottom_height() -> gpui::Pixels {
    px(BOTTOM_DOCK_HEIGHT)
}
