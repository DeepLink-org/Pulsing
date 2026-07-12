use gpui::{div, px, App, Hsla, IntoElement, ParentElement, Styled};
use gpui_component::ActiveTheme;

use crate::ui::icons::{icon_tile, sym};

pub const CONTENT_MAX_W: f32 = 680.0;
pub const PAGE_PX: f32 = 24.0;
pub const BUBBLE_MAX_W: f32 = 560.0;
pub const AVATAR_SIZE: f32 = 30.0;

pub fn status_dot(busy: bool, cx: &App) -> impl IntoElement {
    let color = if busy {
        cx.theme().warning
    } else {
        cx.theme().success
    };
    div()
        .size(px(6.0))
        .rounded_full()
        .bg(color)
        .border_1()
        .border_color(with_alpha(color, 0.45))
        .flex_shrink_0()
}

pub fn brand_mark(cx: &App) -> impl IntoElement {
    icon_tile(cx, sym::BOT, px(28.0))
}

pub fn hero_icon(cx: &App) -> impl IntoElement {
    icon_tile(cx, sym::BOT, px(56.0))
}

/// Shared card shell for composer + message bubbles.
pub fn chat_card(cx: &App) -> gpui::Div {
    div()
        .rounded_xl()
        .border_1()
        .border_color(cx.theme().border)
        .bg(cx.theme().popover)
        .shadow_md()
}

pub fn recessed_well(cx: &App) -> gpui::Div {
    div()
        .rounded_xl()
        .border_1()
        .border_color(cx.theme().border)
        .bg(cx.theme().group_box)
        .shadow_sm()
}

pub fn with_alpha(color: Hsla, alpha: f32) -> Hsla {
    Hsla { a: alpha, ..color }
}

pub fn user_bubble_bg(cx: &App) -> Hsla {
    with_alpha(cx.theme().primary, 0.12)
}

pub fn sidebar_panel(cx: &App, side: PanelSide) -> gpui::Div {
    let mut el = div().size_full().bg(cx.theme().sidebar);
    match side {
        PanelSide::Left => el = el.border_r_1().border_color(cx.theme().sidebar_border),
        PanelSide::Right => el = el.border_l_1().border_color(cx.theme().sidebar_border),
    }
    el
}

pub enum PanelSide {
    Left,
    Right,
}

pub fn composer_strip(cx: &App) -> gpui::Div {
    div()
        .w_full()
        .border_t_1()
        .border_color(cx.theme().border)
        .bg(cx.theme().background)
}

pub fn message_text(text: impl Into<String>) -> gpui::Div {
    div().text_sm().child(text.into())
}
