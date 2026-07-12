//! Unicode glyphs — no icon-font dependency.

use gpui::{div, px, App, Hsla, IntoElement, ParentElement, Styled};
use gpui_component::ActiveTheme;

use crate::ui::style::{with_alpha, AVATAR_SIZE};

pub mod sym {
    pub const USER: &str = "◉";
    pub const BOT: &str = "✦";
    pub const ALERT: &str = "⚠";
    pub const OK: &str = "✓";
    pub const FAIL: &str = "✗";
    pub const BUSY: &str = "⟳";
    pub const CHEVRON_DOWN: &str = "▾";
    pub const CHEVRON_RIGHT: &str = "▸";
    pub const DOT: &str = "·";
    pub const FOLDER: &str = "📁";
    pub const FOLDER_OPEN: &str = "📂";
    pub const FILE: &str = "◇";
    pub const PLUS: &str = "＋";
    pub const GLOBE: &str = "◎";
    pub const UP: &str = "↑";
    pub const INBOX: &str = "▤";
    pub const REFRESH: &str = "↻";
    pub const OPEN: &str = "↗";
}

pub fn labeled(symbol: &'static str, label: impl AsRef<str>) -> String {
    format!("{} {}", symbol, label.as_ref())
}

pub fn glyph(symbol: &'static str) -> gpui::Div {
    div().text_center().child(symbol)
}

pub fn glyph_xs(symbol: &'static str, color: Hsla) -> impl IntoElement {
    glyph(symbol).text_xs().text_color(color)
}

pub fn glyph_sm(symbol: &'static str, color: Hsla) -> impl IntoElement {
    glyph(symbol).text_sm().text_color(color)
}

pub fn role_avatar(cx: &App, symbol: &'static str, tint: Hsla) -> impl IntoElement {
    div()
        .size(px(AVATAR_SIZE))
        .rounded_lg()
        .flex_shrink_0()
        .flex()
        .items_center()
        .justify_center()
        .bg(tint)
        .border_1()
        .border_color(cx.theme().border)
        .child(glyph_sm(symbol, cx.theme().foreground))
}

pub fn icon_tile(cx: &App, symbol: &'static str, size: gpui::Pixels) -> impl IntoElement {
    div()
        .size(size)
        .rounded_xl()
        .flex()
        .items_center()
        .justify_center()
        .bg(with_alpha(cx.theme().primary, 0.22))
        .border_1()
        .border_color(with_alpha(cx.theme().primary, 0.45))
        .shadow_md()
        .child(glyph_sm(symbol, cx.theme().primary))
}

pub fn icon_badge(cx: &App, symbol: &'static str, badge_size: f32) -> impl IntoElement {
    div()
        .size(px(badge_size))
        .rounded_md()
        .flex()
        .items_center()
        .justify_center()
        .bg(cx.theme().muted)
        .child(glyph_sm(symbol, cx.theme().primary))
}

pub fn tree_chevron(is_folder: bool, expanded: bool) -> &'static str {
    if !is_folder {
        sym::DOT
    } else if expanded {
        sym::CHEVRON_DOWN
    } else {
        sym::CHEVRON_RIGHT
    }
}

pub fn tree_entry_icon(is_folder: bool, expanded: bool) -> &'static str {
    if is_folder {
        if expanded {
            sym::FOLDER_OPEN
        } else {
            sym::FOLDER
        }
    } else {
        sym::FILE
    }
}

pub fn tool_status_symbol(running: bool, ok: bool) -> &'static str {
    if running {
        sym::BUSY
    } else if ok {
        sym::OK
    } else {
        sym::FAIL
    }
}
