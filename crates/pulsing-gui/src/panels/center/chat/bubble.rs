use gpui::{div, px, Context, IntoElement, ParentElement, Styled};
use gpui_component::{h_flex, label::Label, ActiveTheme};

use crate::panels::center::chat::ChatPanel;
use crate::ui::icons::{glyph_xs, role_avatar, sym, tool_status_symbol};
use crate::ui::style::{
    chat_card, message_text, user_bubble_bg, with_alpha, AVATAR_SIZE, BUBBLE_MAX_W, CONTENT_MAX_W,
    PAGE_PX,
};

pub fn message_row_shell(child: impl IntoElement) -> impl IntoElement {
    div()
        .w_full()
        .flex()
        .justify_center()
        .px(px(PAGE_PX))
        .py_2()
        .child(div().w_full().max_w(px(CONTENT_MAX_W)).child(child))
}

pub fn user_message(cx: &mut Context<ChatPanel>, text: &str) -> impl IntoElement {
    h_flex()
        .w_full()
        .gap_3()
        .items_start()
        .justify_end()
        .child(
            chat_card(cx)
                .max_w(px(BUBBLE_MAX_W))
                .min_w(px(80.0))
                .px_4()
                .py_3()
                .bg(user_bubble_bg(cx))
                .child(message_text(text).text_color(cx.theme().foreground)),
        )
        .child(role_avatar(
            cx,
            sym::USER,
            with_alpha(cx.theme().primary, 0.18),
        ))
}

pub fn assistant_message(
    cx: &mut Context<ChatPanel>,
    body: &str,
    streaming: bool,
) -> impl IntoElement {
    let display = if streaming && body.is_empty() {
        "▍".to_string()
    } else if streaming {
        format!("{body}▍")
    } else {
        body.to_string()
    };

    h_flex()
        .w_full()
        .gap_3()
        .items_start()
        .child(role_avatar(
            cx,
            sym::BOT,
            with_alpha(cx.theme().primary, 0.18),
        ))
        .child(
            chat_card(cx)
                .flex_grow()
                .max_w(px(BUBBLE_MAX_W))
                .px_4()
                .py_3()
                .child(message_text(display).text_color(cx.theme().foreground)),
        )
}

pub fn error_message(cx: &mut Context<ChatPanel>, text: &str) -> impl IntoElement {
    h_flex()
        .w_full()
        .gap_3()
        .items_start()
        .child(role_avatar(
            cx,
            sym::ALERT,
            with_alpha(cx.theme().danger, 0.18),
        ))
        .child(
            chat_card(cx)
                .flex_grow()
                .max_w(px(BUBBLE_MAX_W))
                .px_4()
                .py_3()
                .border_color(cx.theme().danger)
                .bg(with_alpha(cx.theme().danger, 0.08))
                .child(
                    Label::new(text.to_string())
                        .text_sm()
                        .text_color(cx.theme().danger),
                ),
        )
}

pub fn tool_message(
    cx: &mut Context<ChatPanel>,
    name: &str,
    running: bool,
    ok: bool,
    detail: &str,
) -> impl IntoElement {
    let symbol = tool_status_symbol(running, ok);
    let color = if running {
        cx.theme().muted_foreground
    } else if ok {
        cx.theme().success
    } else {
        cx.theme().danger
    };

    let summary = if detail.is_empty() {
        name.to_string()
    } else {
        format!("{name} — {detail}")
    };

    h_flex()
        .w_full()
        .gap_3()
        .items_start()
        .child(div().size(px(AVATAR_SIZE)).flex_shrink_0())
        .child(
            chat_card(cx).px_3().py_2().child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(glyph_xs(symbol, color))
                    .child(
                        Label::new(summary)
                            .text_xs()
                            .text_color(cx.theme().muted_foreground),
                    ),
            ),
        )
}
