use gpui::{div, px, App, IntoElement, ParentElement, SharedString, Styled};
use gpui_component::{
    alert::Alert, label::Label, text::Text, v_flex, ActiveTheme, Sizable, StyledExt,
};

use crate::ui::icons::{icon_badge, sym};

pub fn phase_placeholder(
    id: &'static str,
    title: impl Into<SharedString>,
    message: impl Into<Text>,
) -> impl IntoElement {
    v_flex()
        .size_full()
        .p_4()
        .child(Alert::info(id, message).title(title).small())
}

pub fn empty_list_hint(
    cx: &App,
    title: impl Into<SharedString>,
    message: impl Into<SharedString>,
) -> impl IntoElement {
    v_flex()
        .size_full()
        .p_6()
        .gap_3()
        .justify_center()
        .items_center()
        .child(icon_badge(cx, sym::INBOX, 40.0))
        .child(
            div()
                .text_sm()
                .font_semibold()
                .text_color(cx.theme().foreground)
                .child(title.into()),
        )
        .child(
            Label::new(message)
                .text_xs()
                .text_color(cx.theme().muted_foreground)
                .text_center()
                .max_w(px(280.0)),
        )
}
