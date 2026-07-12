use gpui::prelude::FluentBuilder as _;
use gpui::{div, px, ClickEvent, Context, Entity, IntoElement, ParentElement, Styled};
use gpui_component::{
    button::{Button, ButtonVariants, DropdownButton},
    group_box::{GroupBox, GroupBoxVariants},
    h_flex,
    input::Input,
    label::Label,
    menu::PopupMenuItem,
    spinner::Spinner,
    v_flex, ActiveTheme, Disableable, Sizable, StyledExt,
};

use crate::panels::center::chat::ChatPanel;
use crate::settings::{provider_available, ChatMode, ModelPreset, MODEL_PRESETS};
use crate::ui::icons::{role_avatar, sym};
use crate::ui::style::{
    chat_card, composer_strip, hero_icon, user_bubble_bg, with_alpha, CONTENT_MAX_W, PAGE_PX,
};

/// Composer mirrors the user-message bubble: same card, avatar on the right.
pub fn render_composer(
    panel: &ChatPanel,
    busy: bool,
    can_send: bool,
    mode_label: &str,
    chat_mode: ChatMode,
    model_label: &str,
    provider: &str,
    model: &str,
    entity: Entity<ChatPanel>,
    cx: &mut Context<ChatPanel>,
) -> impl IntoElement {
    composer_strip(cx).child(
        div().w_full().px(px(PAGE_PX)).py_4().child(
            v_flex()
                .gap_2()
                .max_w(px(CONTENT_MAX_W))
                .mx_auto()
                .w_full()
                .child(composer_bubble(
                    panel,
                    busy,
                    can_send,
                    mode_label,
                    chat_mode,
                    model_label,
                    provider,
                    model,
                    entity,
                    cx,
                ))
                .child(
                    Label::new("Enter to send · Shift+Enter for newline")
                        .text_xs()
                        .text_color(cx.theme().muted_foreground)
                        .text_center()
                        .w_full(),
                ),
        ),
    )
}

fn composer_bubble(
    panel: &ChatPanel,
    busy: bool,
    can_send: bool,
    mode_label: &str,
    chat_mode: ChatMode,
    model_label: &str,
    provider: &str,
    model: &str,
    entity: Entity<ChatPanel>,
    cx: &mut Context<ChatPanel>,
) -> impl IntoElement {
    h_flex()
        .w_full()
        .gap_3()
        .items_end()
        .justify_end()
        .child(
            chat_card(cx)
                .flex_grow()
                .max_w(px(CONTENT_MAX_W))
                .bg(user_bubble_bg(cx))
                .child(
                    v_flex()
                        .px_4()
                        .pt_3()
                        .pb_3()
                        .gap_3()
                        .child(
                            Input::new(&panel.text_input)
                                .appearance(false)
                                .disabled(busy),
                        )
                        .child(
                            h_flex()
                                .justify_between()
                                .items_center()
                                .child(
                                    h_flex()
                                        .gap_2()
                                        .items_center()
                                        .child(mode_picker(
                                            mode_label,
                                            chat_mode,
                                            busy,
                                            entity.clone(),
                                        ))
                                        .child(model_picker(
                                            model_label,
                                            provider,
                                            model,
                                            busy,
                                            entity.clone(),
                                        )),
                                )
                                .child(
                                    h_flex()
                                        .gap_2()
                                        .items_center()
                                        .when(busy, |row| {
                                            row.child(Spinner::new().small()).child(
                                                Button::new("stop")
                                                    .outline()
                                                    .small()
                                                    .label("Stop")
                                                    .on_click(
                                                        cx.listener(ChatPanel::stop_generation),
                                                    ),
                                            )
                                        })
                                        .child(send_button(can_send, cx)),
                                ),
                        ),
                ),
        )
        .child(role_avatar(
            cx,
            sym::USER,
            with_alpha(cx.theme().primary, 0.18),
        ))
}

fn send_button(can_send: bool, cx: &mut Context<ChatPanel>) -> impl IntoElement {
    Button::new("send")
        .primary()
        .small()
        .label(sym::UP)
        .disabled(!can_send)
        .on_click(cx.listener(|this, _, window, cx| {
            this.schedule_send(window, cx);
        }))
}

pub fn render_empty_state(
    entity: Entity<ChatPanel>,
    cx: &mut Context<ChatPanel>,
) -> impl IntoElement {
    let suggestions = [
        "Explain this codebase",
        "Find bugs in recent changes",
        "Add a new feature",
    ];

    v_flex()
        .gap_6()
        .max_w(px(CONTENT_MAX_W))
        .w_full()
        .items_center()
        .child(hero_icon(cx))
        .child(
            v_flex()
                .gap_2()
                .items_center()
                .child(
                    div()
                        .text_2xl()
                        .font_semibold()
                        .text_color(cx.theme().foreground)
                        .child("What are we building?"),
                )
                .child(
                    Label::new("Ask about your project, edit files, or run agent workflows.")
                        .text_sm()
                        .text_color(cx.theme().muted_foreground)
                        .text_center(),
                ),
        )
        .child(
            GroupBox::new().outline().title("Quick starts").child(
                h_flex()
                    .gap_2()
                    .justify_center()
                    .children(suggestions.iter().enumerate().map(|(ix, text)| {
                        let entity = entity.clone();
                        let text = text.to_string();
                        Button::new(("suggestion", ix))
                            .outline()
                            .small()
                            .label(text.clone())
                            .on_click(move |_: &ClickEvent, window, cx| {
                                entity.update(cx, |panel, cx| {
                                    panel.text_input.update(cx, |input, cx| {
                                        input.set_value(&text, window, cx);
                                    });
                                    cx.notify();
                                });
                            })
                    })),
            ),
        )
}

fn model_picker(
    label: &str,
    provider: &str,
    model: &str,
    disabled: bool,
    entity: Entity<ChatPanel>,
) -> impl IntoElement {
    let label = label.to_string();
    let provider = provider.to_string();
    let model = model.to_string();
    DropdownButton::new("model-picker")
        .ghost()
        .small()
        .compact()
        .button(Button::new("model-label").ghost().small().label(label))
        .disabled(disabled)
        .dropdown_menu(move |menu, _window, _cx| build_model_menu(menu, &entity, &provider, &model))
}

fn build_model_menu(
    mut menu: gpui_component::menu::PopupMenu,
    entity: &Entity<ChatPanel>,
    provider: &str,
    model: &str,
) -> gpui_component::menu::PopupMenu {
    let mut last_section = "";
    for preset in MODEL_PRESETS {
        if preset.section != last_section {
            menu = menu.item(PopupMenuItem::Separator);
            menu = menu.item(PopupMenuItem::Label(preset.section.into()));
            last_section = preset.section;
        }
        menu = menu.item(model_preset_item(preset, entity, provider, model));
    }
    menu
}

fn model_preset_item(
    preset: &ModelPreset,
    entity: &Entity<ChatPanel>,
    provider: &str,
    model: &str,
) -> PopupMenuItem {
    let entity = entity.clone();
    let provider_id = preset.provider.to_string();
    let model_id = preset.model.to_string();
    let checked = provider == preset.provider && model == preset.model;
    PopupMenuItem::new(preset.label.to_string())
        .disabled(!provider_available(preset.provider))
        .checked(checked)
        .on_click(move |_, _, cx| {
            if !provider_available(&provider_id) {
                return;
            }
            entity.update(cx, |panel, cx| {
                panel.set_model(&provider_id, &model_id, cx);
            });
        })
}

fn mode_picker(
    label: &str,
    current: ChatMode,
    disabled: bool,
    entity: Entity<ChatPanel>,
) -> impl IntoElement {
    DropdownButton::new("mode-picker")
        .ghost()
        .small()
        .compact()
        .button(
            Button::new("mode-label")
                .ghost()
                .small()
                .label(label.to_string()),
        )
        .disabled(disabled)
        .dropdown_menu(move |menu, _window, _cx| {
            let mut menu = menu;
            for mode in ChatMode::ALL {
                let entity = entity.clone();
                menu = menu.item(
                    PopupMenuItem::new(mode.label())
                        .checked(mode == current)
                        .on_click(move |_, _, cx| {
                            entity.update(cx, |panel, cx| {
                                panel.set_chat_mode(mode, cx);
                            });
                        }),
                );
            }
            menu
        })
}
