mod bubble;
mod composer;
mod messages;

use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use gpui::{
    div, prelude::FluentBuilder as _, px, size, App, AppContext, ClickEvent, Context, Entity,
    EventEmitter, FocusHandle, Focusable, IntoElement, ParentElement, Render, Styled, Window,
};
use gpui_component::{
    dock::{Panel, PanelEvent},
    ActiveTheme, StyledExt, VirtualListScrollHandle,
};
use pulsing_forge::{AgentEvent, InteractiveConfig};

use crate::controller::start_agent_turn;
use crate::dock::layout::PANEL_CHAT;
use crate::model::{SessionStore, WorkspaceModel};
use crate::settings::{build_agent_config, ChatMode};
use crate::ui::style::recessed_well;

use composer::{render_composer, render_empty_state};
use messages::{estimate_message_height, message_list, show_empty_state};

pub struct ChatPanel {
    focus_handle: FocusHandle,
    agent: InteractiveConfig,
    chat_mode: ChatMode,
    workspace: Entity<WorkspaceModel>,
    pub(crate) sessions: Entity<SessionStore>,
    pub(crate) text_input: Entity<gpui_component::input::InputState>,
    message_sizes: Rc<Vec<gpui::Size<gpui::Pixels>>>,
    scroll_handle: VirtualListScrollHandle,
    event_rx: Option<mpsc::Receiver<AgentEvent>>,
}

impl ChatPanel {
    pub fn new(
        agent: InteractiveConfig,
        workspace: Entity<WorkspaceModel>,
        sessions: Entity<SessionStore>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let scroll_handle = VirtualListScrollHandle::new();
        let initial_sizes = Rc::new(vec![]);

        cx.observe(&sessions, |this, _, cx| {
            let chat = this.sessions.read(cx).active_chat().clone();
            this.message_sizes = Rc::new(
                chat.messages
                    .iter()
                    .map(estimate_message_height)
                    .map(|h| size(px(0.), h))
                    .collect(),
            );
            if !chat.messages.is_empty() {
                this.scroll_handle.scroll_to_bottom();
            }
            let title = chat.session_title();
            let busy = chat.busy;
            this.workspace.update(cx, |model, cx| {
                model.set_busy(busy);
                model.set_session_title(title);
                cx.notify();
            });
            cx.notify();
        })
        .detach();

        let text_input = cx.new(|cx| {
            gpui_component::input::InputState::new(window, cx)
                .auto_grow(1, 8)
                .soft_wrap(true)
                .placeholder("Ask anything about your project…")
        });

        cx.subscribe_in(
            &text_input,
            window,
            move |_this, _input, event, window, cx| {
                if matches!(
                    event,
                    gpui_component::input::InputEvent::PressEnter { secondary: false }
                ) {
                    cx.defer_in(window, |panel, window, cx| panel.try_send(window, cx));
                }
            },
        )
        .detach();

        cx.spawn(async move |this, cx| loop {
            cx.background_executor()
                .timer(Duration::from_millis(50))
                .await;
            let _ = this.update(cx, |panel, cx| {
                let mut events = Vec::new();
                if let Some(rx) = panel.event_rx.as_ref() {
                    while let Ok(event) = rx.try_recv() {
                        events.push(event);
                    }
                }
                if events.is_empty() {
                    return;
                }
                panel.sessions.update(cx, |store, cx| {
                    let chat = store.active_chat_mut();
                    for event in events {
                        chat.apply(event);
                    }
                    cx.notify();
                });
                cx.notify();
            });
        })
        .detach();

        Self {
            focus_handle: cx.focus_handle(),
            agent,
            chat_mode: ChatMode::Agent,
            workspace,
            sessions,
            text_input,
            message_sizes: initial_sizes,
            scroll_handle,
            event_rx: None,
        }
    }

    pub(crate) fn schedule_send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        cx.defer_in(window, |panel, window, cx| panel.try_send(window, cx));
    }

    fn try_send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let text = self.text_input.read(cx).value().to_string();
        if text.trim().is_empty() || self.sessions.read(cx).active_chat().busy {
            return;
        }

        self.text_input.update(cx, |input, cx| {
            input.set_value("", window, cx);
        });

        let prompt = text.trim().to_string();
        self.sessions.update(cx, |store, cx| {
            use crate::state::{ChatMessage, MessageKind};
            let chat = store.active_chat_mut();
            chat.messages.push(ChatMessage {
                kind: MessageKind::User(prompt.clone()),
            });
            chat.messages.push(ChatMessage {
                kind: MessageKind::Assistant {
                    body: String::new(),
                    streaming: true,
                },
            });
            chat.busy = true;
            cx.notify();
        });

        let handle = start_agent_turn(
            build_agent_config(
                &self.agent.cwd,
                &self.agent.provider,
                &self.agent.model,
                self.chat_mode,
            ),
            prompt,
        );
        self.event_rx = Some(handle.rx);
        cx.notify();
    }

    pub(crate) fn set_model(&mut self, provider: &str, model: &str, cx: &mut Context<Self>) {
        if self.sessions.read(cx).active_chat().busy {
            return;
        }
        self.agent.provider = provider.into();
        self.agent.model = model.into();
        cx.notify();
    }

    pub(crate) fn set_chat_mode(&mut self, mode: ChatMode, cx: &mut Context<Self>) {
        if self.sessions.read(cx).active_chat().busy {
            return;
        }
        self.chat_mode = mode;
        cx.notify();
    }

    pub(crate) fn stop_generation(
        &mut self,
        _: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.sessions.read(cx).active_chat().busy {
            return;
        }
        self.event_rx = None;
        self.sessions.update(cx, |store, cx| {
            store.active_chat_mut().stop();
            cx.notify();
        });
        cx.notify();
    }

    pub fn reset_input(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.text_input.update(cx, |input, cx| {
            input.set_value("", window, cx);
        });
    }
}

impl Panel for ChatPanel {
    fn panel_name(&self) -> &'static str {
        PANEL_CHAT
    }

    fn title(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        "Chat"
    }

    fn closable(&self, _: &App) -> bool {
        false
    }

    fn inner_padding(&self, _: &App) -> bool {
        false
    }
}

impl EventEmitter<PanelEvent> for ChatPanel {}
impl Focusable for ChatPanel {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ChatPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let chat = self.sessions.read(cx).active_chat().clone();
        let busy = chat.busy;
        let show_empty = show_empty_state(&chat);
        let input_text = self.text_input.read(cx).value().to_string();
        let can_send = !busy && !input_text.trim().is_empty();
        let model_label = short_model_label(&self.agent.model);
        let mode_label = self.chat_mode.label();
        let entity = cx.entity();
        let provider = self.agent.provider.clone();
        let model = self.agent.model.clone();
        let chat_mode = self.chat_mode;

        div()
            .size_full()
            .v_flex()
            .bg(cx.theme().background)
            .child(
                div()
                    .flex_grow()
                    .overflow_hidden()
                    .when(show_empty, |pane| {
                        pane.p_4().child(
                            recessed_well(cx)
                                .size_full()
                                .flex()
                                .flex_col()
                                .items_center()
                                .justify_center()
                                .child(render_empty_state(entity.clone(), cx)),
                        )
                    })
                    .when(!show_empty, |pane| {
                        pane.child(message_list(
                            self,
                            self.message_sizes.clone(),
                            &self.scroll_handle,
                            entity.clone(),
                            cx,
                        ))
                    }),
            )
            .child(render_composer(
                self,
                busy,
                can_send,
                mode_label,
                chat_mode,
                &model_label,
                &provider,
                &model,
                entity,
                cx,
            ))
    }
}

fn short_model_label(model: &str) -> String {
    let m = model.trim();
    if m.len() <= 22 {
        return m.to_string();
    }
    format!("{}…", m.chars().take(21).collect::<String>())
}
