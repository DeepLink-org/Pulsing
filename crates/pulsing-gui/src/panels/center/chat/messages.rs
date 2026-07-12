use gpui::{div, px, AnyElement, Context, IntoElement, ParentElement, Pixels, Styled, Window};
use gpui_component::{v_virtual_list, VirtualListScrollHandle};

use crate::panels::center::chat::bubble::{
    assistant_message, error_message, message_row_shell, tool_message, user_message,
};
use crate::panels::center::chat::ChatPanel;
use crate::state::{ChatMessage, ChatState, MessageKind};

const CHARS_PER_LINE: usize = 52;
const LINE_HEIGHT: f32 = 24.0;
const ROW_PAD: f32 = 56.0;

pub fn estimate_message_height(msg: &ChatMessage) -> Pixels {
    let (lines, min_h, max_h) = match &msg.kind {
        MessageKind::User(text) | MessageKind::Error(text) => {
            let lines = line_count(text);
            (lines, 88.0, 480.0)
        }
        MessageKind::Assistant { body, .. } => {
            let lines = line_count(body).max(1);
            (lines, 88.0, 560.0)
        }
        MessageKind::Tool { .. } => (1, 72.0, 96.0),
    };
    px((lines as f32 * LINE_HEIGHT + ROW_PAD).clamp(min_h, max_h))
}

fn line_count(text: &str) -> usize {
    let chars = text.chars().count().max(1);
    text.lines()
        .count()
        .max(chars.div_ceil(CHARS_PER_LINE))
        .max(1)
}

pub fn show_empty_state(chat: &ChatState) -> bool {
    chat.messages.is_empty() && !chat.busy
}

pub fn message_list(
    _panel: &ChatPanel,
    sizes: std::rc::Rc<Vec<gpui::Size<Pixels>>>,
    scroll_handle: &VirtualListScrollHandle,
    entity: gpui::Entity<ChatPanel>,
    _cx: &mut Context<ChatPanel>,
) -> impl gpui::IntoElement {
    div().size_full().py_4().child(
        v_virtual_list(
            entity,
            "chat-messages",
            sizes,
            |panel, range, window, cx| {
                range
                    .map(|ix| div().w_full().child(panel.render_entry(ix, window, cx)))
                    .collect()
            },
        )
        .track_scroll(scroll_handle),
    )
}

impl ChatPanel {
    pub(crate) fn render_entry(
        &self,
        ix: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(msg) = self
            .sessions
            .read(cx)
            .active_chat()
            .messages
            .get(ix)
            .cloned()
        else {
            return div().into_any_element();
        };

        message_row_shell(match msg.kind {
            MessageKind::User(text) => user_message(cx, &text).into_any_element(),
            MessageKind::Assistant { body, streaming } => {
                if body.is_empty() && !streaming {
                    return div().into_any_element();
                }
                assistant_message(cx, &body, streaming).into_any_element()
            }
            MessageKind::Tool {
                name,
                running,
                ok,
                detail,
            } => tool_message(cx, &name, running, ok, &detail).into_any_element(),
            MessageKind::Error(text) => error_message(cx, &text).into_any_element(),
        })
        .into_any_element()
    }
}
