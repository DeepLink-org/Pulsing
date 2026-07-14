use pulsing_forge::AgentEvent;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MessageKind {
    User(String),
    Assistant {
        body: String,
        streaming: bool,
    },
    Tool {
        name: String,
        running: bool,
        ok: bool,
        detail: String,
    },
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub kind: MessageKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub busy: bool,
}

impl ChatState {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn session_title(&self) -> String {
        self.messages
            .iter()
            .find_map(|m| {
                if let MessageKind::User(text) = &m.kind {
                    let t = text.trim();
                    if t.is_empty() {
                        None
                    } else if t.chars().count() > 28 {
                        Some(format!("{}…", t.chars().take(28).collect::<String>()))
                    } else {
                        Some(t.to_string())
                    }
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "New Chat".into())
    }

    pub fn apply(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta(delta) => {
                if let Some(ChatMessage {
                    kind: MessageKind::Assistant { body, streaming },
                }) = self.messages.last_mut()
                {
                    if *streaming {
                        body.push_str(&delta);
                        return;
                    }
                }
                self.messages.push(ChatMessage {
                    kind: MessageKind::Assistant {
                        body: delta,
                        streaming: true,
                    },
                });
            }
            AgentEvent::ToolStart { name } => {
                self.finish_streaming_assistant();
                self.messages.push(ChatMessage {
                    kind: MessageKind::Tool {
                        name,
                        running: true,
                        ok: true,
                        detail: String::new(),
                    },
                });
            }
            AgentEvent::ToolEnd { name, ok, summary } => {
                if let Some(ChatMessage {
                    kind: MessageKind::Tool {
                        name: n,
                        running,
                        ok: tool_ok,
                        detail,
                    },
                }) = self.messages.iter_mut().rev().find(|m| {
                    matches!(m.kind, MessageKind::Tool { name: ref tn, running: true, .. } if tn == &name)
                }) {
                    *running = false;
                    *tool_ok = ok;
                    *detail = summary;
                    *n = name;
                } else {
                    self.messages.push(ChatMessage {
                        kind: MessageKind::Tool {
                            name,
                            running: false,
                            ok,
                            detail: summary,
                        },
                    });
                }
                self.messages.push(ChatMessage {
                    kind: MessageKind::Assistant {
                        body: String::new(),
                        streaming: true,
                    },
                });
            }
            AgentEvent::Done { text } => {
                self.finish_streaming_assistant();
                if let Some(ChatMessage {
                    kind: MessageKind::Assistant { body, streaming },
                }) = self.messages.last_mut()
                {
                    if body.is_empty() {
                        *body = text;
                    }
                    *streaming = false;
                }
                self.busy = false;
            }
            AgentEvent::Error(message) => {
                self.finish_streaming_assistant();
                self.messages.push(ChatMessage {
                    kind: MessageKind::Error(message),
                });
                self.busy = false;
            }
        }
    }

    pub fn stop(&mut self) {
        self.finish_streaming_assistant();
        self.busy = false;
    }

    fn finish_streaming_assistant(&mut self) {
        if let Some(ChatMessage {
            kind: MessageKind::Assistant { streaming, .. },
        }) = self.messages.last_mut()
        {
            *streaming = false;
        }
    }
}
