use crate::settings::ChatMode;
use crate::state::ChatState;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

pub struct ChatSession {
    pub id: SessionId,
    pub chat: ChatState,
    pub provider: String,
    pub model: String,
    pub mode: ChatMode,
}

pub struct SessionStore {
    sessions: Vec<ChatSession>,
    active: SessionId,
    next_id: u64,
}

impl SessionStore {
    pub fn new(provider: String, model: String, mode: ChatMode) -> Self {
        let id = SessionId(1);
        Self {
            sessions: vec![ChatSession {
                id,
                chat: ChatState::empty(),
                provider,
                model,
                mode,
            }],
            active: id,
            next_id: 2,
        }
    }

    pub fn sessions(&self) -> &[ChatSession] {
        &self.sessions
    }

    pub fn active_id(&self) -> SessionId {
        self.active
    }

    pub fn active_chat(&self) -> &ChatState {
        self.sessions
            .iter()
            .find(|s| s.id == self.active)
            .map(|s| &s.chat)
            .expect("active session exists")
    }

    pub fn active_chat_mut(&mut self) -> &mut ChatState {
        let active = self.active;
        self.sessions
            .iter_mut()
            .find(|s| s.id == active)
            .map(|s| &mut s.chat)
            .expect("active session exists")
    }

    pub fn chat_mut(&mut self, id: SessionId) -> Option<&mut ChatState> {
        self.sessions
            .iter_mut()
            .find(|session| session.id == id)
            .map(|session| &mut session.chat)
    }

    pub fn any_busy(&self) -> bool {
        self.sessions.iter().any(|session| session.chat.busy)
    }

    pub fn session_title(&self, id: SessionId) -> String {
        self.sessions
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.chat.session_title())
            .unwrap_or_else(|| "Chat".into())
    }

    pub fn new_session(&mut self, provider: String, model: String, mode: ChatMode) -> SessionId {
        let id = SessionId(self.next_id);
        self.next_id += 1;
        self.sessions.push(ChatSession {
            id,
            chat: ChatState::empty(),
            provider,
            model,
            mode,
        });
        self.active = id;
        id
    }

    pub fn focus(&mut self, id: SessionId) {
        if self.sessions.iter().any(|s| s.id == id) {
            self.active = id;
        }
    }

    pub fn active_settings(&self) -> (&str, &str, ChatMode) {
        let session = self
            .sessions
            .iter()
            .find(|session| session.id == self.active)
            .expect("active session exists");
        (&session.provider, &session.model, session.mode)
    }

    pub fn set_active_settings(&mut self, provider: String, model: String, mode: ChatMode) {
        let active = self.active;
        let session = self
            .sessions
            .iter_mut()
            .find(|session| session.id == active)
            .expect("active session exists");
        session.provider = provider;
        session.model = model;
        session.mode = mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ChatMessage, MessageKind};

    #[test]
    fn background_session_updates_do_not_touch_active_chat() {
        let mut sessions = SessionStore::new("demo".into(), "demo".into(), ChatMode::Agent);
        let first = sessions.active_id();
        let second = sessions.new_session("demo".into(), "demo".into(), ChatMode::Agent);
        sessions
            .chat_mut(first)
            .unwrap()
            .messages
            .push(ChatMessage {
                kind: MessageKind::Assistant {
                    body: "background".into(),
                    streaming: false,
                },
            });

        assert_eq!(sessions.active_id(), second);
        assert!(sessions.active_chat().messages.is_empty());
        assert_eq!(sessions.chat_mut(first).unwrap().messages.len(), 1);
    }
}
