use crate::state::ChatState;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

pub struct ChatSession {
    pub id: SessionId,
    pub chat: ChatState,
}

pub struct SessionStore {
    sessions: Vec<ChatSession>,
    active: SessionId,
    next_id: u64,
}

impl SessionStore {
    pub fn new() -> Self {
        let id = SessionId(1);
        Self {
            sessions: vec![ChatSession {
                id,
                chat: ChatState::empty(),
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

    pub fn session_title(&self, id: SessionId) -> String {
        self.sessions
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.chat.session_title())
            .unwrap_or_else(|| "Chat".into())
    }

    pub fn new_session(&mut self) -> SessionId {
        let id = SessionId(self.next_id);
        self.next_id += 1;
        self.sessions.push(ChatSession {
            id,
            chat: ChatState::empty(),
        });
        self.active = id;
        id
    }

    pub fn focus(&mut self, id: SessionId) {
        if self.sessions.iter().any(|s| s.id == id) {
            self.active = id;
        }
    }
}
