use std::collections::HashMap;
use std::sync::Mutex;

use crate::protocol::{ForgeEvent, ForgeProtocolError, SessionId};

pub trait EventStore: Send + Sync {
    fn append(&self, event: ForgeEvent) -> Result<(), ForgeProtocolError>;
    fn load(
        &self,
        session_id: &SessionId,
        after_seq: u64,
    ) -> Result<Vec<ForgeEvent>, ForgeProtocolError>;
}

#[derive(Default)]
pub struct InMemoryEventStore {
    events: Mutex<HashMap<SessionId, Vec<ForgeEvent>>>,
}

impl InMemoryEventStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl EventStore for InMemoryEventStore {
    fn append(&self, event: ForgeEvent) -> Result<(), ForgeProtocolError> {
        let mut all = self
            .events
            .lock()
            .map_err(|_| ForgeProtocolError::Internal("event store lock poisoned".into()))?;
        let events = all.entry(event.session_id.clone()).or_default();
        let expected = events.last().map_or(1, |last| last.seq + 1);
        if event.seq != expected {
            return Err(ForgeProtocolError::SequenceConflict {
                expected,
                actual: event.seq,
            });
        }
        events.push(event);
        Ok(())
    }

    fn load(
        &self,
        session_id: &SessionId,
        after_seq: u64,
    ) -> Result<Vec<ForgeEvent>, ForgeProtocolError> {
        let all = self
            .events
            .lock()
            .map_err(|_| ForgeProtocolError::Internal("event store lock poisoned".into()))?;
        Ok(all
            .get(session_id)
            .into_iter()
            .flatten()
            .filter(|event| event.seq > after_seq)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ForgeEventKind, SessionId};

    #[test]
    fn append_requires_contiguous_sequence() {
        let store = InMemoryEventStore::new();
        let session = SessionId::new();
        store
            .append(ForgeEvent::new(
                session.clone(),
                1,
                None,
                None,
                ForgeEventKind::SessionCreated,
            ))
            .unwrap();
        let result = store.append(ForgeEvent::new(
            session,
            3,
            None,
            None,
            ForgeEventKind::SessionClosed,
        ));
        assert!(matches!(
            result,
            Err(ForgeProtocolError::SequenceConflict {
                expected: 2,
                actual: 3
            })
        ));
    }
}
