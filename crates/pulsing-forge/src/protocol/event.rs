use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::{CommandId, EventId, SessionId, TurnId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const V1: Self = Self { major: 1, minor: 0 };
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum ForgeEventKind {
    SessionCreated,
    SessionClosed,
    TurnStarted {
        input: String,
    },
    TurnOutputDelta {
        delta: String,
    },
    TurnCancelRequested,
    TurnCleanupStalled {
        resources: Vec<String>,
    },
    TurnCancelled,
    TurnCompleted {
        text: String,
    },
    TurnFailed {
        message: String,
    },
    ToolStarted {
        name: String,
    },
    ToolCompleted {
        name: String,
        ok: bool,
        summary: String,
    },
    ToolCancelled {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeEvent {
    pub protocol: String,
    pub version: ProtocolVersion,
    pub event_id: EventId,
    pub session_id: SessionId,
    pub seq: u64,
    pub occurred_at: DateTime<Utc>,
    pub turn_id: Option<TurnId>,
    pub causation_id: Option<CommandId>,
    #[serde(flatten)]
    pub kind: ForgeEventKind,
}

impl ForgeEvent {
    pub fn new(
        session_id: SessionId,
        seq: u64,
        turn_id: Option<TurnId>,
        causation_id: Option<CommandId>,
        kind: ForgeEventKind,
    ) -> Self {
        Self {
            protocol: "forge.event".into(),
            version: ProtocolVersion::V1,
            event_id: EventId::new(),
            session_id,
            seq,
            occurred_at: Utc::now(),
            turn_id,
            causation_id,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_kind_is_flattened_into_the_versioned_envelope() {
        let event = ForgeEvent::new(
            SessionId::from_string("session-1"),
            7,
            Some(TurnId::from_string("turn-1")),
            None,
            ForgeEventKind::TurnOutputDelta {
                delta: "hello".into(),
            },
        );

        let value = serde_json::to_value(&event).expect("serialize event");
        assert_eq!(value["protocol"], "forge.event");
        assert_eq!(value["version"]["major"], 1);
        assert_eq!(value["kind"], "turn_output_delta");
        assert_eq!(value["payload"]["delta"], "hello");

        let decoded: ForgeEvent = serde_json::from_value(value).expect("deserialize event");
        assert_eq!(decoded, event);
    }
}
