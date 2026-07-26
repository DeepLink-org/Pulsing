use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::protocol::{
    ForgeEvent, ForgeEventKind, ForgeProtocolError, SessionId, SessionSpec, TurnId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Running,
    Cancelling,
    Unknown,
    Completed,
    Failed,
    Cancelled,
}

impl TurnStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSnapshot {
    pub id: TurnId,
    pub input: String,
    pub status: TurnStatus,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub id: SessionId,
    pub spec: SessionSpec,
    pub status: SessionStatus,
    pub last_seq: u64,
    pub active_turn: Option<TurnId>,
    pub turns: HashMap<TurnId, TurnSnapshot>,
}

impl SessionSnapshot {
    pub(crate) fn uninitialized(id: SessionId, spec: SessionSpec) -> Self {
        Self {
            id,
            spec,
            status: SessionStatus::Active,
            last_seq: 0,
            active_turn: None,
            turns: HashMap::new(),
        }
    }

    pub(crate) fn ensure_can_start(&self) -> Result<(), ForgeProtocolError> {
        if self.status == SessionStatus::Closed {
            return Err(ForgeProtocolError::SessionClosed(self.id.clone()));
        }
        if self.active_turn.is_some() {
            return Err(ForgeProtocolError::SessionBusy(self.id.clone()));
        }
        Ok(())
    }

    pub(crate) fn apply(&mut self, event: &ForgeEvent) -> Result<(), ForgeProtocolError> {
        if event.session_id != self.id {
            return Err(ForgeProtocolError::InvalidTransition(format!(
                "event belongs to {}, reducer belongs to {}",
                event.session_id, self.id
            )));
        }
        let expected = self.last_seq + 1;
        if event.seq != expected {
            return Err(ForgeProtocolError::SequenceConflict {
                expected,
                actual: event.seq,
            });
        }

        match &event.kind {
            ForgeEventKind::SessionCreated => {
                if self.last_seq != 0 {
                    return Err(ForgeProtocolError::InvalidTransition(
                        "session.created must be the first event".into(),
                    ));
                }
            }
            ForgeEventKind::SessionClosed => {
                if self.active_turn.is_some() {
                    return Err(ForgeProtocolError::InvalidTransition(
                        "cannot close a session with an active turn".into(),
                    ));
                }
                self.status = SessionStatus::Closed;
            }
            ForgeEventKind::TurnStarted { input } => {
                self.ensure_can_start()?;
                let turn_id = event.turn_id.clone().ok_or_else(|| {
                    ForgeProtocolError::InvalidTransition("turn.started missing turn_id".into())
                })?;
                self.turns.insert(
                    turn_id.clone(),
                    TurnSnapshot {
                        id: turn_id.clone(),
                        input: input.clone(),
                        status: TurnStatus::Running,
                        output: None,
                        error: None,
                    },
                );
                self.active_turn = Some(turn_id);
            }
            ForgeEventKind::TurnCancelRequested => {
                let turn = self.active_turn_mut(event)?;
                if turn.status != TurnStatus::Running {
                    return Err(ForgeProtocolError::InvalidTransition(format!(
                        "cannot cancel turn in {:?}",
                        turn.status
                    )));
                }
                turn.status = TurnStatus::Cancelling;
            }
            ForgeEventKind::TurnCancelled => {
                let turn = self.active_turn_mut(event)?;
                if !matches!(
                    turn.status,
                    TurnStatus::Running | TurnStatus::Cancelling | TurnStatus::Unknown
                ) {
                    return Err(ForgeProtocolError::InvalidTransition(format!(
                        "cannot complete cancellation from {:?}",
                        turn.status
                    )));
                }
                turn.status = TurnStatus::Cancelled;
                self.active_turn = None;
            }
            ForgeEventKind::TurnCleanupStalled { resources } => {
                let turn = self.active_turn_mut(event)?;
                turn.status = TurnStatus::Unknown;
                turn.error = Some(format!(
                    "waiting for turn resources to stop: {}",
                    resources.join(", ")
                ));
            }
            ForgeEventKind::TurnCompleted { text } => {
                let turn = self.active_turn_mut(event)?;
                turn.status = TurnStatus::Completed;
                turn.output = Some(text.clone());
                self.active_turn = None;
            }
            ForgeEventKind::TurnFailed { message } => {
                let turn = self.active_turn_mut(event)?;
                turn.status = TurnStatus::Failed;
                turn.error = Some(message.clone());
                self.active_turn = None;
            }
            ForgeEventKind::TurnOutputDelta { .. }
            | ForgeEventKind::ToolStarted { .. }
            | ForgeEventKind::ToolCompleted { .. }
            | ForgeEventKind::ToolCancelled { .. } => {
                let turn = self.active_turn_ref(event)?;
                if turn.status.is_terminal() {
                    return Err(ForgeProtocolError::InvalidTransition(
                        "activity after terminal turn".into(),
                    ));
                }
            }
        }

        self.last_seq = event.seq;
        Ok(())
    }

    fn event_turn_id<'a>(
        &'a self,
        event: &'a ForgeEvent,
    ) -> Result<&'a TurnId, ForgeProtocolError> {
        event.turn_id.as_ref().ok_or_else(|| {
            ForgeProtocolError::InvalidTransition("turn event missing turn_id".into())
        })
    }

    fn active_turn_ref(&self, event: &ForgeEvent) -> Result<&TurnSnapshot, ForgeProtocolError> {
        let turn_id = self.event_turn_id(event)?;
        if self.active_turn.as_ref() != Some(turn_id) {
            return Err(ForgeProtocolError::TurnNotActive(turn_id.clone()));
        }
        self.turns
            .get(turn_id)
            .ok_or_else(|| ForgeProtocolError::TurnNotActive(turn_id.clone()))
    }

    fn active_turn_mut(
        &mut self,
        event: &ForgeEvent,
    ) -> Result<&mut TurnSnapshot, ForgeProtocolError> {
        let turn_id = event.turn_id.clone().ok_or_else(|| {
            ForgeProtocolError::InvalidTransition("turn event missing turn_id".into())
        })?;
        if self.active_turn.as_ref() != Some(&turn_id) {
            return Err(ForgeProtocolError::TurnNotActive(turn_id));
        }
        self.turns
            .get_mut(&turn_id)
            .ok_or(ForgeProtocolError::TurnNotActive(turn_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentConfig;
    use crate::protocol::{CommandId, ForgeEventKind};

    fn state() -> SessionSnapshot {
        let id = SessionId::new();
        let mut state = SessionSnapshot::uninitialized(id.clone(), AgentConfig::default().into());
        state
            .apply(&ForgeEvent::new(
                id,
                1,
                None,
                Some(CommandId::new()),
                ForgeEventKind::SessionCreated,
            ))
            .unwrap();
        state
    }

    fn event(state: &SessionSnapshot, turn: Option<TurnId>, kind: ForgeEventKind) -> ForgeEvent {
        ForgeEvent::new(
            state.id.clone(),
            state.last_seq + 1,
            turn,
            Some(CommandId::new()),
            kind,
        )
    }

    #[test]
    fn enforces_single_active_turn() {
        let mut state = state();
        let turn = TurnId::new();
        state
            .apply(&event(
                &state,
                Some(turn.clone()),
                ForgeEventKind::TurnStarted {
                    input: "one".into(),
                },
            ))
            .unwrap();
        let second = event(
            &state,
            Some(TurnId::new()),
            ForgeEventKind::TurnStarted {
                input: "two".into(),
            },
        );
        assert!(matches!(
            state.apply(&second),
            Err(ForgeProtocolError::SessionBusy(_))
        ));
    }

    #[test]
    fn cancellation_is_not_terminal_until_cancelled_event() {
        let mut state = state();
        let turn = TurnId::new();
        state
            .apply(&event(
                &state,
                Some(turn.clone()),
                ForgeEventKind::TurnStarted {
                    input: "one".into(),
                },
            ))
            .unwrap();
        state
            .apply(&event(
                &state,
                Some(turn.clone()),
                ForgeEventKind::TurnCancelRequested,
            ))
            .unwrap();
        assert_eq!(state.active_turn, Some(turn.clone()));
        assert_eq!(state.turns[&turn].status, TurnStatus::Cancelling);

        state
            .apply(&event(
                &state,
                Some(turn.clone()),
                ForgeEventKind::TurnCancelled,
            ))
            .unwrap();
        assert_eq!(state.active_turn, None);
        assert_eq!(state.turns[&turn].status, TurnStatus::Cancelled);
    }

    #[test]
    fn rejects_event_sequence_gaps() {
        let mut state = state();
        let event = ForgeEvent::new(
            state.id.clone(),
            state.last_seq + 2,
            Some(TurnId::new()),
            None,
            ForgeEventKind::TurnStarted {
                input: "gap".into(),
            },
        );
        assert!(matches!(
            state.apply(&event),
            Err(ForgeProtocolError::SequenceConflict { .. })
        ));
    }
}
