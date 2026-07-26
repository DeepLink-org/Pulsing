use crate::agent::AgentConfig;
use crate::protocol::{
    CancelTurn, CommandEnvelope, CommandId, CommandReceipt, CreateSession, ForgeEventKind,
    ForgeProtocolError, SessionId, SessionSpec, StartTurn, TurnId,
};

use super::{EventSubscription, ForgeService, SessionSnapshot};

#[derive(Clone, Default)]
pub struct LocalForgeClient {
    service: ForgeService,
}

impl LocalForgeClient {
    pub fn new(service: ForgeService) -> Self {
        Self { service }
    }

    pub fn service(&self) -> &ForgeService {
        &self.service
    }

    pub async fn create_session(
        &self,
        config: AgentConfig,
    ) -> Result<SessionId, ForgeProtocolError> {
        let session_id = SessionId::new();
        self.create_session_with(CommandId::new(), session_id.clone(), config)
            .await?;
        Ok(session_id)
    }

    pub async fn create_session_with(
        &self,
        command_id: CommandId,
        session_id: SessionId,
        config: AgentConfig,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        self.service
            .create_session(CommandEnvelope::new(
                command_id,
                session_id,
                CreateSession {
                    spec: SessionSpec::from(config),
                },
            ))
            .await
    }

    pub async fn start_turn(
        &self,
        session_id: SessionId,
        input: impl Into<String>,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        self.start_turn_with(CommandId::new(), session_id, TurnId::new(), input.into())
            .await
    }

    pub async fn start_turn_with(
        &self,
        command_id: CommandId,
        session_id: SessionId,
        turn_id: TurnId,
        input: String,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        self.service
            .start_turn(CommandEnvelope::new(
                command_id,
                session_id,
                StartTurn { turn_id, input },
            ))
            .await
    }

    pub async fn cancel_turn(
        &self,
        session_id: SessionId,
        turn_id: TurnId,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        self.service
            .cancel_turn(CommandEnvelope::new(
                CommandId::new(),
                session_id,
                CancelTurn { turn_id },
            ))
            .await
    }

    pub async fn snapshot(
        &self,
        session_id: &SessionId,
    ) -> Result<SessionSnapshot, ForgeProtocolError> {
        self.service.snapshot(session_id).await
    }

    pub async fn subscribe(
        &self,
        session_id: &SessionId,
        after_seq: u64,
    ) -> Result<EventSubscription, ForgeProtocolError> {
        self.service.subscribe(session_id, after_seq).await
    }

    pub async fn run_turn(
        &self,
        session_id: SessionId,
        input: impl Into<String>,
    ) -> Result<String, ForgeProtocolError> {
        let receipt = self.start_turn(session_id.clone(), input).await?;
        let turn_id = receipt
            .turn_id
            .clone()
            .ok_or_else(|| ForgeProtocolError::Internal("start_turn returned no turn_id".into()))?;
        let mut events = self.subscribe(&session_id, receipt.accepted_seq).await?;
        loop {
            let event = events.recv().await?;
            if event.turn_id.as_ref() != Some(&turn_id) {
                continue;
            }
            match event.kind {
                ForgeEventKind::TurnCompleted { text } => return Ok(text),
                ForgeEventKind::TurnFailed { message } => {
                    return Err(ForgeProtocolError::Agent(message));
                }
                ForgeEventKind::TurnCancelled => {
                    return Err(ForgeProtocolError::Agent("turn cancelled".into()));
                }
                _ => {}
            }
        }
    }
}
