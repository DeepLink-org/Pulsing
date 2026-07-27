use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::agent::{AgentCancelled, AgentEvent, ForgeAgent};
use crate::protocol::{
    CancelTurn, CommandEnvelope, CommandId, CommandReceipt, CreateSession, ForgeEvent,
    ForgeEventKind, ForgeProtocolError, SessionId, StartTurn, TurnId,
};
use crate::turn::TurnExecutionContext;
use tokio::sync::{Mutex, RwLock, broadcast};

use super::reducer::SessionSnapshot;
use super::store::{EventStore, InMemoryEventStore};

const EVENT_CHANNEL_CAPACITY: usize = 256;
const TURN_RESOURCE_CLEANUP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Clone)]
pub struct ForgeService {
    inner: Arc<ServiceInner>,
}

struct ServiceInner {
    store: Arc<dyn EventStore>,
    sessions: RwLock<HashMap<SessionId, Arc<SessionRuntime>>>,
}

struct SessionRuntime {
    agent: Mutex<ForgeAgent>,
    control: Mutex<ControlState>,
    events: broadcast::Sender<ForgeEvent>,
}

struct ControlState {
    snapshot: SessionSnapshot,
    receipts: HashMap<CommandId, CommandReceipt>,
    active_execution: Option<(TurnId, Arc<TurnExecutionContext>)>,
}

pub struct EventSubscription {
    replay: VecDeque<ForgeEvent>,
    live: broadcast::Receiver<ForgeEvent>,
    last_seq: u64,
}

impl EventSubscription {
    pub async fn recv(&mut self) -> Result<ForgeEvent, ForgeProtocolError> {
        if let Some(event) = self.replay.pop_front() {
            self.last_seq = self.last_seq.max(event.seq);
            return Ok(event);
        }
        loop {
            match self.live.recv().await {
                Ok(event) if event.seq <= self.last_seq => continue,
                Ok(event) => {
                    self.last_seq = event.seq;
                    return Ok(event);
                }
                Err(broadcast::error::RecvError::Lagged(count)) => {
                    return Err(ForgeProtocolError::SubscriptionLagged(count));
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(ForgeProtocolError::SubscriptionClosed);
                }
            }
        }
    }
}

impl Default for ForgeService {
    fn default() -> Self {
        Self::new()
    }
}

impl ForgeService {
    pub fn new() -> Self {
        Self::with_store(Arc::new(InMemoryEventStore::new()))
    }

    pub fn with_store(store: Arc<dyn EventStore>) -> Self {
        Self {
            inner: Arc::new(ServiceInner {
                store,
                sessions: RwLock::new(HashMap::new()),
            }),
        }
    }

    pub async fn create_session(
        &self,
        command: CommandEnvelope<CreateSession>,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        validate_command(&command)?;
        if let Some(existing) = self
            .inner
            .sessions
            .read()
            .await
            .get(&command.session_id)
            .cloned()
        {
            let control = existing.control.lock().await;
            return control
                .receipts
                .get(&command.command_id)
                .cloned()
                .ok_or_else(|| {
                    ForgeProtocolError::SessionAlreadyExists(command.session_id.clone())
                });
        }

        let agent = ForgeAgent::try_new(command.payload.spec.clone().into())
            .map_err(|err| ForgeProtocolError::Agent(err.to_string()))?;
        let (event_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let mut snapshot =
            SessionSnapshot::uninitialized(command.session_id.clone(), command.payload.spec);
        let event = ForgeEvent::new(
            command.session_id.clone(),
            1,
            None,
            Some(command.command_id.clone()),
            ForgeEventKind::SessionCreated,
        );
        self.inner.store.append(event.clone())?;
        snapshot.apply(&event)?;
        let receipt = CommandReceipt {
            command_id: command.command_id.clone(),
            session_id: command.session_id.clone(),
            turn_id: None,
            accepted_seq: event.seq,
        };
        let runtime = Arc::new(SessionRuntime {
            agent: Mutex::new(agent),
            control: Mutex::new(ControlState {
                snapshot,
                receipts: HashMap::from([(command.command_id, receipt.clone())]),
                active_execution: None,
            }),
            events: event_tx,
        });

        let mut sessions = self.inner.sessions.write().await;
        if sessions.contains_key(&command.session_id) {
            return Err(ForgeProtocolError::SessionAlreadyExists(command.session_id));
        }
        sessions.insert(receipt.session_id.clone(), runtime);
        Ok(receipt)
    }

    pub async fn start_turn(
        &self,
        command: CommandEnvelope<StartTurn>,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        validate_command(&command)?;
        let runtime = self.session(&command.session_id).await?;
        let execution = Arc::new(TurnExecutionContext::new(
            command.session_id.clone(),
            command.payload.turn_id.clone(),
        ));
        let event;
        let receipt;
        {
            let mut control = runtime.control.lock().await;
            if let Some(existing) = control.receipts.get(&command.command_id) {
                return Ok(existing.clone());
            }
            check_expected_seq(command.expected_seq, control.snapshot.last_seq)?;
            control.snapshot.ensure_can_start()?;

            event = ForgeEvent::new(
                command.session_id.clone(),
                control.snapshot.last_seq + 1,
                Some(command.payload.turn_id.clone()),
                Some(command.command_id.clone()),
                ForgeEventKind::TurnStarted {
                    input: command.payload.input.clone(),
                },
            );
            self.inner.store.append(event.clone())?;
            control.snapshot.apply(&event)?;
            control.active_execution = Some((command.payload.turn_id.clone(), execution.clone()));
            receipt = CommandReceipt {
                command_id: command.command_id.clone(),
                session_id: command.session_id.clone(),
                turn_id: Some(command.payload.turn_id.clone()),
                accepted_seq: event.seq,
            };
            control
                .receipts
                .insert(command.command_id.clone(), receipt.clone());
        }
        let _ = runtime.events.send(event);

        let service = self.clone();
        let session_id = command.session_id;
        let turn_id = command.payload.turn_id;
        let input = command.payload.input;
        tokio::spawn(async move {
            service
                .run_turn_task(runtime, session_id, turn_id, input, execution)
                .await;
        });
        Ok(receipt)
    }

    pub async fn cancel_turn(
        &self,
        command: CommandEnvelope<CancelTurn>,
    ) -> Result<CommandReceipt, ForgeProtocolError> {
        validate_command(&command)?;
        let runtime = self.session(&command.session_id).await?;
        let event;
        let receipt;
        let execution;
        {
            let mut control = runtime.control.lock().await;
            if let Some(existing) = control.receipts.get(&command.command_id) {
                return Ok(existing.clone());
            }
            check_expected_seq(command.expected_seq, control.snapshot.last_seq)?;
            let Some((active_turn, active_execution)) = control.active_execution.as_ref() else {
                return Err(ForgeProtocolError::TurnNotActive(command.payload.turn_id));
            };
            if active_turn != &command.payload.turn_id {
                return Err(ForgeProtocolError::TurnNotActive(command.payload.turn_id));
            }
            execution = active_execution.clone();
            event = ForgeEvent::new(
                command.session_id.clone(),
                control.snapshot.last_seq + 1,
                Some(command.payload.turn_id.clone()),
                Some(command.command_id.clone()),
                ForgeEventKind::TurnCancelRequested,
            );
            self.inner.store.append(event.clone())?;
            control.snapshot.apply(&event)?;
            receipt = CommandReceipt {
                command_id: command.command_id.clone(),
                session_id: command.session_id.clone(),
                turn_id: Some(command.payload.turn_id.clone()),
                accepted_seq: event.seq,
            };
            control.receipts.insert(command.command_id, receipt.clone());
        }
        let _ = runtime.events.send(event);
        execution.cancel();
        Ok(receipt)
    }

    pub async fn snapshot(
        &self,
        session_id: &SessionId,
    ) -> Result<SessionSnapshot, ForgeProtocolError> {
        let runtime = self.session(session_id).await?;
        let snapshot = runtime.control.lock().await.snapshot.clone();
        Ok(snapshot)
    }

    pub async fn subscribe(
        &self,
        session_id: &SessionId,
        after_seq: u64,
    ) -> Result<EventSubscription, ForgeProtocolError> {
        let runtime = self.session(session_id).await?;
        // Subscribe before loading history so events emitted during the load are
        // either present in replay or in the live channel. Consumers deduplicate.
        let live = runtime.events.subscribe();
        let replay = self.inner.store.load(session_id, after_seq)?.into();
        Ok(EventSubscription {
            replay,
            live,
            last_seq: after_seq,
        })
    }

    async fn session(
        &self,
        session_id: &SessionId,
    ) -> Result<Arc<SessionRuntime>, ForgeProtocolError> {
        self.inner
            .sessions
            .read()
            .await
            .get(session_id)
            .cloned()
            .ok_or_else(|| ForgeProtocolError::SessionNotFound(session_id.clone()))
    }

    async fn run_turn_task(
        &self,
        runtime: Arc<SessionRuntime>,
        session_id: SessionId,
        turn_id: TurnId,
        input: String,
        execution: Arc<TurnExecutionContext>,
    ) {
        let observer_service = self.clone();
        let observer_runtime = runtime.clone();
        let observer_turn = turn_id.clone();
        let observer = Arc::new(move |agent_event: AgentEvent| {
            let service = observer_service.clone();
            let runtime = observer_runtime.clone();
            let turn = observer_turn.clone();
            Box::pin(async move {
                let kind = match agent_event {
                    AgentEvent::TextDelta(delta) => Some(ForgeEventKind::TurnOutputDelta { delta }),
                    AgentEvent::ToolStart { name } => Some(ForgeEventKind::ToolStarted { name }),
                    AgentEvent::ToolEnd { name, ok, summary } => {
                        Some(ForgeEventKind::ToolCompleted { name, ok, summary })
                    }
                    AgentEvent::ToolCancelled { name } => {
                        Some(ForgeEventKind::ToolCancelled { name })
                    }
                    AgentEvent::Error(_) | AgentEvent::Done { .. } | AgentEvent::Cancelled => None,
                };
                if let Some(kind) = kind {
                    let _ = service.record_event(&runtime, Some(turn), None, kind).await;
                }
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        });

        let result = {
            let mut agent = runtime.agent.lock().await;
            agent.set_event_handler(Some(observer));
            let result = agent.run_in_turn(&input, execution.clone()).await;
            agent.set_event_handler(None);
            result
        };

        if !execution
            .cleanup_and_wait(TURN_RESOURCE_CLEANUP_TIMEOUT)
            .await
        {
            let _ = self
                .record_event(
                    &runtime,
                    Some(turn_id.clone()),
                    None,
                    ForgeEventKind::TurnCleanupStalled {
                        resources: execution.resources().active_kinds(),
                    },
                )
                .await;
            execution.resources().wait_for_idle_unbounded().await;
        }

        let kind = match result {
            _ if execution.is_cancelled() => ForgeEventKind::TurnCancelled,
            Ok(text) => ForgeEventKind::TurnCompleted { text },
            Err(err) if err.downcast_ref::<AgentCancelled>().is_some() => {
                ForgeEventKind::TurnCancelled
            }
            Err(err) => ForgeEventKind::TurnFailed {
                message: err.to_string(),
            },
        };
        let _ = self
            .record_terminal_event(&runtime, &session_id, turn_id, kind)
            .await;
    }

    async fn record_event(
        &self,
        runtime: &SessionRuntime,
        turn_id: Option<TurnId>,
        causation_id: Option<CommandId>,
        kind: ForgeEventKind,
    ) -> Result<ForgeEvent, ForgeProtocolError> {
        let event;
        {
            let mut control = runtime.control.lock().await;
            event = ForgeEvent::new(
                control.snapshot.id.clone(),
                control.snapshot.last_seq + 1,
                turn_id,
                causation_id,
                kind,
            );
            self.inner.store.append(event.clone())?;
            control.snapshot.apply(&event)?;
        }
        let _ = runtime.events.send(event.clone());
        Ok(event)
    }

    async fn record_terminal_event(
        &self,
        runtime: &SessionRuntime,
        session_id: &SessionId,
        turn_id: TurnId,
        kind: ForgeEventKind,
    ) -> Result<ForgeEvent, ForgeProtocolError> {
        let event;
        {
            let mut control = runtime.control.lock().await;
            event = ForgeEvent::new(
                session_id.clone(),
                control.snapshot.last_seq + 1,
                Some(turn_id.clone()),
                None,
                kind,
            );
            self.inner.store.append(event.clone())?;
            control.snapshot.apply(&event)?;
            if control
                .active_execution
                .as_ref()
                .is_some_and(|(active, _)| active == &turn_id)
            {
                control.active_execution = None;
            }
        }
        let _ = runtime.events.send(event.clone());
        Ok(event)
    }
}

fn check_expected_seq(expected: Option<u64>, actual: u64) -> Result<(), ForgeProtocolError> {
    if let Some(expected) = expected
        && expected != actual
    {
        return Err(ForgeProtocolError::SequenceConflict { expected, actual });
    }
    Ok(())
}

fn validate_command<T>(command: &CommandEnvelope<T>) -> Result<(), ForgeProtocolError> {
    if command.protocol != "forge.session" {
        return Err(ForgeProtocolError::InvalidProtocol {
            expected: "forge.session",
            actual: command.protocol.clone(),
        });
    }
    if command.version.major != 1 {
        return Err(ForgeProtocolError::UnsupportedVersion {
            protocol: command.protocol.clone(),
            major: command.version.major,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentConfig;
    use crate::protocol::{CreateSession, SessionSpec};

    async fn create(service: &ForgeService) -> (SessionId, CommandReceipt) {
        let session_id = SessionId::new();
        let command = CommandEnvelope::new(
            CommandId::new(),
            session_id.clone(),
            CreateSession {
                spec: SessionSpec::from(AgentConfig {
                    provider: "demo".into(),
                    model: "demo".into(),
                    ..AgentConfig::default()
                }),
            },
        );
        let receipt = service.create_session(command).await.unwrap();
        (session_id, receipt)
    }

    #[tokio::test]
    async fn duplicate_start_command_is_idempotent() {
        let service = ForgeService::new();
        let (session, _) = create(&service).await;
        let command_id = CommandId::new();
        let command = CommandEnvelope::new(
            command_id,
            session.clone(),
            StartTurn {
                turn_id: TurnId::new(),
                input: "hello".into(),
            },
        );
        let first = service.start_turn(command.clone()).await.unwrap();
        let second = service.start_turn(command).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn rejects_unsupported_command_major_version() {
        let service = ForgeService::new();
        let session_id = SessionId::new();
        let mut command = CommandEnvelope::new(
            CommandId::new(),
            session_id,
            CreateSession {
                spec: AgentConfig::default().into(),
            },
        );
        command.version.major = 2;
        assert!(matches!(
            service.create_session(command).await,
            Err(ForgeProtocolError::UnsupportedVersion { major: 2, .. })
        ));
    }

    #[tokio::test]
    async fn subscription_replays_from_cursor() {
        let service = ForgeService::new();
        let (session, created) = create(&service).await;
        let events = service.inner.store.load(&session, 0).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].seq, created.accepted_seq);

        let mut subscription = service.subscribe(&session, 0).await.unwrap();
        let replay = subscription.recv().await.unwrap();
        assert!(matches!(replay.kind, ForgeEventKind::SessionCreated));
    }

    #[tokio::test]
    async fn local_client_runs_multiple_turns_in_one_session() {
        let client = super::super::LocalForgeClient::default();
        let session = client
            .create_session(AgentConfig {
                provider: "demo".into(),
                model: "demo".into(),
                ..AgentConfig::default()
            })
            .await
            .unwrap();
        let first = client
            .run_turn(session.clone(), "remember alpha")
            .await
            .unwrap();
        let second = client
            .run_turn(session.clone(), "remember beta")
            .await
            .unwrap();
        assert!(first.contains("alpha"));
        assert!(second.contains("beta"));

        let snapshot = client.snapshot(&session).await.unwrap();
        assert_eq!(snapshot.turns.len(), 2);
        assert!(snapshot.active_turn.is_none());
    }
}
