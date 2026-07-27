use std::collections::HashMap;
use std::sync::{mpsc, Arc};
use std::thread;

use pulsing_forge::protocol::{ForgeEventKind, SessionId, TurnId};
use pulsing_forge::{AgentConfig, AgentEvent, LocalForgeClient};
use tokio::sync::{mpsc as tokio_mpsc, Mutex};

enum ControllerCommand {
    StartTurn {
        gui_session_id: u64,
        config: AgentConfig,
        prompt: String,
    },
    CancelTurn {
        gui_session_id: u64,
    },
}

#[derive(Debug)]
pub struct ControllerEvent {
    pub gui_session_id: u64,
    pub event: AgentEvent,
}

#[derive(Clone)]
struct SessionBinding {
    forge_session_id: SessionId,
    active_turn: Option<TurnId>,
}

pub struct ForgeController {
    commands: tokio_mpsc::UnboundedSender<ControllerCommand>,
    events: mpsc::Receiver<ControllerEvent>,
}

impl ForgeController {
    pub fn start() -> Self {
        let (command_tx, command_rx) = tokio_mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::channel();
        thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    let _ = event_tx.send(ControllerEvent {
                        gui_session_id: 0,
                        event: AgentEvent::Error(err.to_string()),
                    });
                    return;
                }
            };
            runtime.block_on(run_controller(command_rx, event_tx));
        });
        Self {
            commands: command_tx,
            events: event_rx,
        }
    }

    pub fn start_turn(&self, gui_session_id: u64, config: AgentConfig, prompt: String) {
        let _ = self.commands.send(ControllerCommand::StartTurn {
            gui_session_id,
            config,
            prompt,
        });
    }

    pub fn cancel_turn(&self, gui_session_id: u64) {
        let _ = self
            .commands
            .send(ControllerCommand::CancelTurn { gui_session_id });
    }

    pub fn try_recv(&self) -> Option<ControllerEvent> {
        self.events.try_recv().ok()
    }
}

async fn run_controller(
    mut commands: tokio_mpsc::UnboundedReceiver<ControllerCommand>,
    events: mpsc::Sender<ControllerEvent>,
) {
    let client = LocalForgeClient::default();
    let sessions: Arc<Mutex<HashMap<u64, SessionBinding>>> = Arc::new(Mutex::new(HashMap::new()));

    while let Some(command) = commands.recv().await {
        match command {
            ControllerCommand::StartTurn {
                gui_session_id,
                config,
                prompt,
            } => match begin_turn(&client, &sessions, gui_session_id, config, prompt).await {
                Ok((forge_session_id, turn_id, after_seq)) => {
                    tokio::spawn(forward_turn_events(
                        client.clone(),
                        sessions.clone(),
                        events.clone(),
                        gui_session_id,
                        forge_session_id,
                        turn_id,
                        after_seq,
                    ));
                }
                Err(err) => {
                    let _ = events.send(ControllerEvent {
                        gui_session_id,
                        event: AgentEvent::Error(err.to_string()),
                    });
                }
            },
            ControllerCommand::CancelTurn { gui_session_id } => {
                let binding = sessions.lock().await.get(&gui_session_id).cloned();
                let Some(SessionBinding {
                    forge_session_id,
                    active_turn: Some(turn_id),
                }) = binding
                else {
                    continue;
                };
                if let Err(err) = client.cancel_turn(forge_session_id, turn_id).await {
                    let _ = events.send(ControllerEvent {
                        gui_session_id,
                        event: AgentEvent::Error(err.to_string()),
                    });
                }
            }
        }
    }
}

async fn begin_turn(
    client: &LocalForgeClient,
    sessions: &Arc<Mutex<HashMap<u64, SessionBinding>>>,
    gui_session_id: u64,
    config: AgentConfig,
    prompt: String,
) -> Result<(SessionId, TurnId, u64), pulsing_forge::protocol::ForgeProtocolError> {
    let forge_session_id = {
        let existing = sessions
            .lock()
            .await
            .get(&gui_session_id)
            .map(|binding| binding.forge_session_id.clone());
        match existing {
            Some(id) => id,
            None => {
                let id = client.create_session(config).await?;
                sessions.lock().await.insert(
                    gui_session_id,
                    SessionBinding {
                        forge_session_id: id.clone(),
                        active_turn: None,
                    },
                );
                id
            }
        }
    };

    let receipt = client.start_turn(forge_session_id.clone(), prompt).await?;
    let turn_id = receipt
        .turn_id
        .clone()
        .expect("start_turn receipt always has turn_id");
    if let Some(binding) = sessions.lock().await.get_mut(&gui_session_id) {
        binding.active_turn = Some(turn_id.clone());
    }
    Ok((forge_session_id, turn_id, receipt.accepted_seq))
}

async fn forward_turn_events(
    client: LocalForgeClient,
    sessions: Arc<Mutex<HashMap<u64, SessionBinding>>>,
    events: mpsc::Sender<ControllerEvent>,
    gui_session_id: u64,
    forge_session_id: SessionId,
    turn_id: TurnId,
    after_seq: u64,
) {
    let mut subscription = client
        .subscribe(&forge_session_id, after_seq)
        .await
        .map_err(|err| err.to_string());
    let subscription = match subscription.as_mut() {
        Ok(subscription) => subscription,
        Err(message) => {
            let _ = events.send(ControllerEvent {
                gui_session_id,
                event: AgentEvent::Error(message.clone()),
            });
            clear_active_turn(&sessions, gui_session_id, &turn_id).await;
            return;
        }
    };
    loop {
        let event = match subscription.recv().await {
            Ok(event) => event,
            Err(err) => {
                let _ = events.send(ControllerEvent {
                    gui_session_id,
                    event: AgentEvent::Error(err.to_string()),
                });
                clear_active_turn(&sessions, gui_session_id, &turn_id).await;
                return;
            }
        };
        if event.turn_id.as_ref() != Some(&turn_id) {
            continue;
        }
        let (agent_event, terminal) = match event.kind {
            ForgeEventKind::TurnOutputDelta { delta } => {
                (Some(AgentEvent::TextDelta(delta)), false)
            }
            ForgeEventKind::ToolStarted { name } => (Some(AgentEvent::ToolStart { name }), false),
            ForgeEventKind::ToolCompleted { name, ok, summary } => {
                (Some(AgentEvent::ToolEnd { name, ok, summary }), false)
            }
            ForgeEventKind::ToolCancelled { name } => {
                (Some(AgentEvent::ToolCancelled { name }), false)
            }
            ForgeEventKind::TurnCompleted { text } => (Some(AgentEvent::Done { text }), true),
            ForgeEventKind::TurnFailed { message } => (Some(AgentEvent::Error(message)), true),
            ForgeEventKind::TurnCancelled => (Some(AgentEvent::Cancelled), true),
            _ => (None, false),
        };
        if let Some(event) = agent_event {
            let _ = events.send(ControllerEvent {
                gui_session_id,
                event,
            });
        }
        if terminal {
            clear_active_turn(&sessions, gui_session_id, &turn_id).await;
            return;
        }
    }
}

async fn clear_active_turn(
    sessions: &Arc<Mutex<HashMap<u64, SessionBinding>>>,
    gui_session_id: u64,
    turn_id: &TurnId,
) {
    if let Some(binding) = sessions.lock().await.get_mut(&gui_session_id) {
        if binding.active_turn.as_ref() == Some(turn_id) {
            binding.active_turn = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    use super::*;

    fn demo_config() -> AgentConfig {
        AgentConfig {
            provider: "demo".into(),
            model: "demo".into(),
            ..AgentConfig::default()
        }
    }

    #[test]
    fn routes_concurrent_session_events_to_their_origin() {
        let controller = ForgeController::start();
        controller.start_turn(11, demo_config(), "remember eleven".into());
        controller.start_turn(22, demo_config(), "remember twenty two".into());

        let deadline = Instant::now() + Duration::from_secs(3);
        let mut completed = HashSet::new();
        while Instant::now() < deadline && completed.len() < 2 {
            if let Some(event) = controller.try_recv() {
                if matches!(event.event, AgentEvent::Done { .. }) {
                    completed.insert(event.gui_session_id);
                }
            } else {
                thread::sleep(Duration::from_millis(10));
            }
        }
        assert_eq!(completed, HashSet::from([11, 22]));
    }
}
