use std::sync::mpsc;
use std::thread;

use pulsing_forge::{run_agent_turn_observed, AgentConfig, AgentEvent};

pub struct ChatTurnHandle {
    pub rx: mpsc::Receiver<AgentEvent>,
}

pub fn start_agent_turn(cfg: AgentConfig, prompt: String) -> ChatTurnHandle {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(err) => {
                let _ = tx.send(AgentEvent::Error(err.to_string()));
                return;
            }
        };
        if let Err(err) = rt.block_on(run_agent_turn_observed(&cfg, &prompt, Some(tx.clone()))) {
            let _ = tx.send(AgentEvent::Error(err.to_string()));
        }
    });
    ChatTurnHandle { rx }
}
