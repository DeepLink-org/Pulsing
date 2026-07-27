//! Agent UI events — consumed by pulsing-cli GUI / TUI renderers.

use std::sync::mpsc::Sender;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TextDelta(String),
    ToolStart {
        name: String,
    },
    ToolEnd {
        name: String,
        ok: bool,
        summary: String,
    },
    ToolCancelled {
        name: String,
    },
    Error(String),
    Cancelled,
    Done {
        text: String,
    },
}

pub type AgentEventTx = Sender<AgentEvent>;

pub(crate) fn emit(tx: &Option<AgentEventTx>, event: AgentEvent) {
    if let Some(tx) = tx {
        let _ = tx.send(event);
    }
}
