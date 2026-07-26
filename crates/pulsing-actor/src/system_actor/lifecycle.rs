//! Node control-plane lifecycle.

use crate::error::{PulsingError, Result, RuntimeError};
use std::sync::atomic::{AtomicU8, Ordering};

/// Observable state of the node control plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum NodeState {
    Booting = 0,
    Starting = 1,
    Ready = 2,
    Draining = 3,
    Failed = 4,
    Stopped = 5,
}

impl NodeState {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Booting => "booting",
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
        }
    }

    const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Booting,
                Self::Starting | Self::Draining | Self::Failed
            ) | (Self::Starting, Self::Ready | Self::Draining | Self::Failed)
                | (Self::Ready, Self::Draining | Self::Failed)
                | (Self::Draining, Self::Stopped | Self::Failed)
                | (Self::Failed, Self::Draining | Self::Stopped)
        )
    }
}

/// Shared, validated state machine for node bootstrap and shutdown.
pub(crate) struct NodeLifecycle {
    state: AtomicU8,
}

impl NodeLifecycle {
    pub(crate) fn new() -> Self {
        Self {
            state: AtomicU8::new(NodeState::Booting as u8),
        }
    }

    pub(crate) fn state(&self) -> NodeState {
        match self.state.load(Ordering::Acquire) {
            0 => NodeState::Booting,
            1 => NodeState::Starting,
            2 => NodeState::Ready,
            3 => NodeState::Draining,
            4 => NodeState::Failed,
            5 => NodeState::Stopped,
            _ => unreachable!("node lifecycle contains an invalid state"),
        }
    }

    pub(crate) fn transition(&self, next: NodeState) -> Result<()> {
        loop {
            let current = self.state();
            if current == next {
                return Ok(());
            }
            if !current.can_transition_to(next) {
                return Err(lifecycle_error(format!(
                    "invalid node lifecycle transition: {current:?} -> {next:?}"
                )));
            }
            if self
                .state
                .compare_exchange(
                    current as u8,
                    next as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Enter draining unless the node has already reached a terminal path.
    pub(crate) fn begin_draining(&self) -> Result<()> {
        match self.state() {
            NodeState::Draining | NodeState::Stopped => Ok(()),
            NodeState::Failed => self.transition(NodeState::Draining),
            _ => self.transition(NodeState::Draining),
        }
    }
}

impl Default for NodeLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

fn lifecycle_error(message: impl Into<String>) -> PulsingError {
    PulsingError::from(RuntimeError::Other(format!(
        "System lifecycle error: {}",
        message.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_accepts_boot_and_shutdown_path() {
        let lifecycle = NodeLifecycle::new();
        lifecycle.transition(NodeState::Starting).unwrap();
        lifecycle.transition(NodeState::Ready).unwrap();
        lifecycle.begin_draining().unwrap();
        lifecycle.transition(NodeState::Stopped).unwrap();
        assert_eq!(lifecycle.state(), NodeState::Stopped);
    }

    #[test]
    fn lifecycle_rejects_skipping_readiness() {
        let lifecycle = NodeLifecycle::new();
        assert!(lifecycle.transition(NodeState::Ready).is_err());
        assert_eq!(lifecycle.state(), NodeState::Booting);
    }
}
