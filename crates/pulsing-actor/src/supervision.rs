//! Supervision strategies for actor fault tolerance
//!
//! This module defines how actors should handle failures (panics or errors)
//! and when they should be restarted.

use rand::Rng;
use std::time::Duration;

/// Restart policy for an actor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RestartPolicy {
    /// Always restart the actor, regardless of the exit reason
    Always,
    /// Restart the actor only if it failed (non-normal exit)
    OnFailure,
    /// Never restart the actor (default). Panic / 不可恢复错误时停止且不恢复
    #[default]
    Never,
}

impl RestartPolicy {
    pub fn should_restart(&self, is_failure: bool) -> bool {
        match self {
            RestartPolicy::Always => true,
            RestartPolicy::OnFailure => is_failure,
            RestartPolicy::Never => false,
        }
    }
}

/// Backoff strategy for restarts
#[derive(Debug, Clone, Copy)]
pub struct BackoffStrategy {
    /// Minimum backoff duration
    pub min: Duration,
    /// Maximum backoff duration
    pub max: Duration,
    /// Jitter factor (0.0 to 1.0)
    pub jitter: f64,
    /// Exponential factor (e.g., 2.0 for doubling)
    pub factor: f64,
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self {
            min: Duration::from_millis(100),
            max: Duration::from_secs(30),
            jitter: 0.2,
            factor: 2.0,
        }
    }
}

impl BackoffStrategy {
    /// Create a new exponential backoff strategy
    pub fn exponential(min: Duration, max: Duration) -> Self {
        Self {
            min,
            max,
            ..Default::default()
        }
    }

    /// Calculate backoff duration for a given attempt number (0-based)
    pub fn duration(&self, attempt: u32) -> Duration {
        let mut duration = self.min.as_secs_f64() * self.factor.powi(attempt as i32);

        // Cap at max
        let max_secs = self.max.as_secs_f64();
        if duration > max_secs {
            duration = max_secs;
        }

        // Add jitter
        if self.jitter > 0.0 {
            let jitter_amount = duration * self.jitter;
            let random_factor = rand::rng().random_range(-1.0..1.0);
            duration += jitter_amount * random_factor;
        }

        // Ensure we don't go below min (though calculation above starts at min)
        if duration < 0.0 {
            duration = 0.0;
        }

        Duration::from_secs_f64(duration)
    }
}

/// Supervision specification for an actor
#[derive(Debug, Clone, Default)]
pub struct SupervisionSpec {
    /// Restart policy
    pub policy: RestartPolicy,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
    /// Maximum number of restarts allowed (used when policy is Always/OnFailure)
    pub max_restarts: u32,
    /// Time window for max_restarts (optional).
    /// If set, max_restarts applies only within this sliding window.
    /// If None, max_restarts applies to the lifetime of the actor.
    pub restart_window: Option<Duration>,
}

impl SupervisionSpec {
    /// Create a spec that never restarts (default)
    pub fn never() -> Self {
        Self::default()
    }

    /// Create a spec that always restarts
    pub fn always() -> Self {
        Self {
            policy: RestartPolicy::Always,
            ..Default::default()
        }
    }

    /// Create a spec that restarts on failure
    pub fn on_failure() -> Self {
        Self {
            policy: RestartPolicy::OnFailure,
            ..Default::default()
        }
    }

    pub fn with_backoff(mut self, backoff: BackoffStrategy) -> Self {
        self.backoff = backoff;
        self
    }

    pub fn with_max_restarts(mut self, max: u32) -> Self {
        self.max_restarts = max;
        self
    }
}
