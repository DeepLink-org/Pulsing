//! Round-robin load balancing policy
//!
//! Selects workers in sequential order, cycling through all healthy workers.
//! Provides deterministic load distribution.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders, Worker};
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
/// Provides deterministic and even distribution when request loads are similar.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    /// Create a new round-robin policy
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::BasicWorker;

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
            Arc::new(BasicWorker::new("http://w3:8000".to_string())),
        ];

        // Should select workers in order: 0, 1, 2, 0, 1, 2, ...
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
    }

    #[test]
    fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
            Arc::new(BasicWorker::new("http://w3:8000".to_string())),
        ];

        // Mark middle worker as unhealthy
        if let Some(w) = workers[1].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }

        // Should skip unhealthy worker: 0, 2, 0, 2, ...
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
    }

    #[test]
    fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
        ];

        // Advance the counter
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}
