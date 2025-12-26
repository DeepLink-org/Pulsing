//! Random load balancing policy
//!
//! Selects workers randomly with uniform distribution among healthy workers.
//! This is the simplest policy and provides good distribution with no coordination.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders, Worker};
use rand::Rng;
use std::any::Any;
use std::sync::Arc;

/// Random selection policy
///
/// Selects workers randomly with uniform distribution among healthy workers.
/// Simple and effective for stateless workloads.
#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    /// Create a new random policy
    pub fn new() -> Self {
        Self
    }
}

impl LoadBalancingPolicy for RandomPolicy {
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

        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..healthy_indices.len());

        Some(healthy_indices[random_idx])
    }

    fn name(&self) -> &'static str {
        "random"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::BasicWorker;
    use std::collections::HashMap;

    #[test]
    fn test_random_selection() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
            Arc::new(BasicWorker::new("http://w3:8000".to_string())),
        ];

        // Test multiple selections to ensure randomness
        let mut counts = HashMap::new();
        for _ in 0..300 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        // All workers should be selected at least once
        assert_eq!(counts.len(), 3);
        assert!(counts.values().all(|&count| count > 0));
    }

    #[test]
    fn test_random_with_unhealthy_workers() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
        ];

        // Mark first worker as unhealthy
        if let Some(w) = workers[0].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }

        // Should always select the healthy worker (index 1)
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, None), Some(1));
        }
    }

    #[test]
    fn test_random_no_healthy_workers() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> =
            vec![Arc::new(BasicWorker::new("http://w1:8000".to_string()))];

        if let Some(w) = workers[0].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }
        assert_eq!(policy.select_worker(&workers, None), None);
    }
}
