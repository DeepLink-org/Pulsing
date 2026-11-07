//! Load balancing policies for actor routing
//!
//! This module provides various load balancing strategies for distributing
//! requests across multiple worker nodes. The policies are designed to work
//! seamlessly with the actor system.
//!
//! # Available Policies
//!
//! - [`RandomPolicy`]: Uniform random selection among healthy workers
//! - [`RoundRobinPolicy`]: Sequential selection cycling through healthy workers
//! - [`PowerOfTwoPolicy`]: Power-of-two choices for better load distribution
//! - [`ConsistentHashPolicy`]: Session/user affinity using consistent hashing
//!
//! # Example
//!
//! ```rust,ignore
//! use pulsing_actor::policies::{LoadBalancingPolicy, RandomPolicy};
//!
//! let policy = RandomPolicy::new();
//! let workers = vec![/* worker references */];
//! if let Some(idx) = policy.select_worker(&workers, None) {
//!     // Route request to workers[idx]
//! }
//! ```

mod consistent_hash;
mod power_of_two;
mod random;
mod round_robin;

pub use consistent_hash::ConsistentHashPolicy;
pub use power_of_two::PowerOfTwoPolicy;
pub use random::RandomPolicy;
pub use round_robin::RoundRobinPolicy;

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Request headers type alias for policy routing decisions
pub type RequestHeaders = HashMap<String, String>;

/// Worker trait that policies can use to inspect worker state
///
/// This trait defines the interface that workers must implement
/// to be compatible with the load balancing policies.
pub trait Worker: Send + Sync + std::fmt::Debug {
    /// Get the worker's URL/identifier
    fn url(&self) -> &str;

    /// Check if the worker is healthy
    fn is_healthy(&self) -> bool;

    /// Set the worker's health status
    fn set_healthy(&mut self, healthy: bool);

    /// Get the current load (number of in-flight requests)
    fn load(&self) -> usize;

    /// Increment the load counter
    fn increment_load(&self);

    /// Decrement the load counter
    fn decrement_load(&self);

    /// Increment the processed counter
    fn increment_processed(&self);

    /// Get the total number of processed requests
    fn processed(&self) -> u64;

    /// Check if the circuit breaker allows execution
    fn circuit_breaker_allows(&self) -> bool {
        true
    }

    /// Get self as Any for downcasting (useful in tests)
    fn as_any(&self) -> &dyn Any;
}

/// Load balancing policy trait
///
/// Defines the interface for all load balancing strategies. Implementations
/// must be thread-safe as policies may be shared across multiple threads.
pub trait LoadBalancingPolicy: Send + Sync + std::fmt::Debug {
    /// Select a worker from the available workers
    ///
    /// # Arguments
    /// * `workers` - List of available workers
    /// * `request_text` - Optional request body for content-based routing
    ///
    /// # Returns
    /// Index of the selected worker, or None if no healthy worker is available
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
    ) -> Option<usize> {
        self.select_worker_with_headers(workers, request_text, None)
    }

    /// Select a worker with request headers for more sophisticated routing
    ///
    /// # Arguments
    /// * `workers` - List of available workers
    /// * `request_text` - Optional request body
    /// * `headers` - Optional HTTP headers for header-based routing
    ///
    /// # Returns
    /// Index of the selected worker, or None if no healthy worker is available
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize>;

    /// Get the policy name for logging and metrics
    fn name(&self) -> &'static str;

    /// Check if this policy requires request text for routing decisions
    fn needs_request_text(&self) -> bool {
        false
    }

    /// Check if this policy requires HTTP headers for routing decisions
    fn needs_headers(&self) -> bool {
        false
    }

    /// Reset policy state (e.g., round-robin counter)
    fn reset(&self) {}

    /// Update cached load information from external monitoring
    fn update_loads(&self, _loads: &HashMap<String, isize>) {}

    /// Callback when a request completes
    fn on_request_complete(&self, _worker_url: &str, _success: bool) {}

    /// Get self as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Get indices of healthy workers that pass circuit breaker check
pub fn get_healthy_worker_indices(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
    workers
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_healthy() && w.circuit_breaker_allows())
        .map(|(i, _)| i)
        .collect()
}

/// Basic worker implementation for testing and simple use cases
#[derive(Debug)]
pub struct BasicWorker {
    url: String,
    healthy: std::sync::atomic::AtomicBool,
    load: std::sync::atomic::AtomicUsize,
    processed: std::sync::atomic::AtomicU64,
}

impl BasicWorker {
    /// Create a new basic worker
    pub fn new(url: String) -> Self {
        Self {
            url,
            healthy: std::sync::atomic::AtomicBool::new(true),
            load: std::sync::atomic::AtomicUsize::new(0),
            processed: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl Worker for BasicWorker {
    fn url(&self) -> &str {
        &self.url
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn set_healthy(&mut self, healthy: bool) {
        self.healthy
            .store(healthy, std::sync::atomic::Ordering::Relaxed);
    }

    fn load(&self) -> usize {
        self.load.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.load.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn decrement_load(&self) {
        self.load.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn increment_processed(&self) {
        self.processed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn processed(&self) -> u64 {
        self.processed.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Allow setting healthy through Arc<BasicWorker>
impl BasicWorker {
    /// Set health status (for testing)
    pub fn set_health(&self, healthy: bool) {
        self.healthy
            .store(healthy, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_worker() {
        let worker = BasicWorker::new("http://localhost:8000".to_string());
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed(), 0);

        worker.increment_load();
        assert_eq!(worker.load(), 1);

        worker.increment_processed();
        assert_eq!(worker.processed(), 1);

        worker.set_health(false);
        assert!(!worker.is_healthy());
    }

    #[test]
    fn test_get_healthy_worker_indices() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
            Arc::new(BasicWorker::new("http://w3:8000".to_string())),
        ];

        // All healthy
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 1, 2]);

        // Mark one as unhealthy
        if let Some(w) = workers[1].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }
        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![0, 2]);
    }

    #[test]
    fn test_basic_worker_url() {
        let worker = BasicWorker::new("http://localhost:9000".to_string());
        assert_eq!(worker.url(), "http://localhost:9000");
    }

    #[test]
    fn test_basic_worker_load_increment_decrement() {
        let worker = BasicWorker::new("http://localhost:8000".to_string());
        assert_eq!(worker.load(), 0);

        worker.increment_load();
        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        worker.decrement_load();
        worker.decrement_load();
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_basic_worker_processed_counter() {
        let worker = BasicWorker::new("http://localhost:8000".to_string());
        assert_eq!(worker.processed(), 0);

        worker.increment_processed();
        worker.increment_processed();
        worker.increment_processed();
        worker.increment_processed();
        worker.increment_processed();
        assert_eq!(worker.processed(), 5);
    }

    #[test]
    fn test_basic_worker_health_toggle() {
        let worker = BasicWorker::new("http://localhost:8000".to_string());
        assert!(worker.is_healthy());

        worker.set_health(false);
        assert!(!worker.is_healthy());

        worker.set_health(true);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_circuit_breaker_default() {
        let worker = BasicWorker::new("http://localhost:8000".to_string());
        assert!(worker.circuit_breaker_allows());
    }

    #[test]
    fn test_basic_worker_as_any() {
        let worker = BasicWorker::new("http://localhost:8000".to_string());
        let any_ref = worker.as_any();
        assert!(any_ref.downcast_ref::<BasicWorker>().is_some());
    }

    #[test]
    fn test_get_healthy_worker_indices_empty() {
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let indices = get_healthy_worker_indices(&workers);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_get_healthy_worker_indices_all_unhealthy() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
        ];

        for worker in &workers {
            if let Some(w) = worker.as_any().downcast_ref::<BasicWorker>() {
                w.set_health(false);
            }
        }

        let indices = get_healthy_worker_indices(&workers);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_get_healthy_worker_indices_single_healthy() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
            Arc::new(BasicWorker::new("http://w3:8000".to_string())),
        ];

        // Mark first two as unhealthy
        if let Some(w) = workers[0].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }
        if let Some(w) = workers[1].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }

        let indices = get_healthy_worker_indices(&workers);
        assert_eq!(indices, vec![2]);
    }

    #[test]
    fn test_request_headers_type() {
        let mut headers: RequestHeaders = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        headers.insert("x-request-id".to_string(), "12345".to_string());

        assert_eq!(
            headers.get("content-type"),
            Some(&"application/json".to_string())
        );
        assert_eq!(headers.get("x-request-id"), Some(&"12345".to_string()));
        assert_eq!(headers.len(), 2);
    }
}
