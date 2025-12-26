//! Cache-aware load balancing policy
//!
//! This router combines two strategies to optimize both cache utilization and request distribution:
//!
//! 1. Cache-Aware Routing (Approximate Tree)
//!    - Maintains an approximate radix tree for each worker based on request history
//!    - Routes requests to workers with highest prefix match (cache hit potential)
//!
//! 2. Load Balancing (Shortest Queue with Balance Thresholds)
//!    - Tracks pending request counts per worker
//!    - Routes to least busy worker when load is imbalanced
//!
//! The router dynamically switches between these strategies based on load conditions:
//! - Uses load balancing when the system is imbalanced
//! - Uses cache-aware routing when the system is balanced
//!
//! A system is considered imbalanced if both conditions are met:
//! 1. (max - min) > abs_threshold
//! 2. max > rel_threshold * min

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders, Tree, Worker};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tracing::debug;

/// Configuration for cache-aware policy
#[derive(Debug, Clone)]
pub struct CacheAwareConfig {
    /// Minimum prefix match ratio to use highest-match routing (0.0 to 1.0)
    pub cache_threshold: f32,
    /// Absolute difference threshold for load imbalance detection
    pub balance_abs_threshold: usize,
    /// Relative ratio threshold for load imbalance detection
    pub balance_rel_threshold: f32,
    /// Interval between LRU eviction cycles (seconds, 0 to disable)
    pub eviction_interval_secs: u64,
    /// Maximum nodes per tree before eviction
    pub max_tree_size: usize,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.0001,
            eviction_interval_secs: 60,
            max_tree_size: 100000,
        }
    }
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<Mutex<HashMap<String, Tree>>>,
    eviction_handle: Option<thread::JoinHandle<()>>,
}

impl CacheAwarePolicy {
    /// Create a new cache-aware policy with default config
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    /// Create a new cache-aware policy with custom config
    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(Mutex::new(HashMap::<String, Tree>::new()));

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                if let Ok(mut trees_guard) = trees_clone.lock() {
                    for (model_id, tree) in trees_guard.iter_mut() {
                        tree.evict_tenant_by_size(max_tree_size);
                        debug!(
                            "Cache eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            eviction_handle,
        }
    }

    /// Initialize the tree with worker URLs
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        if let Ok(mut trees) = self.trees.lock() {
            // Group workers by model
            let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
            for worker in workers {
                let model_id = worker.model_id();
                let tree_key = if model_id.is_empty() || model_id == "unknown" {
                    "default".to_string()
                } else {
                    model_id.to_string()
                };
                model_workers.entry(tree_key).or_default().push(worker);
            }

            // Initialize tree for each model
            for (tree_key, model_workers) in model_workers {
                let tree = trees.entry(tree_key).or_insert_with(Tree::new);
                for worker in model_workers {
                    tree.insert("", worker.url());
                }
            }
        }
    }

    /// Add a single worker to the tree
    pub fn add_worker(&self, worker: &dyn Worker) {
        if let Ok(mut trees) = self.trees.lock() {
            let model_id = worker.model_id();
            let tree_key = if model_id.is_empty() || model_id == "unknown" {
                "default".to_string()
            } else {
                model_id.to_string()
            };
            let tree = trees.entry(tree_key).or_insert_with(Tree::new);
            tree.insert("", worker.url());
        }
    }

    /// Add a worker by URL and model
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        if let Ok(mut trees) = self.trees.lock() {
            let tree = trees.entry(model_id.to_string()).or_insert_with(Tree::new);
            tree.insert("", url);
        }
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        if let Ok(mut trees) = self.trees.lock() {
            let model_id = worker.model_id();
            let tree_key = if model_id.is_empty() || model_id == "unknown" {
                "default".to_string()
            } else {
                model_id.to_string()
            };
            if let Some(tree) = trees.get_mut(&tree_key) {
                tree.remove_tenant(worker.url());
            }
        }
    }

    /// Remove a worker by URL
    pub fn remove_worker_by_url(&self, url: &str) {
        if let Ok(mut trees) = self.trees.lock() {
            for (_model_id, tree) in trees.iter_mut() {
                tree.remove_tenant(url);
            }
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        if let Ok(mut trees) = self.trees.lock() {
            for (model_id, tree) in trees.iter_mut() {
                tree.evict_tenant_by_size(max_size);
                debug!(
                    "Cache eviction for model {}, max_size: {}",
                    model_id, max_size
                );
            }
        }
    }
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Group workers by model
        let mut model_workers: HashMap<String, Vec<usize>> = HashMap::new();
        for idx in &healthy_indices {
            let model_id = workers[*idx].model_id();
            let tree_key = if model_id.is_empty() || model_id == "unknown" {
                "default".to_string()
            } else {
                model_id.to_string()
            };
            model_workers.entry(tree_key).or_default().push(*idx);
        }

        // Get current load statistics
        let loads: Vec<usize> = workers.iter().map(|w| w.load()).collect();
        let max_load = *loads.iter().max().unwrap_or(&0);
        let min_load = *loads.iter().min().unwrap_or(&0);

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            debug!(
                "Load balancing triggered | max: {} | min: {}",
                max_load, min_load
            );

            // Use shortest queue when imbalanced
            let min_load_idx = healthy_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()?;

            // Still update tree to maintain cache state
            if let Some(text) = request_text {
                if let Ok(mut trees) = self.trees.lock() {
                    let model_id = workers[min_load_idx].model_id();
                    let tree_key = if model_id.is_empty() || model_id == "unknown" {
                        "default".to_string()
                    } else {
                        model_id.to_string()
                    };
                    let tree = trees.entry(tree_key).or_insert_with(Tree::new);
                    tree.insert(text, workers[min_load_idx].url());
                }
            }

            return Some(min_load_idx);
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        if let Ok(mut trees) = self.trees.lock() {
            let mut best_match_idx: Option<usize> = None;
            let mut best_match_rate: f32 = 0.0;

            // Find best match across all models
            for (model_id, worker_indices) in &model_workers {
                let tree = trees.entry(model_id.clone()).or_insert_with(Tree::new);

                let (matched_text, matched_worker) = tree.prefix_match(text);
                let match_rate = if text.is_empty() {
                    0.0
                } else {
                    matched_text.chars().count() as f32 / text.chars().count() as f32
                };

                if match_rate > best_match_rate {
                    if let Some(idx) = worker_indices
                        .iter()
                        .find(|&&idx| workers[idx].url() == matched_worker)
                    {
                        best_match_idx = Some(*idx);
                        best_match_rate = match_rate;
                    }
                }
            }

            // Select worker based on cache threshold
            let selected_idx = if let (Some(idx), true) = (
                best_match_idx,
                best_match_rate > self.config.cache_threshold,
            ) {
                debug!("Cache hit with match rate {:.2}", best_match_rate);
                idx
            } else {
                debug!("Cache miss with match rate {:.2}", best_match_rate);

                // Find model with smallest tree (most cache capacity)
                let mut smallest_tree_model = String::new();
                let mut smallest_tree_size = usize::MAX;

                for model_id in model_workers.keys() {
                    let tree = trees.entry(model_id.clone()).or_insert_with(Tree::new);
                    let size = tree.get_used_size_per_tenant().values().sum::<usize>();
                    if size < smallest_tree_size {
                        smallest_tree_size = size;
                        smallest_tree_model = model_id.clone();
                    }
                }

                // Select least loaded worker from model with most cache capacity
                if let Some(worker_indices) = model_workers.get(&smallest_tree_model) {
                    worker_indices
                        .iter()
                        .min_by_key(|&&idx| workers[idx].load())
                        .copied()
                        .unwrap_or(healthy_indices[0])
                } else {
                    healthy_indices[0]
                }
            };

            // Update the tree with this request
            let model_id = workers[selected_idx].model_id();
            let tree_key = if model_id.is_empty() || model_id == "unknown" {
                "default".to_string()
            } else {
                model_id.to_string()
            };
            let tree = trees.entry(tree_key).or_insert_with(Tree::new);
            tree.insert(text, workers[selected_idx].url());

            return Some(selected_idx);
        }

        // Fallback to first healthy worker
        healthy_indices.first().copied()
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        if !success {
            debug!(
                "Request to {} completed with success={}",
                worker_url, success
            );
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // In PD mode:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Note: The eviction thread will continue until the program exits
        // In production, we'd use a channel or atomic flag to signal shutdown
        if let Some(handle) = self.eviction_handle.take() {
            drop(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::BasicWorker;

    #[test]
    fn test_cache_aware_with_balanced_load() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread for testing
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
        ];

        policy.init_workers(&workers);

        // First request
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            max_tree_size: 10000,
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string());
        let worker2 = BasicWorker::new("http://w2:8000".to_string());

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://w1:8000".to_string())),
            Arc::new(BasicWorker::new("http://w2:8000".to_string())),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");

        if let Some(w) = workers[0].as_any().downcast_ref::<BasicWorker>() {
            w.set_health(false);
        }

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }
}
