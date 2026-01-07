//! Actor-based benchmark implementation
//!
//! This module provides an Actor-based architecture for benchmark execution:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         CoordinatorActor                                  │
//! │  - Manages benchmark lifecycle                                            │
//! │  - Spawns ExecutorActors and CollectorActor                              │
//! │  - Handles progress events                                                │
//! └────────────────────────────────────────────────────────────────────────── │
//!              │                                      │
//!              ▼                                      ▼
//! ┌───────────────────────┐                ┌───────────────────────┐
//! │   ExecutorActor[0..N] │                │    CollectorActor     │
//! │  - Sends requests     │ ────────────>  │  - Aggregates results │
//! │  - Handles responses  │   Results      │  - Computes stats     │
//! └───────────────────────┘                └───────────────────────┘
//! ```

mod messages;
mod executor;
mod collector;
mod coordinator;

pub use messages::*;
pub use executor::ExecutorActor;
pub use collector::CollectorActor;
pub use coordinator::CoordinatorActor;

use crate::benchmark::BenchmarkConfig;
use crate::requests::{TextGenerationBackend, TextRequestGenerator};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

/// Run benchmark using Actor-based coordinator
/// 
/// This function creates a CoordinatorActor and runs the benchmark directly
/// without requiring full actor system setup. This is suitable for standalone
/// benchmarks.
pub async fn run_actor_benchmark(
    config: BenchmarkConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
) -> anyhow::Result<crate::results::BenchmarkReport> {
    info!("Starting actor-based benchmark");
    
    // Create coordinator directly (simplified actor model)
    let mut coordinator = CoordinatorActor::new(config, backend, requests);
    
    // Create a minimal context for direct execution
    let _ctx = MinimalContext::new();
    
    // Run benchmark
    let report = coordinator.run_benchmark_direct().await?;
    
    info!("Actor-based benchmark completed");
    Ok(report)
}

/// Minimal context for direct actor execution (without full actor system)
struct MinimalContext;

impl MinimalContext {
    fn new() -> Self {
        Self
    }
}

