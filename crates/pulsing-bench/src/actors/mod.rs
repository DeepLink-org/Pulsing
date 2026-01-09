//! Actor-based benchmark implementation
//!
//! This module provides a refactored benchmark system using the Actor model.
//!
//! ## Architecture
//!
//! ```text
//!                              ┌─────────────────┐
//!                              │   Coordinator   │
//!                              │  (Orchestrator) │
//!                              └────────┬────────┘
//!                                       │
//!         ┌─────────────────────────────┼─────────────────────────────┐
//!         │                             │                             │
//!         ▼                             ▼                             ▼
//! ┌────────────────┐           ┌─────────────────┐           ┌────────────────┐
//! │   Scheduler    │           │   Worker(s)     │           │    Metrics     │
//! │ (Rate Control) │           │ (HTTP Clients)  │           │  Aggregator    │
//! └────────────────┘           └───────┬─────────┘           └────────┬───────┘
//!                                      │                              │
//!                                      │ RequestCompleted             │ DisplayUpdate
//!                                      │                              │
//!                                      ▼                              ▼
//!                              ┌─────────────────┐           ┌────────────────┐
//!                              │    Metrics      │ ───────►  │    Console     │
//!                              │   Aggregator    │           │   Renderer     │
//!                              └─────────────────┘           └────────────────┘
//! ```
//!
//! ## Actors
//!
//! - **Coordinator**: Manages lifecycle, spawns other actors, orchestrates benchmark phases
//! - **Scheduler**: Controls request timing (ConstantVUs, ConstantArrivalRate)
//! - **Worker**: Sends HTTP requests, processes streaming responses, calculates TTFT/TPOT
//! - **MetricsAggregator**: Collects results, computes statistics (percentiles, throughput)
//! - **ConsoleRenderer**: Renders benchmark progress to terminal (tables, progress bars)
//!
//! ## Benefits of Actor Model
//!
//! 1. **Separation of Concerns**: Each actor has a single responsibility
//! 2. **Testability**: Metrics logic can be tested without terminal dependencies
//! 3. **Flexibility**: Easy to replace ConsoleRenderer with JSON/Web output
//! 4. **Scalability**: Workers can be distributed across nodes
//! 5. **Fault Isolation**: Actor failures don't crash the whole system

pub mod console_renderer;
pub mod coordinator;
pub mod messages;
pub mod metrics_aggregator;
pub mod scheduler;
pub mod worker;

// Re-export main types
pub use console_renderer::ConsoleRendererActor;
pub use coordinator::CoordinatorActor;
pub use messages::*;
pub use metrics_aggregator::MetricsAggregatorActor;
pub use scheduler::SchedulerActor;
pub use worker::WorkerActor;
