//! Python bindings for Pulsing Actor System
//!
//! This crate provides Python bindings for the Pulsing distributed actor framework.
//! It is a standalone module that can be used independently of Dynamo.

use pyo3::prelude::*;

mod actor;
mod errors;
mod policies;
mod python_error_converter;
mod python_executor;

pub use python_executor::{init_python_executor, python_executor, ExecutorError};

/// Pulsing Actor System Python module
///
/// This module provides:
/// - ActorSystem: Distributed actor system management
/// - Actor types: NodeId, ActorId, ActorRef
/// - Message types: Message, StreamMessage
/// - Streaming: StreamReader, StreamWriter
/// - Load balancing policies: Random, RoundRobin, PowerOfTwo, ConsistentHash, CacheAware
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init()
        .ok();

    // Add error classes
    errors::add_to_module(m)?;

    // Add actor system classes
    actor::add_to_module(m)?;

    // Add load balancing policies
    policies::add_to_module(m)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
