//! Test helper macros and utilities
//!
//! This module provides reusable test infrastructure to eliminate
//! repetitive test patterns across the codebase.
//!
//! # Example
//!
//! ```rust,ignore
//! use pulsing_actor::test_helper::*;
//! use pulsing_actor::actor_test;
//!
//! actor_test!(test_basic_echo, system, {
//!     let echo = spawn_echo_actor(&system, "test/echo").await;
//!     let response: TestPong = echo.ask(TestPing { value: 21 }).await.unwrap();
//!     assert_eq!(response.result, 42);
//! });
//! ```

use crate::actor::{Actor, ActorContext, ActorRef, Message};
use crate::system::{ActorSystem, SystemConfig};
use crate::ActorSystemCoreExt;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================================
// Common Test Messages
// ============================================================================

/// Simple ping message for testing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TestPing {
    pub value: i32,
}

/// Simple pong response for testing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TestPong {
    pub result: i32,
}

/// Accumulate message for stateful actor testing
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestAccumulate {
    pub amount: i32,
}

/// Get total message for stateful actor testing
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestGetTotal;

/// Total response for stateful actor testing
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TestTotalResponse {
    pub total: i32,
}

// ============================================================================
// Common Test Actors
// ============================================================================

/// Echo actor that doubles the input value
///
/// Useful for basic request-response testing
pub struct TestEchoActor {
    /// Counter to track how many messages were processed
    pub echo_count: Arc<AtomicUsize>,
}

impl TestEchoActor {
    /// Create a new echo actor with an internal counter
    pub fn new() -> Self {
        Self {
            echo_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Create a new echo actor with a shared counter
    pub fn with_counter(counter: Arc<AtomicUsize>) -> Self {
        Self {
            echo_count: counter,
        }
    }

    /// Get the number of messages processed
    pub fn count(&self) -> usize {
        self.echo_count.load(Ordering::SeqCst)
    }
}

impl Default for TestEchoActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for TestEchoActor {
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> crate::error::Result<Message> {
        if msg.msg_type().ends_with("TestPing") {
            let ping: TestPing = msg.unpack()?;
            self.echo_count.fetch_add(1, Ordering::SeqCst);
            return Message::pack(&TestPong {
                result: ping.value * 2,
            });
        }
        Err(crate::error::PulsingError::from(
            crate::error::RuntimeError::Other(format!("Unknown message type: {}", msg.msg_type())),
        ))
    }
}

/// Accumulator actor that maintains a running total
///
/// Useful for testing stateful actors
pub struct TestAccumulatorActor {
    pub total: i32,
}

impl TestAccumulatorActor {
    /// Create a new accumulator with initial value
    pub fn new(initial: i32) -> Self {
        Self { total: initial }
    }
}

impl Default for TestAccumulatorActor {
    fn default() -> Self {
        Self::new(0)
    }
}

#[async_trait]
impl Actor for TestAccumulatorActor {
    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> crate::error::Result<Message> {
        let msg_type = msg.msg_type();
        if msg_type.ends_with("TestAccumulate") {
            let acc: TestAccumulate = msg.unpack()?;
            self.total += acc.amount;
            return Message::pack(&TestTotalResponse { total: self.total });
        }
        if msg_type.ends_with("TestGetTotal") {
            return Message::pack(&TestTotalResponse { total: self.total });
        }
        Err(crate::error::PulsingError::from(
            crate::error::RuntimeError::Other(format!("Unknown message type: {}", msg_type)),
        ))
    }
}

// ============================================================================
// Test Setup Helpers
// ============================================================================

/// Create a standalone actor system for testing
///
/// This creates a system configured for single-node testing without cluster features.
pub async fn create_test_system() -> Arc<ActorSystem> {
    ActorSystem::new(SystemConfig::standalone())
        .await
        .expect("Failed to create test actor system")
}

/// Spawn an echo actor with the given name
pub async fn spawn_echo_actor(system: &Arc<ActorSystem>, name: &str) -> ActorRef {
    system
        .spawn_named(name, TestEchoActor::new())
        .await
        .expect("Failed to spawn echo actor")
}

/// Spawn an echo actor with a shared counter
pub async fn spawn_echo_actor_with_counter(
    system: &Arc<ActorSystem>,
    name: &str,
    counter: Arc<AtomicUsize>,
) -> ActorRef {
    system
        .spawn_named(name, TestEchoActor::with_counter(counter))
        .await
        .expect("Failed to spawn echo actor")
}

/// Spawn an accumulator actor with the given name
pub async fn spawn_accumulator_actor(system: &Arc<ActorSystem>, name: &str) -> ActorRef {
    system
        .spawn_named(name, TestAccumulatorActor::default())
        .await
        .expect("Failed to spawn accumulator actor")
}

/// Spawn an accumulator actor with initial value
pub async fn spawn_accumulator_actor_with_initial(
    system: &Arc<ActorSystem>,
    name: &str,
    initial: i32,
) -> ActorRef {
    system
        .spawn_named(name, TestAccumulatorActor::new(initial))
        .await
        .expect("Failed to spawn accumulator actor")
}

// ============================================================================
// Test Macros
// ============================================================================

/// Macro for creating actor system tests with automatic setup and teardown
///
/// This macro handles:
/// - Creating a standalone ActorSystem
/// - Running the test body
/// - Shutting down the system
///
/// # Example
///
/// ```rust,ignore
/// actor_test!(test_echo_actor, system, {
///     let echo = spawn_echo_actor(&system, "test/echo").await;
///     let response: TestPong = echo.ask(TestPing { value: 21 }).await.unwrap();
///     assert_eq!(response.result, 42);
/// });
/// ```
#[macro_export]
macro_rules! actor_test {
    ($test_name:ident, $system:ident, $test_body:block) => {
        #[tokio::test]
        async fn $test_name() {
            let $system = $crate::test_helper::create_test_system().await;
            // Execute the test body
            $test_body
            // Shutdown the system
            $system.shutdown().await.expect("Failed to shutdown system");
        }
    };
}

/// Macro for creating actor system tests that return a Result
///
/// Similar to `actor_test!` but allows the test body to return a Result for
/// more ergonomic error handling with `?`.
///
/// # Example
///
/// ```rust,ignore
/// actor_test_result!(test_echo_actor, system, {
///     let echo = spawn_echo_actor(&system, "test/echo").await;
///     let response: TestPong = echo.ask(TestPing { value: 21 }).await?;
///     assert_eq!(response.result, 42);
///     Ok(())
/// });
/// ```
#[macro_export]
macro_rules! actor_test_result {
    ($test_name:ident, $system:ident, $test_body:block) => {
        #[tokio::test]
        async fn $test_name() -> $crate::error::Result<()> {
            let $system = $crate::test_helper::create_test_system().await;
            // Execute the test body
            let test_result: $crate::error::Result<()> = $test_body;
            // Shutdown the system regardless of test result
            $system.shutdown().await?;
            test_result
        }
    };
}

/// Macro for creating tests with multiple actors
///
/// This is a convenience macro for spawning multiple named actors at once.
///
/// # Example
///
/// ```rust,ignore
/// spawn_test_actors!(system, {
///     "test/echo1" => TestEchoActor::new(),
///     "test/echo2" => TestEchoActor::new(),
///     "test/acc" => TestAccumulatorActor::default(),
/// });
/// ```
#[macro_export]
macro_rules! spawn_test_actors {
    ($system:expr, { $($name:expr => $actor:expr),* $(,)? }) => {
        {
            let mut refs = Vec::new();
            $(
                refs.push(
                    $system
                        .spawn_named($name, $actor)
                        .await
                        .expect(concat!("Failed to spawn actor: ", $name))
                );
            )*
            refs
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_test_system() {
        let system = create_test_system().await;
        assert!(!system.local_actor_names().is_empty()); // At least SystemActor
        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_echo_actor() {
        let system = create_test_system().await;
        let echo = spawn_echo_actor(&system, "test/echo").await;

        let response: TestPong = echo.ask(TestPing { value: 21 }).await.unwrap();
        assert_eq!(response.result, 42);

        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_echo_actor_with_counter() {
        let counter = Arc::new(AtomicUsize::new(0));
        let system = create_test_system().await;
        let echo = spawn_echo_actor_with_counter(&system, "test/echo", counter.clone()).await;

        let _: TestPong = echo.ask(TestPing { value: 1 }).await.unwrap();
        let _: TestPong = echo.ask(TestPing { value: 2 }).await.unwrap();
        let _: TestPong = echo.ask(TestPing { value: 3 }).await.unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 3);
        system.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_accumulator_actor() {
        let system = create_test_system().await;
        let acc = spawn_accumulator_actor(&system, "test/acc").await;

        let r1: TestTotalResponse = acc.ask(TestAccumulate { amount: 10 }).await.unwrap();
        assert_eq!(r1.total, 10);

        let r2: TestTotalResponse = acc.ask(TestAccumulate { amount: 5 }).await.unwrap();
        assert_eq!(r2.total, 15);

        let r3: TestTotalResponse = acc.ask(TestGetTotal).await.unwrap();
        assert_eq!(r3.total, 15);

        system.shutdown().await.unwrap();
    }

    // Test the actor_test! macro
    actor_test!(test_macro_basic, system, {
        let echo = spawn_echo_actor(&system, "test/echo").await;
        let response: TestPong = echo.ask(TestPing { value: 10 }).await.unwrap();
        assert_eq!(response.result, 20);
    });

    // Test the actor_test_result! macro
    actor_test_result!(test_macro_result, system, {
        let echo = spawn_echo_actor(&system, "test/echo").await;
        let response: TestPong = echo.ask(TestPing { value: 5 }).await?;
        assert_eq!(response.result, 10);
        Ok(())
    });

    // Test macros with multiple actors
    actor_test!(test_macro_multiple_actors, system, {
        let echo1 = spawn_echo_actor(&system, "test/echo1").await;
        let echo2 = spawn_echo_actor(&system, "test/echo2").await;

        let r1: TestPong = echo1.ask(TestPing { value: 1 }).await.unwrap();
        let r2: TestPong = echo2.ask(TestPing { value: 2 }).await.unwrap();

        assert_eq!(r1.result, 2);
        assert_eq!(r2.result, 4);
    });

    // Test the spawn_test_actors! macro
    #[tokio::test]
    async fn test_spawn_test_actors_macro() {
        let system = create_test_system().await;

        let refs = spawn_test_actors!(system, {
            "test/echo1" => TestEchoActor::new(),
            "test/echo2" => TestEchoActor::new(),
        });

        assert_eq!(refs.len(), 2);

        let r1: TestPong = refs[0].ask(TestPing { value: 1 }).await.unwrap();
        let r2: TestPong = refs[1].ask(TestPing { value: 2 }).await.unwrap();

        assert_eq!(r1.result, 2);
        assert_eq!(r2.result, 4);

        system.shutdown().await.unwrap();
    }
}
