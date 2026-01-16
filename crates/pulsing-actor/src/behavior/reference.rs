//! Typed actor references

use crate::actor::ActorRef;
use crate::system::ActorSystem;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

/// A type-safe actor reference
///
/// Unlike `ActorRef`, `TypedRef<M>` knows the message type at compile time,
/// providing type-safe message sending.
///
/// # Example
///
/// ```rust,ignore
/// // Type-safe: only CounterMsg can be sent
/// let counter: TypedRef<CounterMsg> = system.spawn_behavior("counter", counter_behavior).await?;
///
/// // Compile-time error if wrong message type
/// counter.tell(CounterMsg::Increment(5)).await?;
/// ```
pub struct TypedRef<M> {
    inner: Option<ActorRef>,
    name: String,
    system: Option<Arc<ActorSystem>>,
    _marker: PhantomData<M>,
}

impl<M> std::fmt::Debug for TypedRef<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedRef")
            .field("name", &self.name)
            .field("has_inner", &self.inner.is_some())
            .finish()
    }
}

impl<M> Clone for TypedRef<M> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            name: self.name.clone(),
            system: self.system.clone(),
            _marker: PhantomData,
        }
    }
}

impl<M> TypedRef<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    /// Create a typed reference wrapping an existing ActorRef
    pub fn new(name: &str, inner: ActorRef) -> Self {
        Self {
            inner: Some(inner),
            name: name.to_string(),
            system: None,
            _marker: PhantomData,
        }
    }

    /// Create a typed reference from a name (lazy resolution)
    pub(crate) fn from_name(name: &str, system: Arc<ActorSystem>) -> Self {
        Self {
            inner: None,
            name: name.to_string(),
            system: Some(system),
            _marker: PhantomData,
        }
    }

    /// Get the actor's name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Resolve the underlying ActorRef if not already resolved
    async fn resolve(&self) -> anyhow::Result<ActorRef> {
        if let Some(ref inner) = self.inner {
            return Ok(inner.clone());
        }

        if let Some(ref system) = self.system {
            // Get local actor by name
            system
                .local_actor_ref_by_name(&self.name)
                .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", self.name))
        } else {
            Err(anyhow::anyhow!(
                "TypedRef not initialized with system reference"
            ))
        }
    }

    /// Send a message without waiting for response (fire-and-forget)
    ///
    /// This is type-safe: only messages of type M can be sent.
    pub async fn tell(&self, msg: M) -> anyhow::Result<()> {
        let actor_ref = self.resolve().await?;
        actor_ref.tell(msg).await
    }

    /// Send a message and wait for a response
    ///
    /// # Type Parameters
    ///
    /// - `M`: The message type (input)
    /// - `R`: The expected response type
    ///
    /// Note: The response type is not checked at compile time for the receiver.
    /// Ensure the target actor returns the expected type.
    pub async fn ask<R>(&self, msg: M) -> anyhow::Result<R>
    where
        R: DeserializeOwned,
    {
        let actor_ref = self.resolve().await?;
        actor_ref.ask(msg).await
    }

    /// Send a message and wait for a response with timeout
    pub async fn ask_timeout<R>(&self, msg: M, timeout: Duration) -> anyhow::Result<R>
    where
        R: DeserializeOwned,
    {
        tokio::time::timeout(timeout, self.ask(msg))
            .await
            .map_err(|_| anyhow::anyhow!("Ask timeout after {:?}", timeout))?
    }

    /// Get the underlying untyped ActorRef
    ///
    /// Useful when you need to interact with APIs that expect ActorRef
    pub async fn as_untyped(&self) -> anyhow::Result<ActorRef> {
        self.resolve().await
    }
}
