//! Typed actor references

use crate::actor::ActorRef;
use crate::actor::ActorSystemRef;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

/// Resolution mode for TypedRef
#[derive(Clone)]
enum ResolutionMode {
    /// Direct reference - always use this ActorRef
    Direct(ActorRef),
    /// Dynamic resolution - resolve by name each time (no caching)
    Dynamic(Arc<dyn ActorSystemRef>),
}

/// A type-safe actor reference
///
/// Unlike `ActorRef`, `TypedRef<M>` knows the message type at compile time,
/// providing type-safe message sending.
///
/// # Resolution Strategy
///
/// - **Direct mode**: Created with `TypedRef::new()`, uses the provided ActorRef directly
/// - **Dynamic mode**: Created via spawn, resolves actor by name on each call (no stale cache)
///
/// # Example
///
/// ```rust,ignore
/// // Spawn behavior directly (Behavior implements IntoActor)
/// let counter = system.spawn_named("actors/counter", counter_behavior).await?;
///
/// // Or wrap with TypedRef for type-safe sending
/// let counter: TypedRef<CounterMsg> = TypedRef::new("actors/counter", counter);
/// counter.tell(CounterMsg::Increment(5)).await?;
/// ```
pub struct TypedRef<M> {
    name: String,
    mode: ResolutionMode,
    _marker: PhantomData<M>,
}

impl<M> std::fmt::Debug for TypedRef<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedRef")
            .field("name", &self.name)
            .field(
                "mode",
                &match &self.mode {
                    ResolutionMode::Direct(_) => "direct",
                    ResolutionMode::Dynamic(_) => "dynamic",
                },
            )
            .finish()
    }
}

impl<M> Clone for TypedRef<M> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            mode: self.mode.clone(),
            _marker: PhantomData,
        }
    }
}

impl<M> TypedRef<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    /// Create a typed reference wrapping an existing ActorRef (direct mode)
    ///
    /// The provided ActorRef is used directly without re-resolution.
    /// Use this when you have a known, stable actor reference.
    pub fn new(name: &str, inner: ActorRef) -> Self {
        Self {
            name: name.to_string(),
            mode: ResolutionMode::Direct(inner),
            _marker: PhantomData,
        }
    }

    /// Create a typed reference from a name (dynamic resolution mode)
    ///
    /// The actor is resolved by name on each operation, ensuring
    /// the reference is always up-to-date (no stale cache).
    pub(crate) fn from_name(name: &str, system: Arc<dyn ActorSystemRef>) -> Self {
        Self {
            name: name.to_string(),
            mode: ResolutionMode::Dynamic(system),
            _marker: PhantomData,
        }
    }

    /// Get the actor's name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Resolve the underlying ActorRef
    ///
    /// - Direct mode: returns the stored ActorRef
    /// - Dynamic mode: looks up actor by name (fresh each time)
    fn resolve(&self) -> anyhow::Result<ActorRef> {
        match &self.mode {
            ResolutionMode::Direct(inner) => Ok(inner.clone()),
            ResolutionMode::Dynamic(system) => system
                .local_actor_ref_by_name(&self.name)
                .ok_or_else(|| anyhow::anyhow!("Actor not found: {}", self.name)),
        }
    }

    /// Send a message without waiting for response (fire-and-forget)
    ///
    /// This is type-safe: only messages of type M can be sent.
    pub async fn tell(&self, msg: M) -> anyhow::Result<()> {
        let actor_ref = self.resolve()?;
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
        let actor_ref = self.resolve()?;
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
    pub fn as_untyped(&self) -> anyhow::Result<ActorRef> {
        self.resolve()
    }

    /// Check if the referenced actor is currently alive
    pub fn is_alive(&self) -> bool {
        self.resolve().is_ok()
    }
}
