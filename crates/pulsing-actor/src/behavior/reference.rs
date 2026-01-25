use crate::actor::ActorRef;
use crate::actor::ActorSystemRef;
use crate::error::{PulsingError, RuntimeError};
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
enum ResolutionMode {
    Direct(ActorRef),
    Dynamic(Arc<dyn ActorSystemRef>),
}

/// A type-safe actor reference.
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
    /// Create a typed reference wrapping an existing ActorRef.
    pub fn new(name: &str, inner: ActorRef) -> Self {
        Self {
            name: name.to_string(),
            mode: ResolutionMode::Direct(inner),
            _marker: PhantomData,
        }
    }

    /// Create a typed reference from a name.
    pub(crate) fn from_name(name: &str, system: Arc<dyn ActorSystemRef>) -> Self {
        Self {
            name: name.to_string(),
            mode: ResolutionMode::Dynamic(system),
            _marker: PhantomData,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn resolve(&self) -> anyhow::Result<ActorRef> {
        match &self.mode {
            ResolutionMode::Direct(inner) => Ok(inner.clone()),
            ResolutionMode::Dynamic(system) => {
                system.local_actor_ref_by_name(&self.name).ok_or_else(|| {
                    anyhow::Error::from(PulsingError::from(RuntimeError::actor_not_found(
                        self.name.clone(),
                    )))
                })
            }
        }
    }

    /// Send a message without waiting for response.
    pub async fn tell(&self, msg: M) -> anyhow::Result<()> {
        let actor_ref = self.resolve()?;
        actor_ref.tell(msg).await
    }

    /// Send a message and wait for a response.
    pub async fn ask<R>(&self, msg: M) -> anyhow::Result<R>
    where
        R: DeserializeOwned,
    {
        let actor_ref = self.resolve()?;
        actor_ref.ask(msg).await
    }

    /// Send a message and wait for a response with timeout.
    pub async fn ask_timeout<R>(&self, msg: M, timeout: Duration) -> anyhow::Result<R>
    where
        R: DeserializeOwned,
    {
        tokio::time::timeout(timeout, self.ask(msg))
            .await
            .map_err(|_| anyhow::anyhow!("Ask timeout after {:?}", timeout))?
    }

    /// Get the underlying untyped ActorRef.
    pub fn as_untyped(&self) -> anyhow::Result<ActorRef> {
        self.resolve()
    }

    pub fn is_alive(&self) -> bool {
        self.resolve().is_ok()
    }
}
