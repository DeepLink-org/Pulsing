use super::reference::TypedRef;
use crate::actor::ActorId;
use crate::actor::ActorSystemRef;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Context provided to behavior handlers.
pub struct BehaviorContext<M> {
    actor_name: String,
    actor_id: ActorId,
    system: Arc<dyn ActorSystemRef>,
    self_ref: TypedRef<M>,
    cancel_token: CancellationToken,
    _marker: PhantomData<M>,
}

impl<M> BehaviorContext<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    pub(crate) fn new(
        actor_name: String,
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        self_ref: TypedRef<M>,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            actor_name,
            actor_id,
            system,
            self_ref,
            cancel_token,
            _marker: PhantomData,
        }
    }

    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }

    pub fn name(&self) -> &str {
        &self.actor_name
    }

    /// Get a typed reference to self.
    pub fn self_ref(&self) -> TypedRef<M> {
        self.self_ref.clone()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Get a typed reference to another behavior-based actor by name.
    pub fn typed_ref<N>(&self, name: &str) -> TypedRef<N>
    where
        N: Serialize + DeserializeOwned + Send + 'static,
    {
        TypedRef::from_name(name, self.system.clone())
    }

    /// Schedule a message to be sent to self after a delay.
    pub fn schedule_self(&self, msg: M, delay: Duration) {
        let actor_ref = match self.self_ref.as_untyped() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Failed to resolve self for scheduled message: {}", e);
                return;
            }
        };
        let cancel = self.cancel_token.clone();

        tokio::spawn(async move {
            tokio::select! {
                _ = tokio::time::sleep(delay) => {
                    if !cancel.is_cancelled() {
                        if let Err(e) = actor_ref.tell(msg).await {
                            tracing::warn!("Failed to deliver scheduled message: {}", e);
                        }
                    }
                }
                _ = cancel.cancelled() => {
                    tracing::debug!("Scheduled message cancelled due to actor stop");
                }
            }
        });
    }

    pub fn system(&self) -> &Arc<dyn ActorSystemRef> {
        &self.system
    }

    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }
}
