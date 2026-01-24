//! Typed actor context for behavior-based actors

use super::reference::TypedRef;
use crate::actor::ActorId;
use crate::actor::ActorSystemRef;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Context provided to behavior handlers
///
/// Unlike the traditional ActorContext, this is parameterized by message type,
/// providing type-safe self-references and scheduling.
pub struct BehaviorContext<M> {
    /// Actor's name
    actor_name: String,
    /// Actor's unique identifier
    actor_id: ActorId,
    /// Reference to the actor system
    system: Arc<dyn ActorSystemRef>,
    /// Typed self-reference for receiving messages
    self_ref: TypedRef<M>,
    /// Cancellation token for graceful shutdown
    cancel_token: CancellationToken,
    _marker: PhantomData<M>,
}

impl<M> BehaviorContext<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    /// Create a new behavior context
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

    /// Get the actor's unique identifier
    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }

    /// Get the actor's name
    pub fn name(&self) -> &str {
        &self.actor_name
    }

    /// Get a typed reference to self
    ///
    /// This can be passed to other actors for reply-to patterns
    pub fn self_ref(&self) -> TypedRef<M> {
        self.self_ref.clone()
    }

    /// Check if the actor should stop
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Get a typed reference to another behavior-based actor by name
    ///
    /// # Type Safety
    ///
    /// The caller must ensure the target actor actually accepts messages of type N.
    /// This is a runtime check - if types don't match, sends will fail at serialization.
    pub fn typed_ref<N>(&self, name: &str) -> TypedRef<N>
    where
        N: Serialize + DeserializeOwned + Send + 'static,
    {
        TypedRef::from_name(name, self.system.clone())
    }

    /// Schedule a message to be sent to self after a delay
    ///
    /// The message will be delivered to this actor after the specified duration.
    /// If the actor is stopped before the delay expires, the message will not be sent.
    ///
    /// The message goes through the actor's normal mailbox, ensuring proper ordering.
    pub fn schedule_self(&self, msg: M, delay: Duration) {
        // Resolve the ActorRef upfront (ActorRef is Send + Sync)
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
                    // Only send if actor is still running
                    if !cancel.is_cancelled() {
                        if let Err(e) = actor_ref.tell(msg).await {
                            tracing::warn!("Failed to deliver scheduled message: {}", e);
                        }
                    }
                }
                _ = cancel.cancelled() => {
                    // Actor stopped, don't send the message
                    tracing::debug!("Scheduled message cancelled due to actor stop");
                }
            }
        });
    }

    /// Get a reference to the underlying actor system
    pub fn system(&self) -> &Arc<dyn ActorSystemRef> {
        &self.system
    }

    /// Get the cancellation token for cooperative shutdown
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }
}
