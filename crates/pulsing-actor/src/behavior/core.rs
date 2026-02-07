use super::context::BehaviorContext;
use super::reference::TypedRef;
use crate::actor::{Actor, ActorContext, IntoActor, Message};
use crate::error::{PulsingError, Result, RuntimeError};
use async_trait::async_trait;
use futures::future::BoxFuture;
use serde::{de::DeserializeOwned, Serialize};
use std::marker::PhantomData;
use tokio::sync::Mutex;

/// Action returned by a behavior after processing a message.
pub enum BehaviorAction<M> {
    Same,
    Become(Behavior<M>),
    Stop(Option<String>),
    AlreadyStopped,
}

impl<M> BehaviorAction<M> {
    pub fn stop() -> Self {
        Self::Stop(None)
    }

    pub fn stop_with_reason(reason: impl Into<String>) -> Self {
        Self::Stop(Some(reason.into()))
    }

    pub fn is_stop(&self) -> bool {
        matches!(self, Self::Stop(_) | Self::AlreadyStopped)
    }
}

pub type BehaviorFn<M> =
    Box<dyn FnMut(M, &mut BehaviorContext<M>) -> BoxFuture<'_, BehaviorAction<M>> + Send>;

/// A behavior wraps a message-handling function.
pub struct Behavior<M> {
    inner: BehaviorFn<M>,
    _marker: PhantomData<M>,
}

impl<M> Behavior<M>
where
    M: Send + 'static,
{
    pub fn new<F>(f: F) -> Self
    where
        F: FnMut(M, &mut BehaviorContext<M>) -> BoxFuture<'_, BehaviorAction<M>> + Send + 'static,
    {
        Self {
            inner: Box::new(f),
            _marker: PhantomData,
        }
    }

    pub async fn receive(&mut self, msg: M, ctx: &mut BehaviorContext<M>) -> BehaviorAction<M> {
        (self.inner)(msg, ctx).await
    }
}

/// IntoActor implementation for Behavior<M>.
impl<M> IntoActor for Behavior<M>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    type Actor = BehaviorWrapper<M>;

    fn into_actor(self) -> Self::Actor {
        BehaviorWrapper::new(self)
    }
}

/// Wrapper that allows Behavior<M> to be used as an Actor.
pub struct BehaviorWrapper<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    behavior: Mutex<Behavior<M>>,
    behavior_ctx: Mutex<Option<BehaviorContext<M>>>,
    name: Mutex<Option<String>>,
}

impl<M> BehaviorWrapper<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    pub fn new(behavior: Behavior<M>) -> Self {
        Self {
            behavior: Mutex::new(behavior),
            behavior_ctx: Mutex::new(None),
            name: Mutex::new(None),
        }
    }
}

impl<M> From<Behavior<M>> for BehaviorWrapper<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    fn from(behavior: Behavior<M>) -> Self {
        Self::new(behavior)
    }
}

#[async_trait]
impl<M> Actor for BehaviorWrapper<M>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> Result<Message> {
        let typed_msg: M = msg.unpack()?;

        let mut behavior = self.behavior.lock().await;
        let mut ctx_guard = self.behavior_ctx.lock().await;

        let ctx = ctx_guard.as_mut().ok_or_else(|| {
            PulsingError::from(RuntimeError::Other(
                "BehaviorContext not initialized".into(),
            ))
        })?;

        let action = behavior.receive(typed_msg, ctx).await;

        match action {
            BehaviorAction::Same => Message::pack(&()),
            BehaviorAction::Become(new_behavior) => {
                *behavior = new_behavior;
                Message::pack(&())
            }
            BehaviorAction::Stop(reason) => {
                let actor_name = self.name.lock().await;
                let name = actor_name.as_deref().unwrap_or("unknown");
                if let Some(ref r) = reason {
                    tracing::info!(actor = %name, reason = %r, "Behavior actor stopping");
                } else {
                    tracing::info!(actor = %name, "Behavior actor stopping");
                }

                drop(behavior);
                drop(ctx_guard);

                _ctx.cancel_token().cancel();

                Err(PulsingError::from(RuntimeError::Other(format!(
                    "Actor stopped: {}",
                    reason.unwrap_or_default()
                ))))
            }
            BehaviorAction::AlreadyStopped => {
                let actor_name = self.name.lock().await;
                let name = actor_name.as_deref().unwrap_or("unknown");
                tracing::warn!(actor = %name, "Message received after actor stopped");
                Err(PulsingError::from(RuntimeError::Other(
                    "Actor already stopped".into(),
                )))
            }
        }
    }

    async fn on_start(&mut self, ctx: &mut ActorContext) -> Result<()> {
        // Get or derive the actor name
        let actor_name = ctx
            .named_path()
            .map(String::from)
            .unwrap_or_else(|| format!("behavior-{}", ctx.id()));

        // Store name for logging
        *self.name.lock().await = Some(actor_name.clone());

        // Get system reference from the context (always available now)
        let system = ctx.system();

        // Initialize the behavior context
        let actor_id = *ctx.id();

        // Create typed self reference
        let self_ref = TypedRef::from_name(&actor_name, system.clone());

        let behavior_ctx = BehaviorContext::new(
            actor_name,
            actor_id,
            system,
            self_ref,
            ctx.cancel_token().clone(),
        );

        let mut ctx_guard = self.behavior_ctx.lock().await;
        *ctx_guard = Some(behavior_ctx);

        Ok(())
    }
}

// Behavior cannot Clone because it contains mutable state
// But new instances can be created via factory functions

/// Create a stateless behavior (no internal state, just message handling)
///
/// # Example
/// ```rust,ignore
/// let echo = stateless(|msg: String, ctx| {
///     Box::pin(async move {
///         println!("Received: {}", msg);
///         BehaviorAction::Same
///     })
/// });
/// ```
pub fn stateless<M, F>(mut f: F) -> Behavior<M>
where
    M: Send + 'static,
    F: FnMut(M, &mut BehaviorContext<M>) -> BoxFuture<'_, BehaviorAction<M>> + Send + 'static,
{
    Behavior::new(move |msg, ctx| f(msg, ctx))
}

/// Create a stateful behavior with encapsulated state
///
/// The state is owned by the behavior and can be mutated in the handler.
/// The handler receives: state, message, and context.
///
/// After `BehaviorAction::Stop`, subsequent messages will receive `BehaviorAction::AlreadyStopped`
/// instead of panicking.
///
/// # Example
/// ```rust,ignore
/// let counter = stateful(0, |count, msg, ctx| {
///     println!("Actor {} received message", ctx.name());
///     *count += msg;
///     BehaviorAction::Same
/// });
/// ```
pub fn stateful<S, M, F>(initial_state: S, handler: F) -> Behavior<M>
where
    S: Send + 'static,
    M: Send + 'static,
    F: Fn(&mut S, M, &BehaviorContext<M>) -> BehaviorAction<M> + Send + 'static,
{
    // Use a Cell-like pattern: state is moved into the closure and mutated directly.
    // This avoids both std::sync::Mutex (which blocks async runtime) and
    // tokio::sync::Mutex (which would require async in the handler).
    //
    // Safety: The closure is FnMut and only called sequentially by the actor runtime,
    // so no concurrent access is possible.
    let mut state = Some(initial_state);
    Behavior::new(move |msg, ctx| {
        // If state is None, actor has already stopped
        let Some(mut s) = state.take() else {
            return Box::pin(async { BehaviorAction::AlreadyStopped });
        };

        let action = handler(&mut s, msg, ctx);

        // Only restore state if not stopping
        if !action.is_stop() {
            state = Some(s);
        }

        Box::pin(async move { action })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize, serde::Deserialize)]
    enum TestMsg {
        Ping,
        Add(i32),
    }

    #[test]
    fn test_stateful_behavior_creation() {
        let _behavior: Behavior<TestMsg> = stateful(0i32, |count, msg, _ctx| match msg {
            TestMsg::Ping => BehaviorAction::Same,
            TestMsg::Add(n) => {
                *count += n;
                BehaviorAction::Same
            }
        });
    }

    #[test]
    fn test_stateless_behavior_creation() {
        let _behavior: Behavior<TestMsg> = stateless(|msg, _ctx| {
            Box::pin(async move {
                match msg {
                    TestMsg::Ping => BehaviorAction::Same,
                    TestMsg::Add(_) => BehaviorAction::stop(),
                }
            })
        });
    }
}
