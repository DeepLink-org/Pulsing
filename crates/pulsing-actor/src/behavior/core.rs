//! Behavior definitions and combinators

use super::context::BehaviorContext;
use futures::future::BoxFuture;
use std::marker::PhantomData;

/// Action returned by a behavior after processing a message
pub enum BehaviorAction<M> {
    /// Keep the current behavior
    Same,
    /// Switch to a new behavior (state machine transition)
    Become(Behavior<M>),
    /// Stop the actor gracefully with optional reason
    Stop(Option<String>),
    /// Actor is already stopped (internal use)
    /// This is returned when messages arrive after Stop
    AlreadyStopped,
}

impl<M> BehaviorAction<M> {
    /// Create a Stop action without reason
    pub fn stop() -> Self {
        Self::Stop(None)
    }

    /// Create a Stop action with reason
    pub fn stop_with_reason(reason: impl Into<String>) -> Self {
        Self::Stop(Some(reason.into()))
    }

    /// Check if this action indicates the actor should stop
    pub fn is_stop(&self) -> bool {
        matches!(self, Self::Stop(_) | Self::AlreadyStopped)
    }
}

/// The core behavior function type
pub type BehaviorFn<M> =
    Box<dyn FnMut(M, &mut BehaviorContext<M>) -> BoxFuture<'_, BehaviorAction<M>> + Send>;

/// A behavior wraps a message-handling function
///
/// Behaviors are the fundamental building block of this actor model.
/// An actor is simply a behavior that processes messages.
pub struct Behavior<M> {
    inner: BehaviorFn<M>,
    _marker: PhantomData<M>,
}

impl<M> Behavior<M>
where
    M: Send + 'static,
{
    /// Create a new behavior from a function
    pub fn new<F>(f: F) -> Self
    where
        F: FnMut(M, &mut BehaviorContext<M>) -> BoxFuture<'_, BehaviorAction<M>> + Send + 'static,
    {
        Self {
            inner: Box::new(f),
            _marker: PhantomData,
        }
    }

    /// Process a message with this behavior
    pub async fn receive(&mut self, msg: M, ctx: &mut BehaviorContext<M>) -> BehaviorAction<M> {
        (self.inner)(msg, ctx).await
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
