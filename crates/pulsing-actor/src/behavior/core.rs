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
    /// Stop the actor
    Stop,
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

// Behavior 不能 Clone，因为它包含可变状态
// 但可以通过工厂函数创建新的实例

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
/// Due to Rust's ownership rules, the handler must be a sync function that
/// returns a BoxFuture. Use `Box::pin(async move { ... })` in the handler.
///
/// # Example
/// ```rust,ignore
/// let counter = stateful(0, |count: &mut i32, msg: CounterMsg, _ctx| {
///     let result = match msg {
///         CounterMsg::Increment(n) => {
///             *count += n;
///             BehaviorAction::Same
///         }
///         CounterMsg::Get => BehaviorAction::Same,
///     };
///     Box::pin(async move { result })
/// });
/// ```
pub fn stateful<S, M, F>(initial_state: S, handler: F) -> Behavior<M>
where
    S: Send + 'static,
    M: Send + 'static,
    F: Fn(&mut S, M) -> BehaviorAction<M> + Send + 'static,
{
    let state = std::sync::Arc::new(std::sync::Mutex::new(initial_state));
    Behavior::new(move |msg, _ctx| {
        let mut guard = state.lock().unwrap();
        let action = handler(&mut guard, msg);
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
        let _behavior: Behavior<TestMsg> = stateful(0i32, |count, msg| match msg {
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
                    TestMsg::Add(_) => BehaviorAction::Stop,
                }
            })
        });
    }
}
