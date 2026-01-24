//! Behavior-based actor programming model.

mod context;
mod core;
mod reference;
mod spawn;

pub use context::BehaviorContext;
pub use core::{stateful, stateless, Behavior, BehaviorAction, BehaviorFn, BehaviorWrapper};
pub use reference::TypedRef;
