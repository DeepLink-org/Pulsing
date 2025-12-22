//! Actor execution context

use super::reference::ActorRef;
use super::traits::{ActorId, NodeId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Context provided to actors during message handling
///
/// Provides access to:
/// - `id()` - The actor's assigned ID
/// - `actor_ref()` - Get references to other actors
/// - `watch()`/`unwatch()` - Monitor other actors
/// - `is_cancelled()` - Check if shutdown was requested
pub struct ActorContext {
    /// The actor's own ID
    actor_id: ActorId,

    /// Local node ID
    node_id: Option<NodeId>,

    /// Cancellation token for graceful shutdown
    cancel_token: CancellationToken,

    /// Cached actor references
    actor_refs: HashMap<ActorId, ActorRef>,

    /// System reference for spawning new actors
    system: Option<Arc<dyn ActorSystemRef>>,
}

/// Trait for system reference (to avoid circular dependency)
#[async_trait::async_trait]
pub trait ActorSystemRef: Send + Sync {
    /// Get an actor reference by ID
    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef>;

    /// Get the local node ID
    fn node_id(&self) -> &NodeId;

    /// Watch an actor - will receive Terminated message when the watched actor stops
    async fn watch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()>;

    /// Stop watching an actor
    async fn unwatch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()>;
}

impl ActorContext {
    /// Create a new context (for testing)
    pub fn new(actor_id: ActorId) -> Self {
        Self {
            actor_id,
            node_id: None,
            cancel_token: CancellationToken::new(),
            actor_refs: HashMap::new(),
            system: None,
        }
    }

    /// Create context with system reference
    pub fn with_system(
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        cancel_token: CancellationToken,
    ) -> Self {
        let node_id = Some(system.node_id().clone());
        Self {
            actor_id,
            node_id,
            cancel_token,
            actor_refs: HashMap::new(),
            system: Some(system),
        }
    }

    /// Get the actor's ID
    pub fn id(&self) -> &ActorId {
        &self.actor_id
    }

    /// Get the local node ID
    pub fn node_id(&self) -> Option<&NodeId> {
        self.node_id.as_ref()
    }

    /// Get the cancellation token
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Check if shutdown was requested
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Get an actor reference
    pub async fn actor_ref(&mut self, id: &ActorId) -> anyhow::Result<ActorRef> {
        // Check cache first
        if let Some(r) = self.actor_refs.get(id) {
            return Ok(r.clone());
        }

        // Get from system
        if let Some(ref system) = self.system {
            let r = system.actor_ref(id).await?;
            self.actor_refs.insert(id.clone(), r.clone());
            return Ok(r);
        }

        Err(anyhow::anyhow!("No system reference available"))
    }

    /// Schedule a delayed message to self
    pub fn schedule_self<M>(&self, _msg: M, _delay: std::time::Duration) {
        // TODO: implement scheduling
        todo!("Scheduling not yet implemented")
    }

    /// Watch another actor - will receive Terminated message when it stops
    pub async fn watch(&self, target: &ActorId) -> anyhow::Result<()> {
        if let Some(ref system) = self.system {
            system.watch(&self.actor_id, target).await
        } else {
            Err(anyhow::anyhow!("No system reference available"))
        }
    }

    /// Stop watching another actor
    pub async fn unwatch(&self, target: &ActorId) -> anyhow::Result<()> {
        if let Some(ref system) = self.system {
            system.unwatch(&self.actor_id, target).await
        } else {
            Err(anyhow::anyhow!("No system reference available"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = ActorContext::new(ActorId::local("test"));
        assert_eq!(ctx.id().name, "test");
        assert!(!ctx.is_cancelled());
    }

    #[test]
    fn test_context_cancellation() {
        let ctx = ActorContext::new(ActorId::local("test"));
        assert!(!ctx.is_cancelled());
        ctx.cancel_token().cancel();
        assert!(ctx.is_cancelled());
    }
}
