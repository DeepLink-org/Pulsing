//! Actor execution context.

use super::mailbox::Envelope;
use super::reference::ActorRef;
use super::traits::{ActorId, Message, NodeId};
use lru::LruCache;
use serde::Serialize;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Context provided to actors during message handling.
pub struct ActorContext {
    actor_id: ActorId,
    cancel_token: CancellationToken,
    actor_refs: LruCache<ActorId, ActorRef>,
    system: Arc<dyn ActorSystemRef>,
    self_sender: mpsc::Sender<Envelope>,
    named_path: Option<String>,
}

/// Trait for system reference.
#[async_trait::async_trait]
pub trait ActorSystemRef: Send + Sync {
    async fn actor_ref(&self, id: &ActorId) -> crate::error::Result<ActorRef>;

    fn node_id(&self) -> NodeId;

    async fn watch(&self, watcher: &ActorId, target: &ActorId) -> crate::error::Result<()>;

    async fn unwatch(&self, watcher: &ActorId, target: &ActorId) -> crate::error::Result<()>;

    fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef>;
}

impl ActorContext {
    /// Create a new ActorContext with all required fields.
    ///
    /// This is the main constructor for runtime use. All fields are required.
    pub fn new(
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        cancel_token: CancellationToken,
        self_sender: mpsc::Sender<Envelope>,
        named_path: Option<String>,
    ) -> Self {
        Self {
            actor_id,
            cancel_token,
            actor_refs: LruCache::new(NonZeroUsize::new(1024).expect("1024 is non-zero")),
            system,
            self_sender,
            named_path,
        }
    }

    /// Create a context with system but without a named path.
    pub fn with_system(
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        cancel_token: CancellationToken,
        self_sender: mpsc::Sender<Envelope>,
    ) -> Self {
        Self::new(actor_id, system, cancel_token, self_sender, None)
    }

    /// Create a context with system and optional named path.
    pub fn with_system_and_name(
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        cancel_token: CancellationToken,
        self_sender: mpsc::Sender<Envelope>,
        named_path: Option<String>,
    ) -> Self {
        Self::new(actor_id, system, cancel_token, self_sender, named_path)
    }

    pub fn named_path(&self) -> Option<&str> {
        self.named_path.as_deref()
    }

    pub fn system(&self) -> Arc<dyn ActorSystemRef> {
        self.system.clone()
    }

    pub fn id(&self) -> &ActorId {
        &self.actor_id
    }

    /// Get the node ID from the system reference.
    pub fn node_id(&self) -> NodeId {
        self.system.node_id()
    }

    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    pub async fn actor_ref(&mut self, id: &ActorId) -> crate::error::Result<ActorRef> {
        if let Some(r) = self.actor_refs.get(id) {
            return Ok(r.clone());
        }

        let r = self.system.actor_ref(id).await?;
        self.actor_refs.put(*id, r.clone());
        Ok(r)
    }

    /// Schedule a delayed message to self.
    pub fn schedule_self<M: Serialize + Send + 'static>(
        &self,
        msg: M,
        delay: Duration,
    ) -> crate::error::Result<()> {
        let sender = self.self_sender.clone();
        let message = Message::pack(&msg)?;

        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let envelope = Envelope::tell(message);
            if let Err(e) = sender.send(envelope).await {
                tracing::warn!("Failed to deliver scheduled message: {}", e);
            }
        });

        Ok(())
    }

    /// Watch another actor.
    pub async fn watch(&self, target: &ActorId) -> crate::error::Result<()> {
        self.system.watch(&self.actor_id, target).await
    }

    /// Stop watching another actor.
    pub async fn unwatch(&self, target: &ActorId) -> crate::error::Result<()> {
        self.system.unwatch(&self.actor_id, target).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::{ActorSystem, SystemConfig};

    async fn create_test_context(actor_id: ActorId) -> (ActorContext, Arc<ActorSystem>) {
        let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();
        let cancel_token = CancellationToken::new();
        let (tx, _rx) = mpsc::channel(1);
        let system_ref = system.clone() as Arc<dyn ActorSystemRef>;
        let ctx = ActorContext::new(actor_id, system_ref, cancel_token, tx, None);
        (ctx, system)
    }

    #[tokio::test]
    async fn test_context_creation() {
        let (ctx, _system) = create_test_context(ActorId::generate()).await;
        // UUID-based IDs are non-zero
        assert_ne!(ctx.id().0, 0);
        assert!(!ctx.is_cancelled());
    }

    #[tokio::test]
    async fn test_context_cancellation() {
        let (ctx, _system) = create_test_context(ActorId::generate()).await;
        assert!(!ctx.is_cancelled());
        ctx.cancel_token().cancel();
        assert!(ctx.is_cancelled());
    }

    #[tokio::test]
    async fn test_context_node_id() {
        let (ctx, system) = create_test_context(ActorId::generate()).await;
        assert_eq!(ctx.node_id(), *system.node_id());
    }

    #[tokio::test]
    async fn test_context_multiple_actors() {
        let (ctx1, _system1) = create_test_context(ActorId::generate()).await;
        let (ctx2, _system2) = create_test_context(ActorId::generate()).await;
        let (ctx3, _system3) = create_test_context(ActorId::generate()).await;

        // UUID-based IDs should all be unique
        assert_ne!(ctx1.id(), ctx2.id());
        assert_ne!(ctx2.id(), ctx3.id());
        assert_ne!(ctx1.id(), ctx3.id());
    }

    #[tokio::test]
    async fn test_context_cancel_token_clone() {
        let (ctx, _system) = create_test_context(ActorId::generate()).await;
        let token = ctx.cancel_token().clone();

        assert!(!ctx.is_cancelled());
        assert!(!token.is_cancelled());

        token.cancel();

        assert!(ctx.is_cancelled());
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn test_context_actor_ref() {
        let (mut ctx, _system) = create_test_context(ActorId::generate()).await;
        let target_id = ActorId::generate();

        // actor_ref should fail for non-existent actor
        let result = ctx.actor_ref(&target_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_context_watch() {
        let (ctx, _system) = create_test_context(ActorId::generate()).await;
        let target_id = ActorId::generate();

        // watch should work with real system
        let result = ctx.watch(&target_id).await;
        // May fail if target doesn't exist, but should not panic
        let _ = result;
    }

    #[tokio::test]
    async fn test_context_unwatch() {
        let (ctx, _system) = create_test_context(ActorId::generate()).await;
        let target_id = ActorId::generate();

        // unwatch should work with real system
        let result = ctx.unwatch(&target_id).await;
        // May fail if target doesn't exist, but should not panic
        let _ = result;
    }
}
