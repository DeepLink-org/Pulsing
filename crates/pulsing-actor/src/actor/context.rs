//! Actor execution context.

use super::mailbox::Envelope;
use super::reference::ActorRef;
use super::traits::{ActorId, Message, NodeId};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Context provided to actors during message handling.
pub struct ActorContext {
    actor_id: ActorId,

    node_id: Option<NodeId>,

    cancel_token: CancellationToken,

    actor_refs: HashMap<ActorId, ActorRef>,

    system: Option<Arc<dyn ActorSystemRef>>,

    self_sender: Option<mpsc::Sender<Envelope>>,

    named_path: Option<String>,
}

/// Trait for system reference.
#[async_trait::async_trait]
pub trait ActorSystemRef: Send + Sync {
    async fn actor_ref(&self, id: &ActorId) -> anyhow::Result<ActorRef>;

    fn node_id(&self) -> NodeId;

    async fn watch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()>;

    async fn unwatch(&self, watcher: &ActorId, target: &ActorId) -> anyhow::Result<()>;

    fn local_actor_ref_by_name(&self, name: &str) -> Option<ActorRef>;
}

impl ActorContext {
    pub fn new(actor_id: ActorId) -> Self {
        Self {
            actor_id,
            node_id: None,
            cancel_token: CancellationToken::new(),
            actor_refs: HashMap::new(),
            system: None,
            self_sender: None,
            named_path: None,
        }
    }

    pub fn with_system(
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        cancel_token: CancellationToken,
        self_sender: mpsc::Sender<Envelope>,
    ) -> Self {
        let node_id = Some(system.node_id());
        Self {
            actor_id,
            node_id,
            cancel_token,
            actor_refs: HashMap::new(),
            system: Some(system),
            self_sender: Some(self_sender),
            named_path: None,
        }
    }

    pub fn with_system_and_name(
        actor_id: ActorId,
        system: Arc<dyn ActorSystemRef>,
        cancel_token: CancellationToken,
        self_sender: mpsc::Sender<Envelope>,
        named_path: Option<String>,
    ) -> Self {
        let node_id = Some(system.node_id());
        Self {
            actor_id,
            node_id,
            cancel_token,
            actor_refs: HashMap::new(),
            system: Some(system),
            self_sender: Some(self_sender),
            named_path,
        }
    }

    pub fn named_path(&self) -> Option<&str> {
        self.named_path.as_deref()
    }

    pub fn system(&self) -> Option<Arc<dyn ActorSystemRef>> {
        self.system.clone()
    }

    pub fn id(&self) -> &ActorId {
        &self.actor_id
    }

    pub fn node_id(&self) -> Option<&NodeId> {
        self.node_id.as_ref()
    }

    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    pub async fn actor_ref(&mut self, id: &ActorId) -> anyhow::Result<ActorRef> {
        if let Some(r) = self.actor_refs.get(id) {
            return Ok(r.clone());
        }

        if let Some(ref system) = self.system {
            let r = system.actor_ref(id).await?;
            self.actor_refs.insert(*id, r.clone());
            return Ok(r);
        }

        Err(anyhow::anyhow!("No system reference available"))
    }

    /// Schedule a delayed message to self.
    pub fn schedule_self<M: Serialize + Send + 'static>(
        &self,
        msg: M,
        delay: Duration,
    ) -> anyhow::Result<()> {
        let sender = self.self_sender.clone().ok_or_else(|| {
            anyhow::anyhow!("No self sender available (context not fully initialized)")
        })?;

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
    pub async fn watch(&self, target: &ActorId) -> anyhow::Result<()> {
        if let Some(ref system) = self.system {
            system.watch(&self.actor_id, target).await
        } else {
            Err(anyhow::anyhow!("No system reference available"))
        }
    }

    /// Stop watching another actor.
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
        let ctx = ActorContext::new(ActorId::local(1));
        assert_eq!(ctx.id().local_id(), 1);
        assert!(!ctx.is_cancelled());
    }

    #[test]
    fn test_context_cancellation() {
        let ctx = ActorContext::new(ActorId::local(1));
        assert!(!ctx.is_cancelled());
        ctx.cancel_token().cancel();
        assert!(ctx.is_cancelled());
    }

    #[test]
    fn test_context_node_id_none() {
        let ctx = ActorContext::new(ActorId::local(1));
        assert!(ctx.node_id().is_none());
    }

    #[test]
    fn test_context_multiple_actors() {
        let ctx1 = ActorContext::new(ActorId::local(1));
        let ctx2 = ActorContext::new(ActorId::local(2));
        let ctx3 = ActorContext::new(ActorId::local(3));

        assert_eq!(ctx1.id().local_id(), 1);
        assert_eq!(ctx2.id().local_id(), 2);
        assert_eq!(ctx3.id().local_id(), 3);
    }

    #[test]
    fn test_context_cancel_token_clone() {
        let ctx = ActorContext::new(ActorId::local(1));
        let token = ctx.cancel_token().clone();

        assert!(!ctx.is_cancelled());
        assert!(!token.is_cancelled());

        token.cancel();

        assert!(ctx.is_cancelled());
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn test_context_actor_ref_no_system() {
        let mut ctx = ActorContext::new(ActorId::local(1));
        let target_id = ActorId::local(2);

        let result = ctx.actor_ref(&target_id).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No system reference"));
    }

    #[tokio::test]
    async fn test_context_watch_no_system() {
        let ctx = ActorContext::new(ActorId::local(1));
        let target_id = ActorId::local(2);

        let result = ctx.watch(&target_id).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No system reference"));
    }

    #[tokio::test]
    async fn test_context_unwatch_no_system() {
        let ctx = ActorContext::new(ActorId::local(1));
        let target_id = ActorId::local(2);

        let result = ctx.unwatch(&target_id).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No system reference"));
    }
}
