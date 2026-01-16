//! Spawn behavior-based actors into the ActorSystem

use super::context::BehaviorContext;
use super::core::{Behavior, BehaviorAction};
use super::reference::TypedRef;
use crate::actor::{Actor, ActorContext, Message};
use crate::system::{ActorSystem, SpawnOptions};
use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

/// Wrapper that bridges Behavior to the traditional Actor trait
struct BehaviorActor<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    /// Actor name (for context initialization)
    name: String,
    /// Reference to the actor system
    system: Arc<ActorSystem>,
    behavior: Mutex<Behavior<M>>,
    #[allow(dead_code)]
    msg_receiver: Mutex<mpsc::Receiver<M>>,
    msg_sender: mpsc::Sender<M>,
    behavior_ctx: Mutex<Option<BehaviorContext<M>>>,
}

impl<M> BehaviorActor<M>
where
    M: Serialize + DeserializeOwned + Send + 'static,
{
    fn new(
        name: String,
        system: Arc<ActorSystem>,
        behavior: Behavior<M>,
        buffer_size: usize,
    ) -> Self {
        let (tx, rx) = mpsc::channel(buffer_size);
        Self {
            name,
            system,
            behavior: Mutex::new(behavior),
            msg_receiver: Mutex::new(rx),
            msg_sender: tx,
            behavior_ctx: Mutex::new(None),
        }
    }
}

#[async_trait]
impl<M> Actor for BehaviorActor<M>
where
    M: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        // Deserialize the incoming message
        let typed_msg: M = msg.unpack()?;

        // Get mutable access to behavior and context
        let mut behavior = self.behavior.lock().await;
        let mut ctx_guard = self.behavior_ctx.lock().await;

        let ctx = ctx_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("BehaviorContext not initialized"))?;

        // Process the message
        let action = behavior.receive(typed_msg, ctx).await;

        // Handle the action
        match action {
            BehaviorAction::Same => {
                // Return empty response for tell, actual response for ask
                Message::pack(&())
            }
            BehaviorAction::Become(new_behavior) => {
                *behavior = new_behavior;
                Message::pack(&())
            }
            BehaviorAction::Stop => {
                // Signal stop - the actor system will handle cleanup
                Err(anyhow::anyhow!("Actor stopped"))
            }
        }
    }

    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        // Initialize the behavior context
        let actor_id = *ctx.id();

        // Create typed self reference
        let self_ref = TypedRef::from_name(&self.name, self.system.clone());

        let behavior_ctx = BehaviorContext::new(
            self.name.clone(),
            actor_id,
            self.system.clone(),
            self_ref,
            ctx.cancel_token().clone(),
            self.msg_sender.clone(),
        );

        let mut ctx_guard = self.behavior_ctx.lock().await;
        *ctx_guard = Some(behavior_ctx);

        Ok(())
    }
}

/// Extension trait for spawning behavior-based actors
#[async_trait]
pub trait BehaviorSpawner {
    /// Spawn a behavior-based actor
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use pulsing_actor::behavior::*;
    ///
    /// let counter_ref: TypedRef<CounterMsg> = system
    ///     .spawn_behavior("counter", counter(0))
    ///     .await?;
    /// ```
    async fn spawn_behavior<M>(
        self: &Arc<Self>,
        name: impl AsRef<str> + Send,
        behavior: Behavior<M>,
    ) -> anyhow::Result<TypedRef<M>>
    where
        M: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Spawn a behavior-based actor with custom mailbox capacity
    async fn spawn_behavior_with_capacity<M>(
        self: &Arc<Self>,
        name: impl AsRef<str> + Send,
        behavior: Behavior<M>,
        mailbox_capacity: usize,
    ) -> anyhow::Result<TypedRef<M>>
    where
        M: Serialize + DeserializeOwned + Send + Sync + 'static;
}

#[async_trait]
impl BehaviorSpawner for ActorSystem {
    async fn spawn_behavior<M>(
        self: &Arc<Self>,
        name: impl AsRef<str> + Send,
        behavior: Behavior<M>,
    ) -> anyhow::Result<TypedRef<M>>
    where
        M: Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        self.spawn_behavior_with_capacity(name, behavior, 256).await
    }

    async fn spawn_behavior_with_capacity<M>(
        self: &Arc<Self>,
        name: impl AsRef<str> + Send,
        behavior: Behavior<M>,
        mailbox_capacity: usize,
    ) -> anyhow::Result<TypedRef<M>>
    where
        M: Serialize + DeserializeOwned + Send + Sync + 'static,
    {
        let name_str = name.as_ref().to_string();
        let actor = BehaviorActor::new(name_str.clone(), self.clone(), behavior, mailbox_capacity);
        let options = SpawnOptions::new().mailbox_capacity(mailbox_capacity);
        let actor_ref = self.spawn_with_options(&name_str, actor, options).await?;

        Ok(TypedRef::new(&name_str, actor_ref))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::{stateful, BehaviorAction};
    use crate::system::SystemConfig;

    #[derive(Debug, Clone, Serialize, serde::Deserialize)]
    enum TestMsg {
        Ping,
        Increment(i32),
    }

    #[tokio::test]
    async fn test_spawn_behavior() {
        let system = Arc::new(ActorSystem::new(SystemConfig::standalone()).await.unwrap());

        let counter = stateful(0i32, |count, msg| match msg {
            TestMsg::Ping => BehaviorAction::Same,
            TestMsg::Increment(n) => {
                *count += n;
                BehaviorAction::Same
            }
        });

        let counter_ref: TypedRef<TestMsg> =
            system.spawn_behavior("counter", counter).await.unwrap();

        // Type-safe message sending
        counter_ref.tell(TestMsg::Increment(5)).await.unwrap();
        counter_ref.tell(TestMsg::Ping).await.unwrap();

        system.shutdown().await.unwrap();
    }
}
