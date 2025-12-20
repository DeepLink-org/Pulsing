//! Actor reference - location-transparent handle to an actor

use super::mailbox::Envelope;
use super::traits::{ActorId, Message, RawMessage};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Actor reference - handle for sending messages to an actor
#[derive(Clone)]
pub struct ActorRef {
    /// The target actor's ID
    pub(crate) actor_id: ActorId,

    /// Inner implementation (local or remote)
    pub(crate) inner: ActorRefInner,
}

/// Inner actor reference - either local or remote
#[derive(Clone)]
pub enum ActorRefInner {
    /// Local actor - direct channel access
    Local(mpsc::Sender<Envelope>),

    /// Remote actor - via network transport
    Remote(Arc<RemoteActorRef>),
}

/// Remote actor reference
pub struct RemoteActorRef {
    /// Remote node address
    pub node_addr: SocketAddr,

    /// Transport client
    pub transport: Arc<dyn RemoteTransport>,
}

/// Trait for remote transport (TCP, etc.)
#[async_trait::async_trait]
pub trait RemoteTransport: Send + Sync {
    /// Send a request and wait for response
    async fn request(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;

    /// Send a one-way message (no response expected)
    async fn send(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()>;
}

impl ActorRef {
    /// Create a local actor reference
    pub fn local(actor_id: ActorId, sender: mpsc::Sender<Envelope>) -> Self {
        Self {
            actor_id,
            inner: ActorRefInner::Local(sender),
        }
    }

    /// Create a remote actor reference
    pub fn remote(
        actor_id: ActorId,
        node_addr: SocketAddr,
        transport: Arc<dyn RemoteTransport>,
    ) -> Self {
        Self {
            actor_id,
            inner: ActorRefInner::Remote(Arc::new(RemoteActorRef {
                node_addr,
                transport,
            })),
        }
    }

    /// Get the actor ID
    pub fn id(&self) -> &ActorId {
        &self.actor_id
    }

    /// Check if this is a local reference
    pub fn is_local(&self) -> bool {
        matches!(self.inner, ActorRefInner::Local(_))
    }

    /// Ask pattern - send a message and wait for response
    pub async fn ask<M, R>(&self, msg: M) -> anyhow::Result<R>
    where
        M: Message,
        R: Message,
    {
        let payload = bincode::serialize(&msg)?;
        let msg_type = M::type_id();

        let response = match &self.inner {
            ActorRefInner::Local(sender) => {
                let (tx, rx) = oneshot::channel();
                let envelope = Envelope::ask(msg_type.to_string(), payload, tx);

                sender
                    .send(envelope)
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

                rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))??
            }
            ActorRefInner::Remote(remote) => {
                remote
                    .transport
                    .request(&self.actor_id, msg_type, payload)
                    .await?
            }
        };

        Ok(bincode::deserialize(&response)?)
    }

    /// Tell pattern - send a message without waiting for response
    pub async fn tell<M>(&self, msg: M) -> anyhow::Result<()>
    where
        M: Message,
    {
        let payload = bincode::serialize(&msg)?;
        let msg_type = M::type_id();

        match &self.inner {
            ActorRefInner::Local(sender) => {
                let envelope = Envelope::tell(msg_type.to_string(), payload);

                sender
                    .send(envelope)
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;
            }
            ActorRefInner::Remote(remote) => {
                remote
                    .transport
                    .send(&self.actor_id, msg_type, payload)
                    .await?;
            }
        }

        Ok(())
    }

    /// Send raw message (type-erased)
    pub async fn send_raw(&self, msg: RawMessage) -> anyhow::Result<RawMessage> {
        match &self.inner {
            ActorRefInner::Local(sender) => {
                let (tx, rx) = oneshot::channel();
                let envelope = Envelope::ask(msg.msg_type.clone(), msg.payload, tx);

                sender
                    .send(envelope)
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

                let response = rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))??;
                Ok(RawMessage {
                    msg_type: "response".to_string(),
                    payload: response,
                })
            }
            ActorRefInner::Remote(remote) => {
                let response = remote
                    .transport
                    .request(&self.actor_id, &msg.msg_type, msg.payload)
                    .await?;
                Ok(RawMessage {
                    msg_type: "response".to_string(),
                    payload: response,
                })
            }
        }
    }

    /// Send raw message without response (fire-and-forget)
    pub async fn tell_raw(&self, msg: RawMessage) -> anyhow::Result<()> {
        match &self.inner {
            ActorRefInner::Local(sender) => {
                let envelope = Envelope::tell(msg.msg_type, msg.payload);

                sender
                    .send(envelope)
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;
            }
            ActorRefInner::Remote(remote) => {
                remote
                    .transport
                    .send(&self.actor_id, &msg.msg_type, msg.payload)
                    .await?;
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for ActorRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorRef")
            .field("actor_id", &self.actor_id)
            .field(
                "location",
                if self.is_local() {
                    &"local"
                } else {
                    &"remote"
                },
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct TestMsg {
        value: i32,
    }

    impl Message for TestMsg {
        fn type_id() -> &'static str {
            "TestMsg"
        }
    }

    #[tokio::test]
    async fn test_local_actor_ref_tell() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::local("test");
        let actor_ref = ActorRef::local(actor_id, tx);

        actor_ref.tell(TestMsg { value: 42 }).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        assert_eq!(envelope.msg_type, "TestMsg");

        let msg: TestMsg = bincode::deserialize(&envelope.payload).unwrap();
        assert_eq!(msg.value, 42);
    }
}

