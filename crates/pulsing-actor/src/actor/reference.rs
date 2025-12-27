//! Actor reference - location-transparent handle to an actor

use super::mailbox::Envelope;
use super::traits::{ActorId, Message};
use serde::{de::DeserializeOwned, Serialize};
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

/// Trait for remote transport (HTTP/2, TCP, etc.)
#[async_trait::async_trait]
pub trait RemoteTransport: Send + Sync {
    /// Send a request and wait for response (low-level)
    async fn request(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;

    /// Send a one-way message (low-level)
    async fn send(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()>;

    /// Send a message and receive response (unified interface)
    ///
    /// This is the primary method for communication. It automatically handles
    /// both single and stream responses based on the server's response type.
    async fn send_message(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<Message> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming requests not yet supported"));
        };
        let response = self.request(actor_id, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

    /// Send a one-way message (unified interface)
    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!(
                "Streaming not supported for fire-and-forget"
            ));
        };
        self.send(actor_id, &msg_type, data).await
    }
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

    /// Send a raw message and receive response (low-level API)
    ///
    /// Use this when you need direct access to `Message`, e.g., for streaming.
    /// For type-safe communication, prefer `ask()` and `tell()`.
    pub async fn send(&self, msg: Message) -> anyhow::Result<Message> {
        match &self.inner {
            ActorRefInner::Local(sender) => {
                let (tx, rx) = oneshot::channel();
                sender
                    .send(Envelope::ask(msg, tx))
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;
                rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
            }
            ActorRefInner::Remote(remote) => {
                remote.transport.send_message(&self.actor_id, msg).await
            }
        }
    }

    /// Send a raw message without waiting for response (low-level fire-and-forget)
    pub async fn send_oneway(&self, msg: Message) -> anyhow::Result<()> {
        match &self.inner {
            ActorRefInner::Local(sender) => sender
                .send(Envelope::tell(msg))
                .await
                .map_err(|_| anyhow::anyhow!("Actor mailbox closed")),
            ActorRefInner::Remote(remote) => {
                remote.transport.send_oneway(&self.actor_id, msg).await
            }
        }
    }

    /// Ask - send typed message and wait for typed response
    ///
    /// # Example
    /// ```ignore
    /// let pong: Pong = actor.ask(Ping { value: 42 }).await?;
    /// ```
    pub async fn ask<M, R>(&self, msg: M) -> anyhow::Result<R>
    where
        M: Serialize + 'static,
        R: DeserializeOwned,
    {
        self.send(Message::pack(&msg)?).await?.unpack()
    }

    /// Tell - send typed message without waiting for response (fire-and-forget)
    ///
    /// # Example
    /// ```ignore
    /// actor.tell(Ping { value: 42 }).await?;
    /// ```
    pub async fn tell<M>(&self, msg: M) -> anyhow::Result<()>
    where
        M: Serialize + 'static,
    {
        self.send_oneway(Message::pack(&msg)?).await
    }
}

impl std::fmt::Debug for ActorRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActorRef")
            .field("actor_id", &self.actor_id)
            .field(
                "location",
                if self.is_local() { &"local" } else { &"remote" },
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

    #[tokio::test]
    async fn test_local_actor_ref_tell() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::local(1);
        let actor_ref = ActorRef::local(actor_id, tx);

        actor_ref.tell(TestMsg { value: 42 }).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        // type_name includes module path
        assert!(envelope.msg_type().ends_with("TestMsg"));
    }

    #[tokio::test]
    async fn test_local_actor_ref_send_oneway() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::local(1);
        let actor_ref = ActorRef::local(actor_id, tx);

        let msg = Message::single("TestMsg", b"hello");
        actor_ref.send_oneway(msg).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        assert_eq!(envelope.msg_type(), "TestMsg");
        assert!(!envelope.expects_response());
    }
}
