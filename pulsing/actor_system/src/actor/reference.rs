//! Actor reference - location-transparent handle to an actor

use super::mailbox::Envelope;
use super::traits::{ActorId, Message, PayloadStream};
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
    /// Send a request and wait for response
    async fn request(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;

    /// Send a one-way message
    async fn send(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()>;

    /// Send a request and receive a stream of responses
    async fn request_stream(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<PayloadStream> {
        let _ = (actor_id, msg_type, payload);
        Err(anyhow::anyhow!("Streaming not supported by this transport"))
    }

    /// Send a message and receive response (unified interface)
    async fn send_message(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<Message> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming not supported by this transport"));
        };
        let response = self.request(actor_id, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

    /// Send a one-way message (unified interface)
    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming not supported by this transport"));
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

    /// Send a message and receive a response (unified interface)
    ///
    /// This is the primary method for actor communication. It handles all patterns:
    /// - Single request -> Single response
    /// - Single request -> Stream response
    /// - Stream request -> Single response
    /// - Stream request -> Stream response
    pub async fn send(&self, msg: Message) -> anyhow::Result<Message> {
        match &self.inner {
            ActorRefInner::Local(sender) => {
                let (tx, rx) = oneshot::channel();
                let envelope = Envelope::ask(msg, tx);

                sender
                    .send(envelope)
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;

                rx.await.map_err(|_| anyhow::anyhow!("Actor dropped"))?
            }
            ActorRefInner::Remote(remote) => {
                remote.transport.send_message(&self.actor_id, msg).await
            }
        }
    }

    /// Send a message and expect a streaming response
    pub async fn send_stream(&self, msg: Message) -> anyhow::Result<Message> {
        match &self.inner {
            ActorRefInner::Local(_) => {
                // For local actors, send() already handles stream responses
                // because the actor handler can return Message::Stream
                self.send(msg).await
            }
            ActorRefInner::Remote(remote) => {
                // For remote actors, we must use request_stream to set correct headers
                let Message::Single { msg_type, data } = msg else {
                    return Err(anyhow::anyhow!(
                        "Streaming requests not yet supported for remote actors"
                    ));
                };
                let stream = remote
                    .transport
                    .request_stream(&self.actor_id, &msg_type, data)
                    .await?;
                Ok(Message::Stream {
                    msg_type: String::new(),
                    stream,
                })
            }
        }
    }

    /// Send a fire-and-forget message (no response expected)
    pub async fn fire(&self, msg: Message) -> anyhow::Result<()> {
        match &self.inner {
            ActorRefInner::Local(sender) => {
                let envelope = Envelope::tell_msg(msg);

                sender
                    .send(envelope)
                    .await
                    .map_err(|_| anyhow::anyhow!("Actor mailbox closed"))?;
            }
            ActorRefInner::Remote(remote) => {
                remote.transport.send_oneway(&self.actor_id, msg).await?;
            }
        }

        Ok(())
    }

    /// Ask pattern - send a typed message and wait for typed response
    ///
    /// Automatically uses `type_name` to identify the message type.
    ///
    /// # Example
    /// ```ignore
    /// let pong: Pong = actor_ref.ask(Ping { value: 42 }).await?;
    /// ```
    pub async fn ask<M, R>(&self, msg: M) -> anyhow::Result<R>
    where
        M: Serialize + 'static,
        R: DeserializeOwned,
    {
        let actor_msg = Message::pack(&msg)?;
        let response = self.send(actor_msg).await?;
        response.unpack()
    }

    /// Tell pattern - send a message without waiting for response
    ///
    /// Automatically uses `type_name` to identify the message type.
    ///
    /// # Example
    /// ```ignore
    /// actor_ref.tell(Ping { value: 42 }).await?;
    /// ```
    pub async fn tell<M>(&self, msg: M) -> anyhow::Result<()>
    where
        M: Serialize + 'static,
    {
        let actor_msg = Message::pack(&msg)?;
        self.fire(actor_msg).await
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
        let actor_id = ActorId::local("test");
        let actor_ref = ActorRef::local(actor_id, tx);

        actor_ref.tell(TestMsg { value: 42 }).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        // type_name includes module path
        assert!(envelope.msg_type().ends_with("TestMsg"));
    }

    #[tokio::test]
    async fn test_local_actor_ref_fire() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::local("test");
        let actor_ref = ActorRef::local(actor_id, tx);

        let msg = Message::single("TestMsg", b"hello");
        actor_ref.fire(msg).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        assert_eq!(envelope.msg_type(), "TestMsg");
        assert!(!envelope.expects_response());
    }
}
