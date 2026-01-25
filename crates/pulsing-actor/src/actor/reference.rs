//! Actor reference - location-transparent handle to an actor.

use super::address::ActorPath;
use super::mailbox::Envelope;
use super::traits::{ActorId, Message};
use serde::{de::DeserializeOwned, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};

/// Actor reference - handle for sending messages to an actor.
#[derive(Clone)]
pub struct ActorRef {
    pub(crate) actor_id: ActorId,

    pub(crate) inner: ActorRefInner,
}

/// Inner actor reference.
#[derive(Clone)]
pub enum ActorRefInner {
    Local(mpsc::Sender<Envelope>),

    Remote(Arc<RemoteActorRef>),

    Lazy(Arc<LazyActorRef>),
}

/// Remote actor reference.
pub struct RemoteActorRef {
    pub node_addr: SocketAddr,

    pub transport: Arc<dyn RemoteTransport>,
}

/// Lazy actor reference.
pub struct LazyActorRef {
    pub path: ActorPath,

    pub resolver: Arc<dyn ActorResolver>,

    cache: RwLock<Option<CachedRef>>,

    refresh_lock: tokio::sync::Mutex<()>,
}

struct CachedRef {
    actor_ref: ActorRef,
    cached_at: std::time::Instant,
}

const CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// Trait for resolving actor paths to ActorRefs.
#[async_trait::async_trait]
pub trait ActorResolver: Send + Sync {
    async fn resolve_path(&self, path: &ActorPath) -> anyhow::Result<ActorRef>;
}

impl LazyActorRef {
    pub fn new(path: ActorPath, resolver: Arc<dyn ActorResolver>) -> Self {
        Self {
            path,
            resolver,
            cache: RwLock::new(None),
            refresh_lock: tokio::sync::Mutex::new(()),
        }
    }

    async fn get(&self) -> anyhow::Result<ActorRef> {
        {
            let cache = self.cache.read().await;
            if let Some(ref cached) = *cache {
                if cached.cached_at.elapsed() < CACHE_TTL {
                    return Ok(cached.actor_ref.clone());
                }
            }
        }

        let _refresh_guard = self.refresh_lock.lock().await;

        {
            let cache = self.cache.read().await;
            if let Some(ref cached) = *cache {
                if cached.cached_at.elapsed() < CACHE_TTL {
                    return Ok(cached.actor_ref.clone());
                }
            }
        }

        let resolved = self.resolver.resolve_path(&self.path).await?;
        {
            let mut cache = self.cache.write().await;
            *cache = Some(CachedRef {
                actor_ref: resolved.clone(),
                cached_at: std::time::Instant::now(),
            });
        }
        Ok(resolved)
    }

    pub async fn invalidate(&self) {
        let mut cache = self.cache.write().await;
        *cache = None;
    }
}

/// Trait for remote transport (HTTP/2, TCP, etc.).
#[async_trait::async_trait]
pub trait RemoteTransport: Send + Sync {
    async fn request(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>>;

    async fn send(
        &self,
        actor_id: &ActorId,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<()>;

    /// Send a message and receive response (unified interface).
    async fn send_message(&self, actor_id: &ActorId, msg: Message) -> anyhow::Result<Message> {
        let Message::Single { msg_type, data } = msg else {
            return Err(anyhow::anyhow!("Streaming requests not yet supported"));
        };
        let response = self.request(actor_id, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

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
    pub fn local(actor_id: ActorId, sender: mpsc::Sender<Envelope>) -> Self {
        Self {
            actor_id,
            inner: ActorRefInner::Local(sender),
        }
    }

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

    /// Create a lazy actor reference that re-resolves on cache expiry
    ///
    /// This is useful for named actors that may migrate between nodes.
    /// The reference will automatically re-resolve after CACHE_TTL (5 seconds).
    pub fn lazy(path: ActorPath, resolver: Arc<dyn ActorResolver>) -> Self {
        Self {
            // Use a placeholder ID for lazy refs (all zeros)
            actor_id: ActorId::new(0),
            inner: ActorRefInner::Lazy(Arc::new(LazyActorRef::new(path, resolver))),
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

    /// Check if this is a lazy (re-resolving) reference
    pub fn is_lazy(&self) -> bool {
        matches!(self.inner, ActorRefInner::Lazy(_))
    }

    /// Invalidate cache for lazy references (force re-resolution on next call)
    pub async fn invalidate_cache(&self) {
        if let ActorRefInner::Lazy(lazy) = &self.inner {
            lazy.invalidate().await;
        }
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
            ActorRefInner::Lazy(lazy) => {
                // Resolve and delegate to the resolved reference
                let resolved = lazy.get().await?;
                // Box the recursive future to avoid infinite size
                Box::pin(resolved.send(msg)).await
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
            ActorRefInner::Lazy(lazy) => {
                // Resolve and delegate to the resolved reference
                let resolved = lazy.get().await?;
                // Box the recursive future to avoid infinite size
                Box::pin(resolved.send_oneway(msg)).await
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
        let actor_id = ActorId::generate();
        let actor_ref = ActorRef::local(actor_id, tx);

        actor_ref.tell(TestMsg { value: 42 }).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        // type_name includes module path
        assert!(envelope.msg_type().ends_with("TestMsg"));
    }

    #[tokio::test]
    async fn test_local_actor_ref_send_oneway() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::generate();
        let actor_ref = ActorRef::local(actor_id, tx);

        let msg = Message::single("TestMsg", b"hello");
        actor_ref.send_oneway(msg).await.unwrap();

        let envelope = rx.recv().await.unwrap();
        assert_eq!(envelope.msg_type(), "TestMsg");
        assert!(!envelope.expects_response());
    }
}
