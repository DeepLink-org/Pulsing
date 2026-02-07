//! Actor reference - location-transparent handle to an actor.

use super::address::ActorPath;
use super::mailbox::Envelope;
use super::traits::{ActorId, Message};
use crate::error::{PulsingError, Result, RuntimeError};
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
    async fn resolve_path(&self, path: &ActorPath) -> crate::error::Result<ActorRef>;
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

    async fn get(&self) -> Result<ActorRef> {
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
    ) -> Result<Vec<u8>>;

    async fn send(&self, actor_id: &ActorId, msg_type: &str, payload: Vec<u8>) -> Result<()>;

    /// Send a message and receive response (unified interface).
    async fn send_message(&self, actor_id: &ActorId, msg: Message) -> Result<Message> {
        let Message::Single { msg_type, data } = msg else {
            return Err(PulsingError::from(RuntimeError::Other(
                "Streaming requests not yet supported".into(),
            )));
        };
        let response = self.request(actor_id, &msg_type, data).await?;
        Ok(Message::single("", response))
    }

    async fn send_oneway(&self, actor_id: &ActorId, msg: Message) -> Result<()> {
        let Message::Single { msg_type, data } = msg else {
            return Err(PulsingError::from(RuntimeError::Other(
                "Streaming not supported for fire-and-forget".into(),
            )));
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
    pub async fn send(&self, msg: Message) -> Result<Message> {
        match &self.inner {
            ActorRefInner::Local(sender) => {
                let (tx, rx) = oneshot::channel();
                sender.send(Envelope::ask(msg, tx)).await.map_err(|_| {
                    PulsingError::from(RuntimeError::Other("Actor mailbox closed".into()))
                })?;
                rx.await
                    .map_err(|_| PulsingError::from(RuntimeError::Other("Actor dropped".into())))?
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
    pub async fn send_oneway(&self, msg: Message) -> Result<()> {
        match &self.inner {
            ActorRefInner::Local(sender) => sender.send(Envelope::tell(msg)).await.map_err(|_| {
                PulsingError::from(RuntimeError::Other("Actor mailbox closed".into()))
            }),
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
    pub async fn ask<M, R>(&self, msg: M) -> Result<R>
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
    pub async fn tell<M>(&self, msg: M) -> Result<()>
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
    use crate::actor::ActorPath;

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct TestMsg {
        value: i32,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct TestReply {
        result: i32,
    }

    #[tokio::test]
    async fn test_local_actor_ref_tell() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::generate();
        let actor_ref = ActorRef::local(actor_id, tx);

        actor_ref.tell(TestMsg { value: 42 }).await.unwrap();

        let envelope = rx.recv().await.unwrap();
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

    #[tokio::test]
    async fn test_local_actor_ref_ask_success() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::generate();
        let actor_ref = ActorRef::local(actor_id, tx.clone());

        let reply_handle = tokio::spawn(async move {
            let envelope = rx.recv().await.unwrap();
            let (msg, responder) = envelope.into_parts();
            let req: TestMsg = msg.unpack().unwrap();
            responder.send(Ok(Message::pack(&TestReply {
                result: req.value * 2,
            })
            .unwrap()));
        });

        let reply: TestReply = actor_ref.ask(TestMsg { value: 21 }).await.unwrap();
        assert_eq!(reply.result, 42);
        reply_handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_local_actor_ref_mailbox_closed() {
        let (tx, rx) = mpsc::channel(16);
        let actor_id = ActorId::generate();
        let actor_ref = ActorRef::local(actor_id, tx);
        drop(rx);

        let err = actor_ref.tell(TestMsg { value: 1 }).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("mailbox")
                || err.to_string().to_lowercase().contains("closed")
        );
    }

    #[tokio::test]
    async fn test_actor_ref_is_local_is_lazy() {
        let (tx, _rx) = mpsc::channel(16);
        let local_ref = ActorRef::local(ActorId::generate(), tx);
        assert!(local_ref.is_local());
        assert!(!local_ref.is_lazy());

        let path = ActorPath::new("a/b").unwrap();
        struct MockResolver;
        #[async_trait::async_trait]
        impl ActorResolver for MockResolver {
            async fn resolve_path(&self, _path: &ActorPath) -> Result<ActorRef> {
                Err(PulsingError::from(RuntimeError::actor_not_found("mock")))
            }
        }
        let lazy_ref = ActorRef::lazy(path, Arc::new(MockResolver));
        assert!(!lazy_ref.is_local());
        assert!(lazy_ref.is_lazy());
    }

    #[tokio::test]
    async fn test_remote_actor_ref_delegates_to_transport() {
        struct MockTransport {
            request_result: Result<Vec<u8>>,
            send_result: Result<()>,
        }
        #[async_trait::async_trait]
        impl RemoteTransport for MockTransport {
            async fn request(
                &self,
                _actor_id: &ActorId,
                _msg_type: &str,
                _payload: Vec<u8>,
            ) -> Result<Vec<u8>> {
                self.request_result.clone()
            }
            async fn send(
                &self,
                _actor_id: &ActorId,
                _msg_type: &str,
                _payload: Vec<u8>,
            ) -> Result<()> {
                self.send_result.clone()
            }
        }

        let transport = Arc::new(MockTransport {
            request_result: Ok(bincode::serialize(&TestReply { result: 100 }).unwrap()),
            send_result: Ok(()),
        });
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let remote_ref = ActorRef::remote(ActorId::generate(), addr, transport);

        let reply: TestReply = remote_ref.ask(TestMsg { value: 50 }).await.unwrap();
        assert_eq!(reply.result, 100);
        remote_ref.tell(TestMsg { value: 0 }).await.unwrap();
    }

    #[tokio::test]
    async fn test_remote_actor_ref_transport_error() {
        struct FailingTransport;
        #[async_trait::async_trait]
        impl RemoteTransport for FailingTransport {
            async fn request(&self, _: &ActorId, _: &str, _: Vec<u8>) -> Result<Vec<u8>> {
                Err(PulsingError::from(RuntimeError::connection_failed(
                    "127.0.0.1:1".to_string(),
                    "refused".to_string(),
                )))
            }
            async fn send(&self, _: &ActorId, _: &str, _: Vec<u8>) -> Result<()> {
                Err(PulsingError::from(RuntimeError::connection_failed(
                    "127.0.0.1:1".to_string(),
                    "refused".to_string(),
                )))
            }
        }
        let addr: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let remote_ref = ActorRef::remote(ActorId::generate(), addr, Arc::new(FailingTransport));

        let err = remote_ref
            .ask::<TestMsg, TestReply>(TestMsg { value: 1 })
            .await
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("connection")
                || err.to_string().to_lowercase().contains("refused")
        );
    }

    #[tokio::test]
    async fn test_lazy_ref_resolves_and_caches() {
        let (tx, mut rx) = mpsc::channel(16);
        let actor_id = ActorId::generate();
        let local_ref = ActorRef::local(actor_id, tx.clone());

        struct ResolverThatReturns {
            ref_to_return: ActorRef,
        }
        #[async_trait::async_trait]
        impl ActorResolver for ResolverThatReturns {
            async fn resolve_path(&self, _path: &ActorPath) -> Result<ActorRef> {
                Ok(self.ref_to_return.clone())
            }
        }

        let path = ActorPath::new("svc/echo").unwrap();
        let lazy_ref = ActorRef::lazy(
            path,
            Arc::new(ResolverThatReturns {
                ref_to_return: local_ref,
            }),
        );

        let reply_handle = tokio::spawn(async move {
            for _ in 0..2 {
                let Some(envelope) = rx.recv().await else {
                    break;
                };
                let (msg, responder) = envelope.into_parts();
                let req: TestMsg = msg.unpack().unwrap();
                responder.send(Ok(Message::pack(&TestReply {
                    result: req.value + 1,
                })
                .unwrap()));
            }
        });

        let r1: TestReply = lazy_ref.ask(TestMsg { value: 10 }).await.unwrap();
        assert_eq!(r1.result, 11);
        let r2: TestReply = lazy_ref.ask(TestMsg { value: 20 }).await.unwrap();
        assert_eq!(r2.result, 21);
        reply_handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_lazy_ref_resolve_failure() {
        struct FailingResolver;
        #[async_trait::async_trait]
        impl ActorResolver for FailingResolver {
            async fn resolve_path(&self, path: &ActorPath) -> Result<ActorRef> {
                Err(PulsingError::from(RuntimeError::named_actor_not_found(
                    path.as_str(),
                )))
            }
        }
        let path = ActorPath::new("svc/missing").unwrap();
        let lazy_ref = ActorRef::lazy(path, Arc::new(FailingResolver));

        let err = lazy_ref
            .ask::<TestMsg, TestReply>(TestMsg { value: 1 })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing") || err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_invalidate_cache() {
        let path = ActorPath::new("a/b").unwrap();
        struct CountResolver {
            count: std::sync::atomic::AtomicU32,
        }
        #[async_trait::async_trait]
        impl ActorResolver for CountResolver {
            async fn resolve_path(&self, _path: &ActorPath) -> Result<ActorRef> {
                let (tx, rx) = mpsc::channel(16);
                let c = self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let id = ActorId::new(c as u128);
                let ref_ = ActorRef::local(id, tx);
                drop(rx);
                Ok(ref_)
            }
        }
        let resolver = Arc::new(CountResolver {
            count: std::sync::atomic::AtomicU32::new(0),
        });
        let lazy_ref = ActorRef::lazy(path.clone(), resolver);
        lazy_ref.invalidate_cache().await;
        lazy_ref.invalidate_cache().await;
    }
}
