//! Shared test fixtures for transport tests
#![allow(dead_code)]

use pulsing_actor::actor::Message;
use pulsing_actor::transport::{Http2Config, Http2Server, Http2ServerHandler};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Counter for tracking handler calls
#[derive(Default)]
pub struct TestCounters {
    pub ask_count: AtomicUsize,
    pub tell_count: AtomicUsize,
}

/// Test handler that echoes messages and counts calls
pub struct TestHandler {
    pub counters: Arc<TestCounters>,
}

impl TestHandler {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(TestCounters::default()),
        }
    }

    pub fn with_counters(counters: Arc<TestCounters>) -> Self {
        Self { counters }
    }
}

#[async_trait::async_trait]
impl Http2ServerHandler for TestHandler {
    async fn handle_message_simple(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> pulsing_actor::error::Result<Message> {
        self.counters.ask_count.fetch_add(1, Ordering::SeqCst);

        // Echo the payload with path and msg_type prepended
        let mut response = format!("{}:{}:", path, msg_type).into_bytes();
        response.extend(payload);
        Ok(Message::single("", response))
    }

    async fn handle_tell(
        &self,
        _path: &str,
        _msg_type: &str,
        _payload: Vec<u8>,
    ) -> pulsing_actor::error::Result<()> {
        self.counters.tell_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn handle_gossip(
        &self,
        _payload: Vec<u8>,
        _peer_addr: std::net::SocketAddr,
    ) -> pulsing_actor::error::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    async fn health_check(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "healthy",
            "ask_count": self.counters.ask_count.load(Ordering::SeqCst),
            "tell_count": self.counters.tell_count.load(Ordering::SeqCst),
        })
    }

    async fn cluster_members(&self) -> serde_json::Value {
        serde_json::json!([
            {
                "node_id": "12345",
                "addr": "127.0.0.1:8001",
                "status": "Alive"
            },
            {
                "node_id": "67890",
                "addr": "127.0.0.1:8002",
                "status": "Alive"
            }
        ])
    }

    async fn actors_list(&self, include_internal: bool) -> serde_json::Value {
        let mut actors = vec![
            serde_json::json!({
                "name": "counter-1",
                "type": "user",
                "actor_id": "12345:1",
                "class": "Counter",
                "module": "__main__"
            }),
            serde_json::json!({
                "name": "calculator",
                "type": "user",
                "actor_id": "12345:2",
                "class": "Calculator",
                "module": "__main__"
            }),
        ];

        if include_internal {
            actors.push(serde_json::json!({
                "name": "_python_actor_service",
                "type": "system",
                "actor_id": "12345:0"
            }));
        }

        serde_json::json!(actors)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Handler that supports streaming requests
pub struct StreamingHandler {
    pub counters: Arc<TestCounters>,
}

impl StreamingHandler {
    pub fn with_counters(counters: Arc<TestCounters>) -> Self {
        Self { counters }
    }
}

#[async_trait::async_trait]
impl Http2ServerHandler for StreamingHandler {
    async fn handle_message_full(
        &self,
        path: &str,
        msg: Message,
    ) -> pulsing_actor::error::Result<Message> {
        use futures::StreamExt;
        self.counters.ask_count.fetch_add(1, Ordering::SeqCst);

        match msg {
            Message::Single { msg_type, data } => {
                // Echo single message
                let mut response = format!("{}:{}:", path, msg_type).into_bytes();
                response.extend(data);
                Ok(Message::single("echo", response))
            }
            Message::Stream { stream, .. } => {
                // Collect all stream data and echo back as single response
                let mut collected = Vec::new();
                let mut stream = std::pin::pin!(stream);
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(Message::Single { data, .. }) => {
                            collected.extend(data);
                        }
                        Ok(Message::Stream { .. }) => {
                            return Err(pulsing_actor::error::PulsingError::from(
                                pulsing_actor::error::RuntimeError::Other(
                                    "Nested streams not supported".into(),
                                ),
                            ));
                        }
                        Err(e) => return Err(e),
                    }
                }
                let response = format!("{}:collected:{} bytes", path, collected.len()).into_bytes();
                Ok(Message::single("stream_echo", response))
            }
        }
    }

    async fn handle_message_simple(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> pulsing_actor::error::Result<Message> {
        self.counters.ask_count.fetch_add(1, Ordering::SeqCst);
        let mut response = format!("{}:{}:", path, msg_type).into_bytes();
        response.extend(payload);
        Ok(Message::single("", response))
    }

    async fn handle_tell(
        &self,
        _path: &str,
        _msg_type: &str,
        _payload: Vec<u8>,
    ) -> pulsing_actor::error::Result<()> {
        self.counters.tell_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn handle_gossip(
        &self,
        _payload: Vec<u8>,
        _peer_addr: std::net::SocketAddr,
    ) -> pulsing_actor::error::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Helper to create a test server with handler and return (server, counters, cancel_token)
pub async fn create_test_server() -> (Http2Server, Arc<TestCounters>, CancellationToken) {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    (server, counters, cancel)
}
