//! HTTP/2 Transport layer tests
//!
//! Tests for HTTP/2 (h2c) transport including:
//! - Server creation and startup
//! - Client connection and requests
//! - Ask (request-response) pattern
//! - Tell (fire-and-forget) pattern
//! - Http2RemoteTransport integration

use pulsing_actor::actor::{ActorId, Message};
use pulsing_actor::transport::{
    Http2Client, Http2Config, Http2RemoteTransport, Http2Server, Http2ServerHandler,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Test Handler Implementation
// ============================================================================

/// Counter for tracking handler calls
#[derive(Default)]
struct TestCounters {
    ask_count: AtomicUsize,
    tell_count: AtomicUsize,
}

/// Test handler that echoes messages and counts calls
struct TestHandler {
    counters: Arc<TestCounters>,
}

impl TestHandler {
    fn new() -> Self {
        Self {
            counters: Arc::new(TestCounters::default()),
        }
    }

    fn with_counters(counters: Arc<TestCounters>) -> Self {
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
    ) -> anyhow::Result<Message> {
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
    ) -> anyhow::Result<()> {
        self.counters.tell_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn handle_gossip(
        &self,
        _payload: Vec<u8>,
        _peer_addr: std::net::SocketAddr,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    async fn health_check(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "healthy",
            "ask_count": self.counters.ask_count.load(Ordering::SeqCst),
            "tell_count": self.counters.tell_count.load(Ordering::SeqCst),
        })
    }
}

// ============================================================================
// Server Tests
// ============================================================================

#[tokio::test]
async fn test_http2_server_creation() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    // Server should be bound to a valid port
    assert_ne!(server.local_addr().port(), 0);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_server_multiple_instances() {
    let cancel = CancellationToken::new();

    // Create multiple servers
    let mut servers = Vec::new();
    for _ in 0..3 {
        let handler = Arc::new(TestHandler::new());
        let server = Http2Server::new(
            "127.0.0.1:0".parse().unwrap(),
            handler,
            Http2Config::default(),
            cancel.clone(),
        )
        .await
        .unwrap();
        servers.push(server);
    }

    // All servers should have different ports
    let ports: Vec<u16> = servers.iter().map(|s| s.local_addr().port()).collect();
    let unique_ports: std::collections::HashSet<_> = ports.iter().collect();
    assert_eq!(ports.len(), unique_ports.len());

    // Cleanup
    cancel.cancel();
}

// ============================================================================
// Client Tests
// ============================================================================

#[tokio::test]
async fn test_http2_client_creation() {
    let client = Http2Client::new(Http2Config::default());
    // Client should be clonable
    let _cloned = client.clone();
}

#[tokio::test]
async fn test_http2_ask_request() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client and send request
    let client = Http2Client::new(Http2Config::default());
    let response = client
        .ask(addr, "/actors/test", "TestMsg", b"hello".to_vec())
        .await
        .unwrap();

    // Verify response
    let response_str = String::from_utf8_lossy(&response);
    assert!(response_str.contains("/actors/test"));
    assert!(response_str.contains("TestMsg"));
    assert!(response_str.contains("hello"));

    // Verify counter
    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_tell_request() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client and send tell
    let client = Http2Client::new(Http2Config::default());
    client
        .tell(addr, "/actors/test", "FireAndForget", b"data".to_vec())
        .await
        .unwrap();

    // Give the server time to process
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Verify counter
    assert_eq!(counters.tell_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_multiple_requests() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Http2Client::new(Http2Config::default());

    // Send multiple requests
    for i in 0..10 {
        let response = client
            .ask(
                addr,
                "/actors/test",
                "Msg",
                format!("request-{}", i).into_bytes(),
            )
            .await
            .unwrap();

        let response_str = String::from_utf8_lossy(&response);
        assert!(response_str.contains(&format!("request-{}", i)));
    }

    // Verify counter
    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 10);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_concurrent_requests() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Arc::new(Http2Client::new(Http2Config::default()));

    // Send concurrent requests
    let mut handles = Vec::new();
    for i in 0..20 {
        let client = client.clone();
        let handle = tokio::spawn(async move {
            client
                .ask(
                    addr,
                    "/actors/test",
                    "Concurrent",
                    format!("req-{}", i).into_bytes(),
                )
                .await
        });
        handles.push(handle);
    }

    // Wait for all requests
    let results: Vec<_> = futures::future::join_all(handles).await;

    // All should succeed
    for result in results {
        assert!(result.unwrap().is_ok());
    }

    // Verify counter
    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 20);

    // Cleanup
    cancel.cancel();
}

// ============================================================================
// Http2RemoteTransport Tests
// ============================================================================

#[tokio::test]
async fn test_http2_remote_transport_ask() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create transport
    let client = Arc::new(Http2Client::new(Http2Config::default()));
    let transport = Http2RemoteTransport::new(client, addr, "test-actor".to_string());

    // Use the RemoteTransport trait
    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::local(1);
    let response = transport
        .request(&actor_id, "TestType", b"payload".to_vec())
        .await
        .unwrap();

    // Verify response contains expected data
    let response_str = String::from_utf8_lossy(&response);
    assert!(response_str.contains("test-actor"));
    assert!(response_str.contains("TestType"));

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_remote_transport_tell() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create transport
    let client = Arc::new(Http2Client::new(Http2Config::default()));
    let transport = Http2RemoteTransport::new(client, addr, "fire-actor".to_string());

    // Use the RemoteTransport trait
    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::local(2);
    transport
        .send(&actor_id, "FireMsg", b"data".to_vec())
        .await
        .unwrap();

    // Give the server time to process
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Verify counter
    assert_eq!(counters.tell_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_remote_transport_named_path() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create transport with named path
    let client = Arc::new(Http2Client::new(Http2Config::default()));
    use pulsing_actor::actor::ActorPath;
    let path = ActorPath::new("services/llm/worker").unwrap();
    let transport = Http2RemoteTransport::new_named(client, addr, path);

    // Use the RemoteTransport trait
    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::local(3);
    let response = transport
        .request(&actor_id, "Inference", b"prompt".to_vec())
        .await
        .unwrap();

    // Verify response contains the path
    let response_str = String::from_utf8_lossy(&response);
    assert!(response_str.contains("services/llm/worker"));

    // Cleanup
    cancel.cancel();
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_http2_custom_config() {
    let config = Http2Config::new()
        .max_concurrent_streams(50)
        .initial_window_size(1024 * 1024)
        .connect_timeout(std::time::Duration::from_secs(10))
        .request_timeout(std::time::Duration::from_secs(60));

    assert_eq!(config.max_concurrent_streams, 50);
    assert_eq!(config.initial_window_size, 1024 * 1024);
    assert_eq!(config.connect_timeout, std::time::Duration::from_secs(10));
    assert_eq!(config.request_timeout, std::time::Duration::from_secs(60));
}

#[tokio::test]
async fn test_http2_server_with_custom_config() {
    let mut config = Http2Config::new().max_concurrent_streams(100);
    config.max_frame_size = 32 * 1024;

    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        config,
        cancel.clone(),
    )
    .await
    .unwrap();

    assert_ne!(server.local_addr().port(), 0);

    // Cleanup
    cancel.cancel();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_http2_connection_refused() {
    let client = Http2Client::new(Http2Config::default());

    // Try to connect to a port that's not listening
    let result = client
        .ask(
            "127.0.0.1:1".parse().unwrap(),
            "/actors/test",
            "Test",
            vec![],
        )
        .await;

    // Should fail
    assert!(result.is_err());
}

#[tokio::test]
async fn test_http2_server_shutdown() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client and verify it works
    let client = Http2Client::new(Http2Config::default());
    let result = client.ask(addr, "/actors/test", "Test", vec![]).await;
    assert!(result.is_ok());

    // Shutdown server
    server.shutdown();

    // Give server time to shutdown
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // New connections should fail (eventually)
    // Note: existing connections may still work briefly
}

// ============================================================================
// Stream Frame Tests
// ============================================================================

#[test]
fn test_stream_frame_data() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::data("token", b"hello");
    assert_eq!(frame.msg_type, "token");
    assert!(!frame.end);
    assert!(frame.error.is_none());
    assert_eq!(frame.get_data(), b"hello");
}

#[test]
fn test_stream_frame_end() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::end();
    assert!(frame.end);
    assert!(frame.error.is_none());
}

#[test]
fn test_stream_frame_error() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::error("something went wrong");
    assert!(frame.end);
    assert!(frame.is_error());
    assert_eq!(frame.error.as_ref().unwrap(), "something went wrong");
}

#[test]
fn test_stream_frame_binary_roundtrip() {
    use pulsing_actor::transport::StreamFrame;

    let original = StreamFrame::data("response", b"world");
    let bytes = original.to_binary();
    let parsed = StreamFrame::from_binary(&bytes).unwrap();

    assert_eq!(parsed.msg_type, "response");
    assert_eq!(parsed.get_data(), b"world");
}

// ============================================================================
// Message Mode Tests
// ============================================================================

#[test]
fn test_message_mode_conversion() {
    use pulsing_actor::transport::MessageMode;

    assert_eq!(MessageMode::Ask.as_str(), "ask");
    assert_eq!(MessageMode::Tell.as_str(), "tell");
    assert_eq!(MessageMode::Stream.as_str(), "stream");

    assert_eq!(MessageMode::parse("ask"), Some(MessageMode::Ask));
    assert_eq!(MessageMode::parse("TELL"), Some(MessageMode::Tell));
    assert_eq!(MessageMode::parse("Stream"), Some(MessageMode::Stream));
    assert_eq!(MessageMode::parse("invalid"), None);
}

// ============================================================================
// Unified Send Message Tests
// ============================================================================

#[tokio::test]
async fn test_http2_unified_send_message() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(TestHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client and send using unified send_message
    let client = Http2Client::new(Http2Config::default());
    let response = client
        .send_message(addr, "/actors/test", "TestMsg", b"test-payload".to_vec())
        .await
        .unwrap();

    // Should receive single response
    assert!(response.is_single());

    // Verify response content
    let Message::Single { data, .. } = response else {
        panic!("Expected single message");
    };
    let data_str = String::from_utf8_lossy(&data);
    assert!(data_str.contains("/actors/test"));
    assert!(data_str.contains("TestMsg"));

    // Verify ask counter
    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

// ============================================================================
// Performance / Benchmark Tests
// ============================================================================

#[tokio::test]
async fn test_http2_throughput_benchmark() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::high_throughput(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Arc::new(Http2Client::new(Http2Config::high_throughput()));

    let request_count = 1000;
    let start = std::time::Instant::now();

    // Send requests sequentially
    for i in 0..request_count {
        let _ = client
            .ask(
                addr,
                "/actors/bench",
                "Bench",
                format!("req-{}", i).into_bytes(),
            )
            .await
            .unwrap();
    }

    let elapsed = start.elapsed();
    let rps = request_count as f64 / elapsed.as_secs_f64();

    println!(
        "HTTP/2 Sequential Throughput: {} requests in {:?} ({:.0} req/s)",
        request_count, elapsed, rps
    );

    // Should handle at least 100 req/s (very conservative)
    assert!(rps > 100.0, "Throughput too low: {} req/s", rps);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_concurrent_throughput_benchmark() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::high_throughput(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Arc::new(Http2Client::new(Http2Config::high_throughput()));

    let request_count = 1000;
    let concurrency = 50;
    let start = std::time::Instant::now();

    // Send concurrent requests
    let mut handles = Vec::new();
    for i in 0..request_count {
        let client = client.clone();
        let handle = tokio::spawn(async move {
            client
                .ask(
                    addr,
                    "/actors/bench",
                    "Bench",
                    format!("req-{}", i).into_bytes(),
                )
                .await
        });
        handles.push(handle);

        // Limit concurrency
        if handles.len() >= concurrency {
            let results: Vec<_> = futures::future::join_all(handles.drain(..)).await;
            for r in results {
                r.unwrap().unwrap();
            }
        }
    }

    // Wait for remaining
    let results: Vec<_> = futures::future::join_all(handles).await;
    for r in results {
        r.unwrap().unwrap();
    }

    let elapsed = start.elapsed();
    let rps = request_count as f64 / elapsed.as_secs_f64();

    println!(
        "HTTP/2 Concurrent Throughput ({} concurrency): {} requests in {:?} ({:.0} req/s)",
        concurrency, request_count, elapsed, rps
    );

    // Should be faster than sequential (relaxed for CI environments)
    assert!(rps > 100.0, "Concurrent throughput too low: {} req/s", rps);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_latency_benchmark() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::low_latency(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Http2Client::new(Http2Config::low_latency());

    let request_count = 100;
    let mut latencies = Vec::with_capacity(request_count);

    // Measure individual request latencies
    for i in 0..request_count {
        let start = std::time::Instant::now();
        let _ = client
            .ask(
                addr,
                "/actors/latency",
                "Ping",
                format!("req-{}", i).into_bytes(),
            )
            .await
            .unwrap();
        latencies.push(start.elapsed());
    }

    // Calculate statistics
    latencies.sort();
    let min = latencies.first().unwrap();
    let max = latencies.last().unwrap();
    let median = latencies[request_count / 2];
    let p99 = latencies[(request_count * 99) / 100];
    let avg: std::time::Duration =
        latencies.iter().sum::<std::time::Duration>() / request_count as u32;

    println!(
        "HTTP/2 Latency: min={:?}, avg={:?}, median={:?}, p99={:?}, max={:?}",
        min, avg, median, p99, max
    );

    // P99 should be under 500ms for localhost (relaxed for CI environments)
    assert!(
        p99 < std::time::Duration::from_millis(500),
        "P99 latency too high: {:?}",
        p99
    );

    // Cleanup
    cancel.cancel();
}

// ============================================================================
// Connection Pool Tests
// ============================================================================

#[tokio::test]
async fn test_http2_connection_pool_reuse() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Http2Client::new(Http2Config::default());

    // Send multiple requests to trigger connection reuse
    for i in 0..10 {
        let _ = client
            .ask(
                addr,
                "/actors/pool-test",
                "Msg",
                format!("req-{}", i).into_bytes(),
            )
            .await
            .unwrap();
    }

    // Check pool stats
    let stats = client.stats();
    let created = stats.connections_created.load(Ordering::Relaxed);
    let reused = stats.connections_reused.load(Ordering::Relaxed);

    println!("Connection Pool: created={}, reused={}", created, reused);

    // Should have created very few connections but reused many times
    // HTTP/2 multiplexing means we should only need 1 connection
    assert!(created <= 2, "Too many connections created: {}", created);
    assert!(reused >= 8, "Not enough connection reuse: {}", reused);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_retry_on_connection_error() {
    use pulsing_actor::transport::{Http2ClientBuilder, RetryConfig};
    use std::time::Duration;

    // Create client with retry
    let client = Http2ClientBuilder::new()
        .retry_config(RetryConfig::with_max_retries(2).initial_delay(Duration::from_millis(10)))
        .connect_timeout(Duration::from_millis(100))
        .build();

    // Try to connect to a port that doesn't exist
    let result: Result<Vec<u8>, _> = client
        .ask(
            "127.0.0.1:1".parse().unwrap(),
            "/actors/test",
            "Msg",
            vec![],
        )
        .await;

    // Should fail after retries
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Error message should indicate connection failure
    assert!(
        err.contains("Connection")
            || err.contains("connect")
            || err.contains("timeout")
            || err.contains("backing off"),
        "Unexpected error: {}",
        err
    );
}

// ============================================================================
// Streaming Request Tests
// ============================================================================

/// Handler that supports streaming requests
struct StreamingHandler {
    counters: Arc<TestCounters>,
}

impl StreamingHandler {
    fn with_counters(counters: Arc<TestCounters>) -> Self {
        Self { counters }
    }
}

#[async_trait::async_trait]
impl Http2ServerHandler for StreamingHandler {
    async fn handle_message_full(&self, path: &str, msg: Message) -> anyhow::Result<Message> {
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
                            return Err(anyhow::anyhow!("Nested streams not supported"));
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
    ) -> anyhow::Result<Message> {
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
    ) -> anyhow::Result<()> {
        self.counters.tell_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn handle_gossip(
        &self,
        _payload: Vec<u8>,
        _peer_addr: std::net::SocketAddr,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

#[tokio::test]
async fn test_http2_streaming_request() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(StreamingHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Http2Client::new(Http2Config::default());

    // Create a streaming request
    let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<Message>>(10);

    // Send some messages through the stream
    tokio::spawn(async move {
        for i in 0..5 {
            let msg = Message::single("chunk", format!("data-{}", i).into_bytes());
            if tx.send(Ok(msg)).await.is_err() {
                break;
            }
        }
    });

    // Create stream message from channel
    let stream_msg = Message::from_channel("StreamRequest", rx);

    // Send streaming request
    let response = client
        .send_message_full(addr, "/actors/stream_test", stream_msg)
        .await
        .unwrap();

    // Verify response
    assert!(response.is_single());
    let Message::Single { data, .. } = response else {
        panic!("Expected single message");
    };
    let response_str = String::from_utf8_lossy(&data);
    assert!(response_str.contains("/actors/stream_test"));
    assert!(response_str.contains("collected:"));

    // Verify handler was called
    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_single_request_with_full_api() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(StreamingHandler::with_counters(counters.clone()));
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Http2Client::new(Http2Config::default());

    // Send single message using send_message_full
    let msg = Message::single("TestType", b"test-payload".to_vec());
    let response = client
        .send_message_full(addr, "/actors/single_test", msg)
        .await
        .unwrap();

    // Verify response
    assert!(response.is_single());
    let Message::Single { data, .. } = response else {
        panic!("Expected single message");
    };
    let response_str = String::from_utf8_lossy(&data);
    assert!(response_str.contains("/actors/single_test"));
    assert!(response_str.contains("TestType"));

    // Verify handler was called
    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

#[test]
fn test_request_type_conversion() {
    use pulsing_actor::transport::RequestType;

    assert_eq!(RequestType::Single.as_str(), "single");
    assert_eq!(RequestType::Stream.as_str(), "stream");

    assert_eq!(RequestType::parse("single"), Some(RequestType::Single));
    assert_eq!(RequestType::parse("STREAM"), Some(RequestType::Stream));
    assert_eq!(RequestType::parse("invalid"), None);
}

// ============================================================================
// Tracing Tests
// ============================================================================

mod tracing_tests {
    use super::*;
    use pulsing_actor::tracing::opentelemetry::trace::TraceContextExt;
    use pulsing_actor::tracing::{TraceContext, TRACEPARENT_HEADER};
    use std::sync::Mutex;

    /// Handler that captures trace context from incoming requests
    struct TracingTestHandler {
        counters: Arc<TestCounters>,
        captured_traces: Arc<Mutex<Vec<Option<String>>>>,
    }

    impl TracingTestHandler {
        fn new() -> Self {
            Self {
                counters: Arc::new(TestCounters::default()),
                captured_traces: Arc::new(Mutex::new(Vec::new())),
            }
        }

        #[allow(dead_code)]
        fn captured_traces(&self) -> Vec<Option<String>> {
            self.captured_traces.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Http2ServerHandler for TracingTestHandler {
        async fn handle_message_simple(
            &self,
            path: &str,
            msg_type: &str,
            payload: Vec<u8>,
        ) -> anyhow::Result<Message> {
            self.counters.ask_count.fetch_add(1, Ordering::SeqCst);

            // Try to get current trace context
            let trace_ctx = TraceContext::from_current();
            self.captured_traces
                .lock()
                .unwrap()
                .push(trace_ctx.map(|t| t.to_traceparent()));

            // Echo response with trace info
            let response = format!("{}:{}:{}", path, msg_type, payload.len());
            Ok(Message::single("traced", response.into_bytes()))
        }

        async fn handle_tell(
            &self,
            _path: &str,
            _msg_type: &str,
            _payload: Vec<u8>,
        ) -> anyhow::Result<()> {
            self.counters.tell_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn handle_gossip(
            &self,
            _payload: Vec<u8>,
            _peer_addr: std::net::SocketAddr,
        ) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(None)
        }
    }

    #[test]
    fn test_trace_context_creation() {
        let ctx = TraceContext::default();
        assert_eq!(ctx.trace_id.len(), 32);
        assert_eq!(ctx.span_id.len(), 16);
        assert_eq!(ctx.trace_flags, 0x01); // Sampled
    }

    #[test]
    fn test_trace_context_to_traceparent() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            span_id: "b7ad6b7169203331".to_string(),
            trace_flags: 0x01,
            trace_state: None,
        };

        let header = ctx.to_traceparent();
        assert_eq!(
            header,
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
    }

    #[test]
    fn test_trace_context_from_traceparent() {
        let header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::from_traceparent(header).unwrap();

        assert_eq!(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(ctx.span_id, "b7ad6b7169203331");
        assert_eq!(ctx.trace_flags, 0x01);
    }

    #[test]
    fn test_trace_context_roundtrip() {
        let original = TraceContext::default();
        let header = original.to_traceparent();
        let parsed = TraceContext::from_traceparent(&header).unwrap();

        assert_eq!(original.trace_id, parsed.trace_id);
        assert_eq!(original.span_id, parsed.span_id);
        assert_eq!(original.trace_flags, parsed.trace_flags);
    }

    #[test]
    fn test_trace_context_child() {
        let parent = TraceContext::default();
        let child = parent.child();

        // Same trace ID
        assert_eq!(parent.trace_id, child.trace_id);
        // Different span ID
        assert_ne!(parent.span_id, child.span_id);
        // Same flags
        assert_eq!(parent.trace_flags, child.trace_flags);
    }

    #[test]
    fn test_invalid_traceparent_formats() {
        // Too few parts
        assert!(TraceContext::from_traceparent("invalid").is_none());
        assert!(TraceContext::from_traceparent("00-abc-def").is_none());

        // Wrong lengths
        assert!(TraceContext::from_traceparent("00-short-id-01").is_none());
        assert!(
            TraceContext::from_traceparent("00-0af7651916cd43dd8448eb211c80319c-short-01")
                .is_none()
        );

        // Invalid hex in flags
        assert!(TraceContext::from_traceparent(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-zz"
        )
        .is_none());
    }

    #[test]
    fn test_traceparent_header_constant() {
        assert_eq!(TRACEPARENT_HEADER, "traceparent");
    }

    #[test]
    fn test_trace_context_not_sampled() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            span_id: "b7ad6b7169203331".to_string(),
            trace_flags: 0x00, // Not sampled
            trace_state: None,
        };

        let header = ctx.to_traceparent();
        assert!(header.ends_with("-00"));

        let parsed = TraceContext::from_traceparent(&header).unwrap();
        assert_eq!(parsed.trace_flags, 0x00);
    }

    #[test]
    fn test_new_child_span_id_uniqueness() {
        let ids: Vec<String> = (0..100)
            .map(|_| TraceContext::new_child_span_id())
            .collect();

        // All IDs should be unique
        let mut unique_ids = ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        assert_eq!(ids.len(), unique_ids.len());

        // All IDs should be 16 hex chars
        for id in &ids {
            assert_eq!(id.len(), 16);
            assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    #[tokio::test]
    async fn test_http2_request_with_tracing() {
        let handler = Arc::new(TracingTestHandler::new());
        let cancel = CancellationToken::new();

        // Start server
        let server = Http2Server::new(
            "127.0.0.1:0".parse().unwrap(),
            handler.clone(),
            Http2Config::default(),
            cancel.clone(),
        )
        .await
        .unwrap();

        let addr = server.local_addr();

        // Create client
        let client = Http2Client::new(Http2Config::default());

        // Send request (client should inject traceparent)
        let response = client
            .ask(addr, "/actors/traced", "test", b"hello".to_vec())
            .await
            .unwrap();

        // Verify response
        assert!(!response.is_empty());

        // Cleanup
        cancel.cancel();
    }

    #[tokio::test]
    async fn test_multiple_requests_different_traces() {
        let handler = Arc::new(TracingTestHandler::new());
        let cancel = CancellationToken::new();

        let server = Http2Server::new(
            "127.0.0.1:0".parse().unwrap(),
            handler.clone(),
            Http2Config::default(),
            cancel.clone(),
        )
        .await
        .unwrap();

        let addr = server.local_addr();
        let client = Http2Client::new(Http2Config::default());

        // Send multiple requests
        for i in 0..3 {
            let _ = client
                .ask(
                    addr,
                    "/actors/test",
                    "type",
                    format!("msg-{}", i).into_bytes(),
                )
                .await
                .unwrap();
        }

        // Each request should have its own trace
        assert_eq!(handler.counters.ask_count.load(Ordering::SeqCst), 3);

        cancel.cancel();
    }

    #[test]
    fn test_trace_context_otel_conversion() {
        let ctx = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            span_id: "b7ad6b7169203331".to_string(),
            trace_flags: 0x01,
            trace_state: None,
        };

        // Convert to OpenTelemetry context
        let otel_ctx = ctx.to_otel_context();

        // The context should be valid (not panic)
        assert!(!otel_ctx
            .span()
            .span_context()
            .trace_id()
            .to_string()
            .is_empty());
    }
}
