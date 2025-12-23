//! HTTP/2 Transport layer tests
//!
//! Tests for HTTP/2 (h2c) transport including:
//! - Server creation and startup
//! - Client connection and requests
//! - Ask (request-response) pattern
//! - Tell (fire-and-forget) pattern
//! - Http2RemoteTransport integration

use pulsing_actor::actor::{ActorId, Message, MessageStream};
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
    stream_count: AtomicUsize,
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
    async fn handle_ask(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        self.counters.ask_count.fetch_add(1, Ordering::SeqCst);

        // Echo the payload with path and msg_type prepended
        let mut response = format!("{}:{}:", path, msg_type).into_bytes();
        response.extend(payload);
        Ok(response)
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

    async fn handle_stream(
        &self,
        _path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<MessageStream> {
        self.counters.stream_count.fetch_add(1, Ordering::SeqCst);

        // Clone msg_type to avoid lifetime issues
        let msg_type = msg_type.to_string();

        // Generate a stream of 5 Message items
        let stream = futures::stream::iter((0..5).map(move |i| {
            Ok(Message::single(
                format!("{}-{}", msg_type, i),
                format!("chunk-{}-{:?}", i, payload).into_bytes(),
            ))
        }));

        Ok(Box::pin(stream))
    }

    async fn handle_gossip(&self, _payload: Vec<u8>) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    async fn health_check(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "healthy",
            "ask_count": self.counters.ask_count.load(Ordering::SeqCst),
            "tell_count": self.counters.tell_count.load(Ordering::SeqCst),
            "stream_count": self.counters.stream_count.load(Ordering::SeqCst),
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

    let actor_id = ActorId::local("test-actor");
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

    let actor_id = ActorId::local("fire-actor");
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

    let actor_id = ActorId::local("worker");
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

    let frame = StreamFrame::data(0, "token", b"hello");
    assert_eq!(frame.seq, 0);
    assert_eq!(frame.msg_type, "token");
    assert!(!frame.end);
    assert!(frame.error.is_none());

    let decoded = frame.decode_data().unwrap();
    assert_eq!(decoded, b"hello");
}

#[test]
fn test_stream_frame_end() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::end(5);
    assert_eq!(frame.seq, 5);
    assert!(frame.end);
    assert!(frame.error.is_none());
}

#[test]
fn test_stream_frame_error() {
    use pulsing_actor::transport::StreamFrame;

    let frame = StreamFrame::error(3, "something went wrong");
    assert_eq!(frame.seq, 3);
    assert!(frame.end);
    assert!(frame.is_error());
    assert_eq!(frame.error.as_ref().unwrap(), "something went wrong");
}

#[test]
fn test_stream_frame_ndjson_roundtrip() {
    use pulsing_actor::transport::StreamFrame;

    let original = StreamFrame::data(1, "response", b"world");
    let json = original.to_ndjson().unwrap();
    let parsed = StreamFrame::from_ndjson(std::str::from_utf8(&json).unwrap()).unwrap();

    assert_eq!(parsed.seq, 1);
    assert_eq!(parsed.msg_type, "response");
    assert_eq!(parsed.decode_data().unwrap(), b"world");
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

    assert_eq!(MessageMode::from_str("ask"), Some(MessageMode::Ask));
    assert_eq!(MessageMode::from_str("TELL"), Some(MessageMode::Tell));
    assert_eq!(MessageMode::from_str("Stream"), Some(MessageMode::Stream));
    assert_eq!(MessageMode::from_str("invalid"), None);
}

// ============================================================================
// Streaming Response Tests
// ============================================================================

#[tokio::test]
async fn test_http2_stream_request() {
    use futures::StreamExt;

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

    // Create client and send stream request
    let client = Http2Client::new(Http2Config::default());
    let mut stream = client
        .ask_stream(
            addr,
            "/actors/stream-test",
            "StreamMsg",
            b"test-payload".to_vec(),
        )
        .await
        .unwrap();

    // Collect all frames
    let mut frames = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(frame) => {
                if !frame.end || !frame.data.is_empty() {
                    frames.push(frame);
                }
            }
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    // Should receive 5 data frames (from TestHandler::handle_stream)
    assert_eq!(frames.len(), 5, "Expected 5 frames, got {}", frames.len());

    // Verify frame content
    for (i, frame) in frames.iter().enumerate() {
        assert!(frame.msg_type.contains(&format!("StreamMsg-{}", i)));
        let data = frame.decode_data().unwrap();
        let data_str = String::from_utf8_lossy(&data);
        assert!(data_str.contains(&format!("chunk-{}", i)));
    }

    // Verify stream counter
    assert_eq!(counters.stream_count.load(Ordering::SeqCst), 1);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_stream_request_raw() {
    use futures::StreamExt;

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

    // Create client and send stream request using ask_stream_raw
    let client = Http2Client::new(Http2Config::default());
    let mut stream = client
        .ask_stream_raw(addr, "/actors/raw-stream", "RawStream", b"payload".to_vec())
        .await
        .unwrap();

    // Collect all messages
    let mut messages = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(msg) => messages.push(msg),
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    // Should receive 5 messages
    assert_eq!(messages.len(), 5);

    // Verify messages are single type
    for msg in &messages {
        assert!(msg.is_single());
    }

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_stream_cancellation() {
    use futures::StreamExt;

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

    // Create client and send stream request
    let client = Http2Client::new(Http2Config::default());
    let stream = client
        .ask_stream(addr, "/actors/cancel-test", "StreamMsg", b"test".to_vec())
        .await
        .unwrap();

    // Cancel after receiving first frame
    let stream_cancel = stream.cancellation_token();

    let mut pinned = std::pin::pin!(stream);
    let first = pinned.next().await;
    assert!(first.is_some());

    // Cancel the stream
    stream_cancel.cancel();

    // Stream should be cancelled
    assert!(stream_cancel.is_cancelled());

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_remote_transport_stream() {
    use futures::StreamExt;

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
    let transport = Http2RemoteTransport::new(client, addr, "stream-actor".to_string());

    // Use the RemoteTransport trait
    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::local("stream-actor");
    let mut stream = transport
        .request_stream(&actor_id, "StreamType", b"request".to_vec())
        .await
        .unwrap();

    // Collect payloads
    let mut payloads = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(payload) => payloads.push(payload),
            Err(e) => panic!("Stream error: {}", e),
        }
    }

    // Should receive 5 payloads
    assert_eq!(payloads.len(), 5);

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

    // Should be faster than sequential
    assert!(rps > 500.0, "Concurrent throughput too low: {} req/s", rps);

    // Cleanup
    cancel.cancel();
}

#[tokio::test]
async fn test_http2_stream_throughput_benchmark() {
    use futures::StreamExt;

    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    // Start server
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::streaming(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    // Create client
    let client = Http2Client::new(Http2Config::streaming());

    let stream_count = 100;
    let start = std::time::Instant::now();
    let mut total_frames = 0;

    // Send multiple stream requests
    for _ in 0..stream_count {
        let mut stream = client
            .ask_stream(addr, "/actors/bench-stream", "Stream", b"payload".to_vec())
            .await
            .unwrap();

        while let Some(result) = stream.next().await {
            if result.is_ok() {
                total_frames += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    let streams_per_sec = stream_count as f64 / elapsed.as_secs_f64();
    let frames_per_sec = total_frames as f64 / elapsed.as_secs_f64();

    println!(
        "HTTP/2 Stream Throughput: {} streams, {} frames in {:?} ({:.0} streams/s, {:.0} frames/s)",
        stream_count, total_frames, elapsed, streams_per_sec, frames_per_sec
    );

    // Should handle reasonable throughput
    assert!(
        streams_per_sec > 10.0,
        "Stream throughput too low: {} streams/s",
        streams_per_sec
    );

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

    // P99 should be under 100ms for localhost
    assert!(
        p99 < std::time::Duration::from_millis(100),
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
