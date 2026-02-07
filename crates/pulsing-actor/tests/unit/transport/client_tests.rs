//! HTTP/2 Client tests

use crate::common::fixtures::{StreamingHandler, TestCounters, TestHandler};
use pulsing_actor::actor::{ActorId, Message};
use pulsing_actor::transport::{Http2Client, Http2Config, Http2RemoteTransport, Http2Server};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

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

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    let client = Http2Client::new(Http2Config::default());
    let response = client
        .ask(addr, "/actors/test", "TestMsg", b"hello".to_vec())
        .await
        .unwrap();

    let response_str = String::from_utf8_lossy(&response);
    assert!(response_str.contains("/actors/test"));
    assert!(response_str.contains("TestMsg"));
    assert!(response_str.contains("hello"));

    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_tell_request() {
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

    let addr = server.local_addr();

    let client = Http2Client::new(Http2Config::default());
    client
        .tell(addr, "/actors/test", "FireAndForget", b"data".to_vec())
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    assert_eq!(counters.tell_count.load(Ordering::SeqCst), 1);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_multiple_requests() {
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

    let addr = server.local_addr();
    let client = Http2Client::new(Http2Config::default());

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

    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 10);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_concurrent_requests() {
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

    let addr = server.local_addr();
    let client = Arc::new(Http2Client::new(Http2Config::default()));

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

    let results: Vec<_> = futures::future::join_all(handles).await;

    for result in results {
        assert!(result.unwrap().is_ok());
    }

    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 20);

    cancel.cancel();
}

// ============================================================================
// Http2RemoteTransport Tests
// ============================================================================

#[tokio::test]
async fn test_http2_remote_transport_ask() {
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

    let client = Arc::new(Http2Client::new(Http2Config::default()));
    let transport = Http2RemoteTransport::new(client, addr, "test-actor".to_string());

    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::generate();
    let response = transport
        .request(&actor_id, "TestType", b"payload".to_vec())
        .await
        .unwrap();

    let response_str = String::from_utf8_lossy(&response);
    assert!(response_str.contains("test-actor"));
    assert!(response_str.contains("TestType"));

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_remote_transport_tell() {
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

    let addr = server.local_addr();

    let client = Arc::new(Http2Client::new(Http2Config::default()));
    let transport = Http2RemoteTransport::new(client, addr, "fire-actor".to_string());

    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::generate();
    transport
        .send(&actor_id, "FireMsg", b"data".to_vec())
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    assert_eq!(counters.tell_count.load(Ordering::SeqCst), 1);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_remote_transport_named_path() {
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

    let client = Arc::new(Http2Client::new(Http2Config::default()));
    use pulsing_actor::actor::ActorPath;
    let path = ActorPath::new("services/llm/worker").unwrap();
    let transport = Http2RemoteTransport::new_named(client, addr, path);

    use pulsing_actor::actor::RemoteTransport;

    let actor_id = ActorId::generate();
    let response = transport
        .request(&actor_id, "Inference", b"prompt".to_vec())
        .await
        .unwrap();

    let response_str = String::from_utf8_lossy(&response);
    assert!(response_str.contains("services/llm/worker"));

    cancel.cancel();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_http2_connection_refused() {
    let client = Http2Client::new(Http2Config::default());

    let result = client
        .ask(
            "127.0.0.1:1".parse().unwrap(),
            "/actors/test",
            "Test",
            vec![],
        )
        .await;

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

    let client = Http2Client::new(Http2Config::default());
    let result = client.ask(addr, "/actors/test", "Test", vec![]).await;
    assert!(result.is_ok());

    server.shutdown();

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
}

// ============================================================================
// Unified Send Message Tests
// ============================================================================

#[tokio::test]
async fn test_http2_unified_send_message() {
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

    let addr = server.local_addr();

    let client = Http2Client::new(Http2Config::default());
    let response = client
        .send_message(addr, "/actors/test", "TestMsg", b"test-payload".to_vec())
        .await
        .unwrap();

    assert!(response.is_single());

    let Message::Single { data, .. } = response else {
        panic!("Expected single message");
    };
    let data_str = String::from_utf8_lossy(&data);
    assert!(data_str.contains("/actors/test"));
    assert!(data_str.contains("TestMsg"));

    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    cancel.cancel();
}

// ============================================================================
// Connection Pool Tests
// ============================================================================

#[tokio::test]
async fn test_http2_connection_pool_reuse() {
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
    let client = Http2Client::new(Http2Config::default());

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

    let stats = client.stats();
    let created = stats.connections_created.load(Ordering::Relaxed);
    let reused = stats.connections_reused.load(Ordering::Relaxed);

    println!("Connection Pool: created={}, reused={}", created, reused);

    assert!(created <= 2, "Too many connections created: {}", created);
    assert!(reused >= 8, "Not enough connection reuse: {}", reused);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_retry_on_connection_error() {
    use pulsing_actor::transport::{Http2ClientBuilder, RetryConfig};
    use std::time::Duration;

    let client = Http2ClientBuilder::new()
        .retry_config(RetryConfig::with_max_retries(2).initial_delay(Duration::from_millis(10)))
        .connect_timeout(Duration::from_millis(100))
        .build();

    let result: Result<Vec<u8>, _> = client
        .ask(
            "127.0.0.1:1".parse().unwrap(),
            "/actors/test",
            "Msg",
            vec![],
        )
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
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

#[tokio::test]
async fn test_http2_streaming_request() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(StreamingHandler::with_counters(counters.clone()));
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
    let client = Http2Client::new(Http2Config::default());

    let (tx, rx) = tokio::sync::mpsc::channel::<pulsing_actor::error::Result<Message>>(10);

    tokio::spawn(async move {
        for i in 0..5 {
            let msg = Message::single("chunk", format!("data-{}", i).into_bytes());
            if tx.send(Ok(msg)).await.is_err() {
                break;
            }
        }
    });

    let stream_msg = Message::from_channel("StreamRequest", rx);

    let response = client
        .send_message_full(addr, "/actors/stream_test", stream_msg)
        .await
        .unwrap();

    assert!(response.is_single());
    let Message::Single { data, .. } = response else {
        panic!("Expected single message");
    };
    let response_str = String::from_utf8_lossy(&data);
    assert!(response_str.contains("/actors/stream_test"));
    assert!(response_str.contains("collected:"));

    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_single_request_with_full_api() {
    let counters = Arc::new(TestCounters::default());
    let handler = Arc::new(StreamingHandler::with_counters(counters.clone()));
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
    let client = Http2Client::new(Http2Config::default());

    let msg = Message::single("TestType", b"test-payload".to_vec());
    let response = client
        .send_message_full(addr, "/actors/single_test", msg)
        .await
        .unwrap();

    assert!(response.is_single());
    let Message::Single { data, .. } = response else {
        panic!("Expected single message");
    };
    let response_str = String::from_utf8_lossy(&data);
    assert!(response_str.contains("/actors/single_test"));
    assert!(response_str.contains("TestType"));

    assert_eq!(counters.ask_count.load(Ordering::SeqCst), 1);

    cancel.cancel();
}
