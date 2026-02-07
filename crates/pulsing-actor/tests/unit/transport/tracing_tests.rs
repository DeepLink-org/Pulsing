//! Tracing and trace context tests

use crate::common::fixtures::TestCounters;
use pulsing_actor::actor::Message;
use pulsing_actor::tracing::opentelemetry::trace::TraceContextExt;
use pulsing_actor::tracing::{TraceContext, TRACEPARENT_HEADER};
use pulsing_actor::transport::{Http2Client, Http2Config, Http2Server, Http2ServerHandler};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

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
    ) -> pulsing_actor::error::Result<Message> {
        self.counters.ask_count.fetch_add(1, Ordering::SeqCst);

        let trace_ctx = TraceContext::from_current();
        self.captured_traces
            .lock()
            .unwrap()
            .push(trace_ctx.map(|t| t.to_traceparent()));

        let response = format!("{}:{}:{}", path, msg_type, payload.len());
        Ok(Message::single("traced", response.into_bytes()))
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
        TraceContext::from_traceparent("00-0af7651916cd43dd8448eb211c80319c-short-01").is_none()
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

    let response = client
        .ask(addr, "/actors/traced", "test", b"hello".to_vec())
        .await
        .unwrap();

    assert!(!response.is_empty());

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

    let otel_ctx = ctx.to_otel_context();

    assert!(!otel_ctx
        .span()
        .span_context()
        .trace_id()
        .to_string()
        .is_empty());
}
