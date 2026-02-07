//! HTTP/2 Server tests

use crate::common::fixtures::TestHandler;
use pulsing_actor::transport::{Http2Config, Http2Server};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

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
