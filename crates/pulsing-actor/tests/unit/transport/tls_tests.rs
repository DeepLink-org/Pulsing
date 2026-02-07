//! TLS tests (requires `tls` feature)

use crate::common::fixtures::TestHandler;
use pulsing_actor::transport::http2::TlsConfig;
use pulsing_actor::transport::{Http2Client, Http2Config, Http2Server};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Test TLS configuration creation from passphrase
#[test]
fn test_tls_config_from_passphrase() {
    let config = TlsConfig::from_passphrase("test-cluster-password");
    assert!(config.is_ok(), "TLS config creation failed: {:?}", config);
}

/// Test that same passphrase produces deterministic CA
#[test]
fn test_tls_deterministic_ca() {
    let config1 = TlsConfig::from_passphrase("deterministic-test-password").unwrap();
    let config2 = TlsConfig::from_passphrase("deterministic-test-password").unwrap();

    assert_eq!(config1.passphrase_hash(), config2.passphrase_hash());
}

/// Test that different passphrase produces different CA
#[test]
fn test_tls_different_passphrase() {
    let config1 = TlsConfig::from_passphrase("password-one").unwrap();
    let config2 = TlsConfig::from_passphrase("password-two").unwrap();

    assert_ne!(config1.passphrase_hash(), config2.passphrase_hash());
}

/// Test TLS-enabled HTTP/2 server and client communication
#[tokio::test]
async fn test_tls_server_client_communication() {
    let tls_config = TlsConfig::from_passphrase("test-cluster-tls").unwrap();

    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let http2_config = Http2Config::default().tls_config(tls_config.clone());

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler.clone(),
        http2_config.clone(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    let client = Http2Client::new(http2_config);

    let response = client
        .ask(addr, "/actors/test", "test-msg", b"hello tls".to_vec())
        .await;

    assert!(response.is_ok(), "TLS request failed: {:?}", response);

    let body = response.unwrap();
    let response_str = String::from_utf8_lossy(&body);
    assert!(
        response_str.contains("hello tls"),
        "Response should contain original message"
    );

    cancel.cancel();
}

/// Test that different passphrase fails TLS handshake
#[tokio::test]
async fn test_tls_different_passphrase_fails() {
    let server_tls = TlsConfig::from_passphrase("server-password").unwrap();
    let client_tls = TlsConfig::from_passphrase("wrong-password").unwrap();

    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server_config = Http2Config::default().tls_config(server_tls);
    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        server_config,
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();

    let client_config = Http2Config::default().tls_config(client_tls);
    let client = Http2Client::new(client_config);

    let response = client
        .ask(addr, "/actors/test", "test", b"test".to_vec())
        .await;

    assert!(
        response.is_err(),
        "Request with different passphrase should fail"
    );

    cancel.cancel();
}
