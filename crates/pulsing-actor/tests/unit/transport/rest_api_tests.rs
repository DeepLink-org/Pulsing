//! REST API endpoint tests

use crate::common::fixtures::TestHandler;
use pulsing_actor::transport::{Http2Config, Http2Server, Http2ServerHandler};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Test /cluster/members endpoint
#[tokio::test]
async fn test_cluster_members_endpoint() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let _server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler.clone(),
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let members = handler.cluster_members().await;
    assert!(members.is_array());
    let members_array = members.as_array().unwrap();
    assert_eq!(members_array.len(), 2);

    let member = &members_array[0];
    assert!(member.get("node_id").is_some());
    assert!(member.get("addr").is_some());
    assert!(member.get("status").is_some());

    cancel.cancel();
}

/// Test /actors endpoint
#[tokio::test]
async fn test_actors_list_endpoint() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let _server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler.clone(),
        Http2Config::default(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let actors = handler.actors_list(false).await;
    assert!(actors.is_array());
    let actors_array = actors.as_array().unwrap();
    assert_eq!(actors_array.len(), 2);

    let actor = &actors_array[0];
    assert!(actor.get("name").is_some());
    assert!(actor.get("type").is_some());
    assert_eq!(actor.get("type").unwrap(), "user");

    let all_actors = handler.actors_list(true).await;
    let all_actors_array = all_actors.as_array().unwrap();
    assert_eq!(all_actors_array.len(), 3);

    cancel.cancel();
}

/// Test actor metadata in actors list
#[tokio::test]
async fn test_actors_list_metadata() {
    let handler = Arc::new(TestHandler::new());

    let actors = handler.actors_list(false).await;
    let actors_array = actors.as_array().unwrap();
    let actor = &actors_array[0];

    assert!(actor.get("actor_id").is_some());
    assert!(actor.get("class").is_some());
    assert!(actor.get("module").is_some());

    assert_eq!(actor.get("class").unwrap(), "Counter");
    assert_eq!(actor.get("module").unwrap(), "__main__");
}

/// Test health check endpoint returns expected structure
#[tokio::test]
async fn test_health_check_endpoint() {
    let handler = Arc::new(TestHandler::new());

    let health = handler.health_check().await;

    assert!(health.get("status").is_some());
    assert_eq!(health.get("status").unwrap(), "healthy");
    assert!(health.get("ask_count").is_some());
    assert!(health.get("tell_count").is_some());
}
