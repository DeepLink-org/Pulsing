//! SystemActor tests
//!
//! Tests for the built-in SystemActor functionality

use pulsing_actor::actor::ActorId;
use pulsing_actor::prelude::*;
use pulsing_actor::system_actor::{
    ActorInfo, ActorRegistry, SystemMessage, SystemMetrics, SystemResponse, SYSTEM_ACTOR_PATH,
};
use std::time::Duration;

// ============================================================================
// Helper functions
// ============================================================================

async fn create_test_system() -> std::sync::Arc<ActorSystem> {
    ActorSystem::new(SystemConfig::standalone()).await.unwrap()
}

fn parse_system_response(msg: Message) -> SystemResponse {
    match msg {
        Message::Single { data, .. } => serde_json::from_slice(&data).unwrap(),
        _ => panic!("Expected Single message"),
    }
}

fn create_system_message(msg: &SystemMessage) -> Message {
    let json = serde_json::to_vec(msg).unwrap();
    Message::Single {
        msg_type: "SystemMessage".to_string(),
        data: json,
    }
}

// ============================================================================
// SystemActor Path Tests
// ============================================================================

#[test]
fn test_system_actor_path_constant() {
    assert_eq!(SYSTEM_ACTOR_PATH, "system/core");
}

// ============================================================================
// SystemMessage Serialization Tests
// ============================================================================

#[test]
fn test_system_message_ping_serialization() {
    let msg = SystemMessage::Ping;
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("Ping"));

    let parsed: SystemMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, SystemMessage::Ping));
}

#[test]
fn test_system_message_list_actors_serialization() {
    let msg = SystemMessage::ListActors;
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("ListActors"));

    let parsed: SystemMessage = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, SystemMessage::ListActors));
}

#[test]
fn test_system_message_get_actor_serialization() {
    let msg = SystemMessage::GetActor {
        name: "test_actor".to_string(),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("GetActor"));
    assert!(json.contains("test_actor"));

    let parsed: SystemMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        SystemMessage::GetActor { name } => assert_eq!(name, "test_actor"),
        _ => panic!("Expected GetActor"),
    }
}

#[test]
fn test_system_message_create_actor_serialization() {
    let msg = SystemMessage::CreateActor {
        actor_type: "Counter".to_string(),
        name: "counter1".to_string(),
        params: serde_json::json!({"init_value": 10}),
        public: true,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("CreateActor"));
    assert!(json.contains("Counter"));
    assert!(json.contains("counter1"));

    let parsed: SystemMessage = serde_json::from_str(&json).unwrap();
    match parsed {
        SystemMessage::CreateActor {
            actor_type,
            name,
            params,
            public,
        } => {
            assert_eq!(actor_type, "Counter");
            assert_eq!(name, "counter1");
            assert_eq!(params["init_value"], 10);
            assert!(public);
        }
        _ => panic!("Expected CreateActor"),
    }
}

// ============================================================================
// SystemResponse Serialization Tests
// ============================================================================

#[test]
fn test_system_response_ok_serialization() {
    let resp = SystemResponse::Ok;
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("Ok"));

    let parsed: SystemResponse = serde_json::from_str(&json).unwrap();
    assert!(matches!(parsed, SystemResponse::Ok));
}

#[test]
fn test_system_response_error_serialization() {
    let resp = SystemResponse::Error {
        message: "test error".to_string(),
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("Error"));
    assert!(json.contains("test error"));

    let parsed: SystemResponse = serde_json::from_str(&json).unwrap();
    match parsed {
        SystemResponse::Error { message } => assert_eq!(message, "test error"),
        _ => panic!("Expected Error"),
    }
}

#[test]
fn test_system_response_pong_serialization() {
    let resp = SystemResponse::Pong {
        node_id: 12345,
        timestamp: 1234567890,
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("Pong"));
    assert!(json.contains("12345"));

    let parsed: SystemResponse = serde_json::from_str(&json).unwrap();
    match parsed {
        SystemResponse::Pong { node_id, timestamp } => {
            assert_eq!(node_id, 12345);
            assert_eq!(timestamp, 1234567890);
        }
        _ => panic!("Expected Pong"),
    }
}

#[test]
fn test_actor_info_serialization() {
    let info = ActorInfo {
        name: "test".to_string(),
        actor_id: 123,
        actor_type: "TestActor".to_string(),
        uptime_secs: 60,
        metadata: std::collections::HashMap::new(),
    };
    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("test"));
    assert!(json.contains("123"));
    assert!(json.contains("TestActor"));

    let parsed: ActorInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.name, "test");
    assert_eq!(parsed.actor_id, 123);
    assert_eq!(parsed.actor_type, "TestActor");
    assert_eq!(parsed.uptime_secs, 60);
}

// ============================================================================
// SystemActor Integration Tests
// ============================================================================

#[tokio::test]
async fn test_system_actor_auto_start() {
    let system = create_test_system().await;

    // SystemActor should be automatically started
    let sys_ref = system.system().await.unwrap();
    assert!(sys_ref.is_local());

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_ping() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    let msg = create_system_message(&SystemMessage::Ping);
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::Pong { node_id, timestamp } => {
            assert_eq!(node_id, system.node_id().0);
            assert!(timestamp > 0);
        }
        _ => panic!("Expected Pong response"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_health_check() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    let msg = create_system_message(&SystemMessage::HealthCheck);
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::Health {
            status,
            actors_count: _,
            uptime_secs: _,
        } => {
            assert_eq!(status, "healthy");
        }
        _ => panic!("Expected Health response"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_get_node_info() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    let msg = create_system_message(&SystemMessage::GetNodeInfo);
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::NodeInfo {
            node_id,
            addr,
            uptime_secs: _,
        } => {
            assert_eq!(node_id, system.node_id().0);
            assert!(!addr.is_empty());
        }
        _ => panic!("Expected NodeInfo response"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_get_metrics() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    // Send a few messages first
    for _ in 0..3 {
        let msg = create_system_message(&SystemMessage::Ping);
        let _ = sys_ref.send(msg).await.unwrap();
    }

    let msg = create_system_message(&SystemMessage::GetMetrics);
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::Metrics {
            actors_count: _,
            messages_total,
            actors_created: _,
            actors_stopped: _,
            uptime_secs: _,
        } => {
            assert!(messages_total >= 4); // 3 pings + 1 get_metrics
        }
        _ => panic!("Expected Metrics response"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_list_actors() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    let msg = create_system_message(&SystemMessage::ListActors);
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::ActorList { actors } => {
            // Initially empty (SystemActor doesn't register itself in the registry)
            assert!(actors.is_empty() || actors.iter().all(|a| a.name != "system/core"));
        }
        _ => panic!("Expected ActorList response"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_get_actor_not_found() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    let msg = create_system_message(&SystemMessage::GetActor {
        name: "nonexistent".to_string(),
    });
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::Error { message } => {
            assert!(message.contains("not found"));
        }
        _ => panic!("Expected Error response"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_create_actor_not_supported() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    // CreateActor should return error in pure Rust mode
    let msg = create_system_message(&SystemMessage::CreateActor {
        actor_type: "Counter".to_string(),
        name: "test".to_string(),
        params: serde_json::Value::Null,
        public: true,
    });
    let resp = sys_ref.send(msg).await.unwrap();
    let parsed = parse_system_response(resp);

    match parsed {
        SystemResponse::Error { message } => {
            assert!(message.contains("not supported"));
        }
        _ => panic!("Expected Error response for CreateActor"),
    }

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_system_actor_multiple_requests() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    // Send multiple concurrent requests
    let mut handles = vec![];
    for _ in 0..10 {
        let ref_clone = sys_ref.clone();
        handles.push(tokio::spawn(async move {
            let msg = create_system_message(&SystemMessage::Ping);
            ref_clone.send(msg).await
        }));
    }

    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        let parsed = parse_system_response(result.unwrap());
        assert!(matches!(parsed, SystemResponse::Pong { .. }));
    }

    system.shutdown().await.unwrap();
}

// ============================================================================
// SystemActor with Uptime Tests
// ============================================================================

#[tokio::test]
async fn test_system_actor_uptime_increases() {
    let system = create_test_system().await;
    let sys_ref = system.system().await.unwrap();

    // Get initial uptime
    let msg = create_system_message(&SystemMessage::GetNodeInfo);
    let resp = sys_ref.send(msg).await.unwrap();
    let initial_uptime = match parse_system_response(resp) {
        SystemResponse::NodeInfo { uptime_secs, .. } => uptime_secs,
        _ => panic!("Expected NodeInfo"),
    };

    // Wait a bit
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Get uptime again
    let msg = create_system_message(&SystemMessage::GetNodeInfo);
    let resp = sys_ref.send(msg).await.unwrap();
    let later_uptime = match parse_system_response(resp) {
        SystemResponse::NodeInfo { uptime_secs, .. } => uptime_secs,
        _ => panic!("Expected NodeInfo"),
    };

    assert!(later_uptime >= initial_uptime);

    system.shutdown().await.unwrap();
}

// ============================================================================
// ActorRegistry Tests (moved from src/system_actor/mod.rs)
// ============================================================================

#[test]
fn test_actor_registry() {
    let registry = ActorRegistry::new();
    let actor_id = ActorId::generate();

    registry.register("test", actor_id, "TestActor");
    assert!(registry.contains("test"));
    assert_eq!(registry.count(), 1);

    let info = registry.get_info("test").unwrap();
    assert_eq!(info.name, "test");
    assert_eq!(info.actor_type, "TestActor");

    registry.unregister("test");
    assert!(!registry.contains("test"));
}

#[test]
fn test_actor_registry_list_all() {
    let registry = ActorRegistry::new();

    registry.register("actor1", ActorId::generate(), "TypeA");
    registry.register("actor2", ActorId::generate(), "TypeB");

    let actors = registry.list_all();
    assert_eq!(actors.len(), 2);
}

#[test]
fn test_actor_registry_get_not_found() {
    let registry = ActorRegistry::new();
    assert!(registry.get("nonexistent").is_none());
    assert!(registry.get_info("nonexistent").is_none());
}

// ============================================================================
// SystemMetrics Tests (moved from src/system_actor/mod.rs)
// ============================================================================

#[test]
fn test_system_metrics() {
    let metrics = SystemMetrics::new();

    metrics.inc_message();
    metrics.inc_message();
    assert_eq!(metrics.messages_total(), 2);

    metrics.inc_actor_created();
    assert_eq!(metrics.actors_created(), 1);

    metrics.inc_actor_stopped();
    assert_eq!(metrics.actors_stopped(), 1);
}

#[test]
fn test_system_metrics_concurrent() {
    use std::sync::Arc;
    use std::thread;

    let metrics = Arc::new(SystemMetrics::new());
    let mut handles = vec![];

    for _ in 0..10 {
        let m = metrics.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                m.inc_message();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(metrics.messages_total(), 1000);
}
