//! Static route seeding tests
//!
//! Tests for `ActorSystem::seed_static_route` / `GossipCluster::seed_static_route`:
//! registering a deterministic path->addr route that bypasses gossip convergence.

use pulsing_actor::actor::ActorPath;
use pulsing_actor::prelude::*;

// ============================================================================
// Error paths
// ============================================================================

#[tokio::test]
async fn test_seed_static_route_rejects_bad_addr() {
    let system = ActorSystem::new(SystemConfig::standalone()).await.unwrap();

    let err = system
        .seed_static_route("shops/worker_1", "not-a-socket-addr")
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("bad static addr"));

    system.shutdown().await.unwrap();
}

// ============================================================================
// Registration & resolution (gossip backend, as created by builder().addr())
// ============================================================================

async fn mesh_system() -> std::sync::Arc<ActorSystem> {
    ActorSystem::builder()
        .addr("127.0.0.1:0")
        .build()
        .await
        .unwrap()
}

#[tokio::test]
async fn test_seed_static_route_registers_named_instance() {
    let system = mesh_system().await;

    // A node at an addr that never joins gossip: without seeding, lookup misses.
    let path = ActorPath::new("shops/remote_worker").unwrap();
    assert!(system.lookup_named(&path).await.is_none());

    system
        .seed_static_route("shops/remote_worker", "127.0.0.1:19001")
        .await
        .unwrap();

    // After seeding the named actor resolves to an instance on the synthetic node.
    let info = system.lookup_named(&path).await;
    assert!(
        info.is_some(),
        "seeded route must register a named instance"
    );
    let info = info.unwrap();
    assert_eq!(info.path, path);
    assert!(!info.instance_nodes.is_empty());

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_seed_static_route_enables_resolve() {
    let system = mesh_system().await;

    system
        .seed_static_route("shops/remote_worker", "127.0.0.1:19002")
        .await
        .unwrap();

    // resolve() must succeed without any gossip round-trip: this is the whole
    // point of seeding (pre-fix this fails with "No healthy instances").
    let r = system.resolve_named("shops/remote_worker", None).await;
    assert!(
        r.is_ok(),
        "resolve after seeding must succeed, got: {:?}",
        r.err()
    );

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_seed_static_route_is_idempotent() {
    let system = mesh_system().await;

    system
        .seed_static_route("shops/remote_worker", "127.0.0.1:19003")
        .await
        .unwrap();
    // Re-seeding the same route (refresh path) must not error and must keep
    // the actor resolvable.
    system
        .seed_static_route("shops/remote_worker", "127.0.0.1:19003")
        .await
        .unwrap();
    assert!(system
        .resolve_named("shops/remote_worker", None)
        .await
        .is_ok());

    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_seed_static_route_multiple_routes_same_addr() {
    let system = mesh_system().await;

    // Several names on one node (the 10k-shard shape: N shops on one addr).
    for i in 1..=5 {
        system
            .seed_static_route(&format!("shops/worker_{i}"), "127.0.0.1:19004")
            .await
            .unwrap();
    }
    for i in 1..=5 {
        assert!(system
            .resolve_named(format!("shops/worker_{i}"), None)
            .await
            .is_ok());
    }

    system.shutdown().await.unwrap();
}
