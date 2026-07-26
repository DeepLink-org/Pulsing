//! Narrow host capabilities granted to system services.

use super::lifecycle::{NodeLifecycle, NodeState};
use super::{ActorInfo, ActorRegistry, ShmManager, SystemMetrics, SystemRef, SYSTEM_ACTOR_PATH};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::performance_store::PerformanceStore;
use crate::system::ActorSystem;
use async_trait::async_trait;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};
use std::time::Instant;

/// Actor operations available to the node control plane.
#[async_trait]
pub(crate) trait ActorControl: Send + Sync {
    fn list(&self) -> Vec<ActorInfo>;
    fn get(&self, name: &str) -> Option<ActorInfo>;
    fn count(&self) -> usize;
    fn total_messages(&self) -> u64;
    async fn stop(&self, name: &str) -> Result<bool>;
}

/// Explicit capability view passed to system services.
///
/// It contains no transport or cluster implementation details. Services can
/// only perform the node operations declared here.
pub(crate) struct SystemHost {
    node_id: crate::actor::NodeId,
    addr: std::net::SocketAddr,
    start_time: Instant,
    actors: Arc<dyn ActorControl>,
    metrics: Arc<SystemMetrics>,
    performance_store: Arc<PerformanceStore>,
    shm_manager: Arc<ShmManager>,
    lifecycle: Arc<NodeLifecycle>,
}

impl SystemHost {
    pub(crate) fn standalone(
        system_ref: &SystemRef,
        registry: Arc<ActorRegistry>,
        metrics: Arc<SystemMetrics>,
        performance_store: Arc<PerformanceStore>,
    ) -> Self {
        let actors = Arc::new(LegacyActorControl {
            registry,
            metrics: metrics.clone(),
        });
        Self {
            node_id: system_ref.node_id,
            addr: system_ref.addr,
            start_time: Instant::now(),
            actors,
            metrics,
            performance_store,
            shm_manager: Arc::new(ShmManager::new()),
            lifecycle: Arc::new(NodeLifecycle::new()),
        }
    }

    pub(crate) fn hosted(
        system_ref: &SystemRef,
        system: Weak<ActorSystem>,
        metrics: Arc<SystemMetrics>,
        performance_store: Arc<PerformanceStore>,
        shm_manager: Arc<ShmManager>,
        lifecycle: Arc<NodeLifecycle>,
    ) -> Self {
        Self {
            node_id: system_ref.node_id,
            addr: system_ref.addr,
            start_time: Instant::now(),
            actors: Arc::new(HostedActorControl { system }),
            metrics,
            performance_store,
            shm_manager,
            lifecycle,
        }
    }

    pub(crate) fn with_actor_control(mut self, actors: Arc<dyn ActorControl>) -> Self {
        self.actors = actors;
        self
    }

    pub(crate) fn with_shm_manager(mut self, manager: Arc<ShmManager>) -> Self {
        self.shm_manager = manager;
        self
    }

    pub(crate) fn node_state(&self) -> NodeState {
        self.lifecycle.state()
    }

    pub(crate) fn node_id(&self) -> crate::actor::NodeId {
        self.node_id
    }

    pub(crate) fn addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    pub(crate) fn start_time(&self) -> Instant {
        self.start_time
    }

    pub(crate) fn actors(&self) -> Arc<dyn ActorControl> {
        self.actors.clone()
    }

    pub(crate) fn metrics(&self) -> Arc<SystemMetrics> {
        self.metrics.clone()
    }

    pub(crate) fn performance_store(&self) -> Arc<PerformanceStore> {
        self.performance_store.clone()
    }

    pub(crate) fn shm_manager(&self) -> &Arc<ShmManager> {
        &self.shm_manager
    }

    pub(crate) fn lifecycle(&self) -> Arc<NodeLifecycle> {
        self.lifecycle.clone()
    }
}

struct LegacyActorControl {
    registry: Arc<ActorRegistry>,
    metrics: Arc<SystemMetrics>,
}

#[async_trait]
impl ActorControl for LegacyActorControl {
    fn list(&self) -> Vec<ActorInfo> {
        self.registry.list_all()
    }

    fn get(&self, name: &str) -> Option<ActorInfo> {
        self.registry.get_info(name)
    }

    fn count(&self) -> usize {
        self.registry.count()
    }

    fn total_messages(&self) -> u64 {
        self.metrics.messages_total()
    }

    async fn stop(&self, name: &str) -> Result<bool> {
        Ok(self.registry.unregister(name).is_some())
    }
}

pub(crate) struct RegistryActorControl {
    registry: Arc<crate::system::registry::ActorRegistry>,
}

impl RegistryActorControl {
    pub(crate) fn new(registry: Arc<crate::system::registry::ActorRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl ActorControl for RegistryActorControl {
    fn list(&self) -> Vec<ActorInfo> {
        actor_infos(&self.registry)
    }

    fn get(&self, name: &str) -> Option<ActorInfo> {
        actor_info(&self.registry, name)
    }

    fn count(&self) -> usize {
        actor_count(&self.registry)
    }

    fn total_messages(&self) -> u64 {
        total_messages(&self.registry)
    }

    async fn stop(&self, _name: &str) -> Result<bool> {
        Err(host_error(
            "actor stop requires an ActorSystem-owned lifecycle capability",
        ))
    }
}

struct HostedActorControl {
    system: Weak<ActorSystem>,
}

#[async_trait]
impl ActorControl for HostedActorControl {
    fn list(&self) -> Vec<ActorInfo> {
        self.system
            .upgrade()
            .map(|system| actor_infos(&system.registry))
            .unwrap_or_default()
    }

    fn get(&self, name: &str) -> Option<ActorInfo> {
        self.system
            .upgrade()
            .and_then(|system| actor_info(&system.registry, name))
    }

    fn count(&self) -> usize {
        self.system
            .upgrade()
            .map(|system| actor_count(&system.registry))
            .unwrap_or_default()
    }

    fn total_messages(&self) -> u64 {
        self.system
            .upgrade()
            .map(|system| total_messages(&system.registry))
            .unwrap_or_default()
    }

    async fn stop(&self, name: &str) -> Result<bool> {
        if name == SYSTEM_ACTOR_PATH {
            return Ok(false);
        }
        let system = self
            .system
            .upgrade()
            .ok_or_else(|| host_error("ActorSystem is no longer available"))?;
        let resolved_name = if system.registry.has_name(name) {
            Some(name.to_string())
        } else if !name.contains('/') {
            let prefixed = format!("actors/{name}");
            system.registry.has_name(&prefixed).then_some(prefixed)
        } else {
            None
        };
        let Some(resolved_name) = resolved_name else {
            return Ok(false);
        };
        system.stop(&resolved_name).await?;
        Ok(true)
    }
}

fn actor_infos(registry: &crate::system::registry::ActorRegistry) -> Vec<ActorInfo> {
    registry
        .actor_names
        .iter()
        .filter_map(|entry| actor_info(registry, entry.key()))
        .collect()
}

fn actor_info(registry: &crate::system::registry::ActorRegistry, name: &str) -> Option<ActorInfo> {
    if name == SYSTEM_ACTOR_PATH {
        return None;
    }
    let actor_id = registry.get_actor_id(name)?;
    let handle = registry.get_handle(&actor_id)?;
    let actor_type = handle
        .metadata
        .get("class")
        .or_else(|| handle.metadata.get("type"))
        .cloned()
        .unwrap_or_else(|| "Actor".to_string());
    Some(ActorInfo {
        name: name.to_string(),
        actor_id: handle.actor_id.0,
        actor_type,
        uptime_secs: handle.started_at.elapsed().as_secs(),
        metadata: handle.metadata.clone(),
    })
}

fn actor_count(registry: &crate::system::registry::ActorRegistry) -> usize {
    registry
        .actor_names
        .iter()
        .filter(|entry| entry.key().as_str() != SYSTEM_ACTOR_PATH)
        .count()
}

fn total_messages(registry: &crate::system::registry::ActorRegistry) -> u64 {
    registry
        .iter_actors()
        .map(|entry| entry.value().stats.message_count.load(Ordering::Relaxed))
        .sum()
}

fn host_error(message: impl Into<String>) -> PulsingError {
    PulsingError::from(RuntimeError::Other(format!(
        "System host capability error: {}",
        message.into()
    )))
}
