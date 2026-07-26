//! Built-in system-service registrations.

use super::host::{ActorControl, SystemHost};
use super::lifecycle::{NodeLifecycle, NodeState};
use super::service::{
    service_error, ActorsCommand, MetricsCommand, OperationAccess, OperationManifest,
    RuntimeCommand, ShmCommand, StatelessComponent, SystemCommand, SystemComponent,
    SystemRequestHandler, SystemServiceExposure, SystemServiceKind, SystemServiceManifest,
    SystemServiceRegistration, ACTORS_SERVICE, METRICS_SERVICE, RUNTIME_SERVICE, SHM_SERVICE,
};
use super::{ActorFactory, ShmManager, SystemMetrics, SystemResponse};
use crate::error::Result;
use crate::performance_store::{PerformanceSnapshot, PerformanceStore};
use async_trait::async_trait;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const RUNTIME_OPERATIONS: &[OperationManifest] = &[
    operation("node_info", OperationAccess::Read, true),
    operation("health", OperationAccess::Read, true),
    operation("ping", OperationAccess::Read, true),
];
const ACTOR_OPERATIONS: &[OperationManifest] = &[
    operation("list", OperationAccess::Read, true),
    operation("get", OperationAccess::Read, true),
    operation("create", OperationAccess::Admin, false),
    operation("stop", OperationAccess::Operate, false),
    operation("extension", OperationAccess::Admin, false),
];
const METRICS_OPERATIONS: &[OperationManifest] =
    &[operation("snapshot", OperationAccess::Read, true)];
const SHM_OPERATIONS: &[OperationManifest] = &[operation("stats", OperationAccess::Read, true)];

const fn operation(
    name: &'static str,
    access: OperationAccess,
    allowed_while_draining: bool,
) -> OperationManifest {
    OperationManifest {
        name,
        access,
        allowed_while_draining,
    }
}

pub(crate) fn registrations(
    host: &SystemHost,
    factory: Arc<dyn ActorFactory>,
) -> Vec<SystemServiceRegistration> {
    let stateless: Arc<dyn SystemComponent> = Arc::new(StatelessComponent);
    let shm = Arc::new(ShmService {
        manager: host.shm_manager().clone(),
    });
    vec![
        SystemServiceRegistration {
            manifest: core_manifest(
                RUNTIME_SERVICE,
                SystemServiceExposure::AuthenticatedRemote,
                RUNTIME_OPERATIONS,
            ),
            component: stateless.clone(),
            handler: Arc::new(RuntimeService {
                node_id: host.node_id(),
                addr: host.addr(),
                start_time: host.start_time(),
                actors: host.actors(),
                lifecycle: host.lifecycle(),
            }),
        },
        SystemServiceRegistration {
            manifest: core_manifest(
                ACTORS_SERVICE,
                SystemServiceExposure::AuthenticatedRemote,
                ACTOR_OPERATIONS,
            ),
            component: stateless.clone(),
            handler: Arc::new(ActorsService {
                actors: host.actors(),
                factory,
            }),
        },
        SystemServiceRegistration {
            manifest: core_manifest(
                METRICS_SERVICE,
                SystemServiceExposure::AuthenticatedRemote,
                METRICS_OPERATIONS,
            ),
            component: stateless,
            handler: Arc::new(MetricsService {
                node_id: host.node_id(),
                start_time: host.start_time(),
                actors: host.actors(),
                metrics: host.metrics(),
                performance_store: host.performance_store(),
            }),
        },
        SystemServiceRegistration {
            manifest: core_manifest(
                SHM_SERVICE,
                SystemServiceExposure::LocalOnly,
                SHM_OPERATIONS,
            ),
            component: shm.clone(),
            handler: shm,
        },
    ]
}

fn core_manifest(
    id: super::service::SystemServiceId,
    exposure: SystemServiceExposure,
    operations: &'static [OperationManifest],
) -> SystemServiceManifest {
    SystemServiceManifest {
        id,
        kind: SystemServiceKind::Core,
        exposure,
        operations,
    }
}

struct RuntimeService {
    node_id: crate::actor::NodeId,
    addr: SocketAddr,
    start_time: Instant,
    actors: Arc<dyn ActorControl>,
    lifecycle: Arc<NodeLifecycle>,
}

#[async_trait]
impl SystemRequestHandler for RuntimeService {
    async fn handle(&self, command: SystemCommand) -> Result<SystemResponse> {
        match command {
            SystemCommand::Runtime(RuntimeCommand::NodeInfo) => Ok(SystemResponse::NodeInfo {
                node_id: self.node_id.0,
                addr: self.addr.to_string(),
                uptime_secs: self.start_time.elapsed().as_secs(),
            }),
            SystemCommand::Runtime(RuntimeCommand::Health) => {
                let state = self.lifecycle.state();
                Ok(SystemResponse::Health {
                    status: if state == NodeState::Ready {
                        "healthy".to_string()
                    } else {
                        state.as_str().to_string()
                    },
                    actors_count: self.actors.count(),
                    uptime_secs: self.start_time.elapsed().as_secs(),
                })
            }
            SystemCommand::Runtime(RuntimeCommand::Ping) => Ok(SystemResponse::Pong {
                node_id: self.node_id.0,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            }),
            _ => Err(service_error("runtime service received an invalid command")),
        }
    }
}

struct ActorsService {
    actors: Arc<dyn ActorControl>,
    factory: Arc<dyn ActorFactory>,
}

#[async_trait]
impl SystemRequestHandler for ActorsService {
    async fn handle(&self, command: SystemCommand) -> Result<SystemResponse> {
        match command {
            SystemCommand::Actors(ActorsCommand::List) => Ok(SystemResponse::ActorList {
                actors: self.actors.list(),
            }),
            SystemCommand::Actors(ActorsCommand::Get { name }) => {
                Ok(match self.actors.get(&name) {
                    Some(info) => SystemResponse::ActorInfo(info),
                    None => SystemResponse::Error {
                        message: format!("Actor not found: {name}"),
                    },
                })
            }
            SystemCommand::Actors(ActorsCommand::Create {
                actor_type,
                name,
                params,
                public,
            }) => {
                let _ = (actor_type, name, params, public);
                Ok(SystemResponse::Error {
                    message: "CreateActor not supported in pure Rust mode. Use Python extension."
                        .to_string(),
                })
            }
            SystemCommand::Actors(ActorsCommand::Stop { name }) => {
                if self.actors.stop(&name).await? {
                    Ok(SystemResponse::Ok)
                } else {
                    Ok(SystemResponse::Error {
                        message: format!("Actor not found: {name}"),
                    })
                }
            }
            SystemCommand::Actors(ActorsCommand::Extension { handler, payload }) => {
                Ok(self.factory.handle_extension(&handler, payload).await)
            }
            _ => Err(service_error("actors service received an invalid command")),
        }
    }
}

struct MetricsService {
    node_id: crate::actor::NodeId,
    start_time: Instant,
    actors: Arc<dyn ActorControl>,
    metrics: Arc<SystemMetrics>,
    performance_store: Arc<PerformanceStore>,
}

#[async_trait]
impl SystemRequestHandler for MetricsService {
    async fn handle(&self, command: SystemCommand) -> Result<SystemResponse> {
        match command {
            SystemCommand::Metrics(MetricsCommand::Snapshot) => {
                let actors_count = self.actors.count();
                let messages_total = self.actors.total_messages();
                let actors_created = self.metrics.actors_created();
                let actors_stopped = self.metrics.actors_stopped();
                let uptime_secs = self.start_time.elapsed().as_secs();
                let ts_unix_micros = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|duration| duration.as_micros() as u64)
                    .unwrap_or(0);
                let node_id = self.node_id.to_string();

                self.performance_store.record(PerformanceSnapshot {
                    ts_unix_micros,
                    node_id: node_id.clone(),
                    actors_count: actors_count as u64,
                    messages_total,
                    actors_created,
                    actors_stopped,
                    uptime_secs,
                });
                crate::metrics_store::write_metrics_snapshot(
                    ts_unix_micros,
                    &node_id,
                    actors_count as u64,
                    messages_total,
                    actors_created,
                    actors_stopped,
                    uptime_secs,
                );

                Ok(SystemResponse::Metrics {
                    actors_count,
                    messages_total,
                    actors_created,
                    actors_stopped,
                    uptime_secs,
                })
            }
            _ => Err(service_error("metrics service received an invalid command")),
        }
    }
}

struct ShmService {
    manager: Arc<ShmManager>,
}

#[async_trait]
impl SystemComponent for ShmService {
    async fn stop(&self) -> Result<()> {
        self.manager.clear();
        Ok(())
    }
}

#[async_trait]
impl SystemRequestHandler for ShmService {
    async fn handle(&self, command: SystemCommand) -> Result<SystemResponse> {
        match command {
            SystemCommand::Shm(ShmCommand::Stats) => {
                let stats = self.manager.stats();
                Ok(SystemResponse::ShmStats {
                    backend: self.manager.backend().as_str().to_string(),
                    regions: stats.regions,
                    published_regions: stats.published_regions,
                    active_leases: stats.active_leases,
                    bytes: stats.bytes,
                })
            }
            _ => Err(service_error("shm service received an invalid command")),
        }
    }
}

pub(crate) fn prometheus_metrics(host: &SystemHost) -> crate::metrics::SystemMetrics {
    let actors = host.actors();
    let metrics = host.metrics();
    crate::metrics::SystemMetrics {
        node_id: host.node_id().0,
        actors_count: actors.count(),
        messages_total: actors.total_messages(),
        actors_created: metrics.actors_created(),
        actors_stopped: metrics.actors_stopped(),
        cluster_members: std::collections::HashMap::new(),
    }
}
