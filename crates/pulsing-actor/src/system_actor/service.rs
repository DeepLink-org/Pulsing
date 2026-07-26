//! System-service registration, lifecycle, and routing.
//!
//! A service is a governed node capability, not a resource owner. The host owns
//! resources, the component owns lifecycle hooks, and the handler owns request
//! semantics. Keeping those roles separate allows future Forge and agent
//! extensions without turning SystemRoot into a service locator.

use super::lifecycle::NodeState;
use super::{SystemMessage, SystemResponse};
use crate::error::{PulsingError, Result, RuntimeError};
use async_trait::async_trait;
use std::sync::{Arc, RwLock};

/// Stable logical identity of a system service.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SystemServiceId {
    pub namespace: &'static str,
    pub major: u16,
}

/// Whether a service is built into the node or installed as an extension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SystemServiceKind {
    Core,
    #[allow(dead_code)]
    Extension,
}

/// Declared exposure policy. Enforcement happens at the protocol boundary
/// once a request carries an authenticated origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SystemServiceExposure {
    LocalOnly,
    AuthenticatedRemote,
}

/// Permission class of an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OperationAccess {
    Read,
    Operate,
    Admin,
}

/// Static operation contract used for policy and discovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OperationManifest {
    pub name: &'static str,
    pub access: OperationAccess,
    pub allowed_while_draining: bool,
}

/// Static identity and policy contract for a system service.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SystemServiceManifest {
    pub id: SystemServiceId,
    pub kind: SystemServiceKind,
    pub exposure: SystemServiceExposure,
    pub operations: &'static [OperationManifest],
}

/// Observable lifecycle state of a registered service.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SystemServiceState {
    Registered,
    Starting,
    Ready,
    Failed,
    Stopping,
    Stopped,
}

/// Internal typed command set. New wire protocols adapt into this model while
/// the public legacy enum remains unchanged.
#[derive(Debug)]
pub(crate) enum SystemCommand {
    Runtime(RuntimeCommand),
    Actors(ActorsCommand),
    Metrics(MetricsCommand),
    Shm(ShmCommand),
}

#[derive(Debug)]
pub(crate) enum RuntimeCommand {
    NodeInfo,
    Health,
    Ping,
}

#[derive(Debug)]
pub(crate) enum ActorsCommand {
    List,
    Get {
        name: String,
    },
    Create {
        actor_type: String,
        name: String,
        params: serde_json::Value,
        public: bool,
    },
    Stop {
        name: String,
    },
    Extension {
        handler: String,
        payload: serde_json::Value,
    },
}

#[derive(Debug)]
pub(crate) enum MetricsCommand {
    Snapshot,
}

#[derive(Debug)]
pub(crate) enum ShmCommand {
    Stats,
}

impl SystemCommand {
    pub(crate) fn from_legacy(message: SystemMessage) -> Self {
        match message {
            SystemMessage::Ping => Self::Runtime(RuntimeCommand::Ping),
            SystemMessage::GetNodeInfo => Self::Runtime(RuntimeCommand::NodeInfo),
            SystemMessage::HealthCheck => Self::Runtime(RuntimeCommand::Health),
            SystemMessage::ListActors => Self::Actors(ActorsCommand::List),
            SystemMessage::GetActor { name } => Self::Actors(ActorsCommand::Get { name }),
            SystemMessage::CreateActor {
                actor_type,
                name,
                params,
                public,
            } => Self::Actors(ActorsCommand::Create {
                actor_type,
                name,
                params,
                public,
            }),
            SystemMessage::StopActor { name } => Self::Actors(ActorsCommand::Stop { name }),
            SystemMessage::Extension { handler, payload } => {
                Self::Actors(ActorsCommand::Extension { handler, payload })
            }
            SystemMessage::GetMetrics => Self::Metrics(MetricsCommand::Snapshot),
            SystemMessage::GetShmStats => Self::Shm(ShmCommand::Stats),
        }
    }

    pub(crate) fn target(&self) -> SystemServiceId {
        match self {
            Self::Runtime(_) => RUNTIME_SERVICE,
            Self::Actors(_) => ACTORS_SERVICE,
            Self::Metrics(_) => METRICS_SERVICE,
            Self::Shm(_) => SHM_SERVICE,
        }
    }

    pub(crate) fn operation(&self) -> &'static str {
        match self {
            Self::Runtime(RuntimeCommand::NodeInfo) => "node_info",
            Self::Runtime(RuntimeCommand::Health) => "health",
            Self::Runtime(RuntimeCommand::Ping) => "ping",
            Self::Actors(ActorsCommand::List) => "list",
            Self::Actors(ActorsCommand::Get { .. }) => "get",
            Self::Actors(ActorsCommand::Create { .. }) => "create",
            Self::Actors(ActorsCommand::Stop { .. }) => "stop",
            Self::Actors(ActorsCommand::Extension { .. }) => "extension",
            Self::Metrics(MetricsCommand::Snapshot) => "snapshot",
            Self::Shm(ShmCommand::Stats) => "stats",
        }
    }
}

pub(crate) const RUNTIME_SERVICE: SystemServiceId = SystemServiceId {
    namespace: "runtime",
    major: 1,
};
pub(crate) const ACTORS_SERVICE: SystemServiceId = SystemServiceId {
    namespace: "actors",
    major: 1,
};
pub(crate) const METRICS_SERVICE: SystemServiceId = SystemServiceId {
    namespace: "metrics",
    major: 1,
};
pub(crate) const SHM_SERVICE: SystemServiceId = SystemServiceId {
    namespace: "shm",
    major: 1,
};

/// Lifecycle hooks for a node component.
#[async_trait]
pub(crate) trait SystemComponent: Send + Sync {
    async fn start(&self) -> Result<()> {
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        Ok(())
    }
}

/// Request semantics for one registered service.
#[async_trait]
pub(crate) trait SystemRequestHandler: Send + Sync {
    async fn handle(&self, command: SystemCommand) -> Result<SystemResponse>;
}

pub(crate) struct StatelessComponent;

#[async_trait]
impl SystemComponent for StatelessComponent {}

/// Complete bootstrap registration for one service.
pub(crate) struct SystemServiceRegistration {
    pub manifest: SystemServiceManifest,
    pub component: Arc<dyn SystemComponent>,
    pub handler: Arc<dyn SystemRequestHandler>,
}

struct RegisteredSystemService {
    registration: SystemServiceRegistration,
    state: SystemServiceState,
}

/// Bootstrap-time registry and runtime router owned by SystemRoot.
///
/// Registration is intentionally closed after startup. Runtime installation
/// will require a staged generation/rollback protocol rather than direct
/// mutation of this registry.
#[derive(Default)]
pub(crate) struct SystemServiceRegistry {
    entries: RwLock<Vec<RegisteredSystemService>>,
}

impl SystemServiceRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn register(&self, registration: SystemServiceRegistration) -> Result<()> {
        let manifest = &registration.manifest;
        let mut entries = self
            .entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if entries
            .iter()
            .any(|entry| entry.registration.manifest.id == manifest.id)
        {
            return Err(service_error(format!(
                "service {}@{} is already registered",
                manifest.id.namespace, manifest.id.major
            )));
        }
        validate_manifest(manifest)?;
        entries.push(RegisteredSystemService {
            registration,
            state: SystemServiceState::Registered,
        });
        Ok(())
    }

    pub(crate) fn statuses(&self) -> Vec<(SystemServiceManifest, SystemServiceState)> {
        self.entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|entry| (entry.registration.manifest.clone(), entry.state))
            .collect()
    }

    pub(crate) async fn start_all(&self) -> Result<()> {
        let components: Vec<_> = self
            .entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|entry| {
                (
                    entry.registration.manifest.id,
                    entry.registration.component.clone(),
                )
            })
            .collect();

        for (id, component) in components {
            self.transition_service(
                id,
                SystemServiceState::Registered,
                SystemServiceState::Starting,
            )?;
            if let Err(error) = component.start().await {
                let _ = self.set_state(id, SystemServiceState::Failed);
                let _ = self.stop_ready().await;
                return Err(error);
            }
            self.transition_service(id, SystemServiceState::Starting, SystemServiceState::Ready)?;
        }
        Ok(())
    }

    pub(crate) async fn stop_all(&self) -> Result<()> {
        self.stop_ready().await
    }

    pub(crate) async fn dispatch(
        &self,
        command: SystemCommand,
        node_state: NodeState,
    ) -> Result<SystemResponse> {
        let target = command.target();
        let operation_name = command.operation();
        let (handler, operation) = {
            let entries = self
                .entries
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let entry = entries
                .iter()
                .find(|entry| entry.registration.manifest.id == target)
                .ok_or_else(|| {
                    service_error(format!(
                        "service {}@{} is not registered",
                        target.namespace, target.major
                    ))
                })?;
            if entry.state != SystemServiceState::Ready {
                return Err(service_error(format!(
                    "service {}@{} is not ready ({:?})",
                    target.namespace, target.major, entry.state
                )));
            }
            let operation = entry
                .registration
                .manifest
                .operations
                .iter()
                .find(|operation| operation.name == operation_name)
                .copied()
                .ok_or_else(|| {
                    service_error(format!(
                        "operation {operation_name} is not declared by {}@{}",
                        target.namespace, target.major
                    ))
                })?;
            (entry.registration.handler.clone(), operation)
        };

        if node_state == NodeState::Draining && !operation.allowed_while_draining {
            return Err(service_error(format!(
                "operation {} is unavailable while the node is draining",
                operation.name
            )));
        }
        handler.handle(command).await
    }

    fn transition_service(
        &self,
        id: SystemServiceId,
        expected: SystemServiceState,
        next: SystemServiceState,
    ) -> Result<()> {
        let mut entries = self
            .entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = entries
            .iter_mut()
            .find(|entry| entry.registration.manifest.id == id)
            .ok_or_else(|| service_error("registered service disappeared"))?;
        if entry.state != expected {
            return Err(service_error(format!(
                "invalid service transition for {}@{}: {:?} -> {:?}",
                id.namespace, id.major, entry.state, next
            )));
        }
        entry.state = next;
        Ok(())
    }

    fn set_state(&self, id: SystemServiceId, state: SystemServiceState) -> Result<()> {
        let mut entries = self
            .entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = entries
            .iter_mut()
            .find(|entry| entry.registration.manifest.id == id)
            .ok_or_else(|| service_error("registered service disappeared"))?;
        entry.state = state;
        Ok(())
    }

    async fn stop_ready(&self) -> Result<()> {
        let components: Vec<_> = self
            .entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .rev()
            .filter(|entry| entry.state == SystemServiceState::Ready)
            .map(|entry| {
                (
                    entry.registration.manifest.id,
                    entry.registration.component.clone(),
                )
            })
            .collect();
        let mut failures = Vec::new();
        for (id, component) in components {
            self.transition_service(id, SystemServiceState::Ready, SystemServiceState::Stopping)?;
            match component.stop().await {
                Ok(()) => self.transition_service(
                    id,
                    SystemServiceState::Stopping,
                    SystemServiceState::Stopped,
                )?,
                Err(error) => {
                    self.set_state(id, SystemServiceState::Failed)?;
                    failures.push(format!("{}@{}: {error}", id.namespace, id.major));
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(service_error(format!(
                "failed to stop system services: {}",
                failures.join("; ")
            )))
        }
    }
}

fn validate_manifest(manifest: &SystemServiceManifest) -> Result<()> {
    if manifest.id.namespace.is_empty() {
        return Err(service_error("service namespace must not be empty"));
    }
    if manifest.id.major == 0 {
        return Err(service_error("service major version must be non-zero"));
    }
    for (index, operation) in manifest.operations.iter().enumerate() {
        if operation.name.is_empty() {
            return Err(service_error("operation name must not be empty"));
        }
        if manifest.operations[..index]
            .iter()
            .any(|existing| existing.name == operation.name)
        {
            return Err(service_error(format!(
                "operation {} is declared more than once",
                operation.name
            )));
        }
    }
    Ok(())
}

pub(crate) fn service_error(message: impl Into<String>) -> PulsingError {
    PulsingError::from(RuntimeError::Other(format!(
        "System service error: {}",
        message.into()
    )))
}

#[cfg(test)]
mod tests {
    use super::super::builtin;
    use super::super::host::SystemHost;
    use super::super::{
        ActorRegistry, DefaultActorFactory, SystemMetrics, SystemRef, SystemResponse,
    };
    use super::*;
    use crate::actor::NodeId;
    use crate::performance_store::PerformanceStore;
    use std::sync::Arc;

    fn test_host() -> SystemHost {
        let system_ref = SystemRef::new(NodeId::generate(), "127.0.0.1:0".parse().unwrap());
        SystemHost::standalone(
            &system_ref,
            Arc::new(ActorRegistry::new()),
            Arc::new(SystemMetrics::new()),
            Arc::new(PerformanceStore::new(8)),
        )
    }

    fn builtins(host: &SystemHost) -> Vec<SystemServiceRegistration> {
        builtin::registrations(host, Arc::new(DefaultActorFactory))
    }

    #[test]
    fn registry_rejects_duplicate_service_major() {
        let registry = SystemServiceRegistry::new();
        let host = test_host();
        registry.register(builtins(&host).remove(0)).unwrap();
        assert!(registry.register(builtins(&host).remove(0)).is_err());
    }

    #[tokio::test]
    async fn registry_requires_ready_and_honors_draining_policy() {
        let registry = SystemServiceRegistry::new();
        let host = test_host();
        for registration in builtins(&host) {
            registry.register(registration).unwrap();
        }

        assert!(registry
            .dispatch(
                SystemCommand::Runtime(RuntimeCommand::Ping),
                host.node_state(),
            )
            .await
            .is_err());

        host.lifecycle().transition(NodeState::Starting).unwrap();
        registry.start_all().await.unwrap();
        host.lifecycle().transition(NodeState::Ready).unwrap();
        assert!(matches!(
            registry
                .dispatch(
                    SystemCommand::Runtime(RuntimeCommand::Ping),
                    host.node_state(),
                )
                .await
                .unwrap(),
            SystemResponse::Pong { .. }
        ));

        host.lifecycle().begin_draining().unwrap();
        assert!(registry
            .dispatch(
                SystemCommand::Actors(ActorsCommand::Stop {
                    name: "actors/test".to_string(),
                }),
                host.node_state(),
            )
            .await
            .is_err());
        assert!(registry
            .dispatch(
                SystemCommand::Runtime(RuntimeCommand::Ping),
                host.node_state(),
            )
            .await
            .is_ok());

        registry.stop_all().await.unwrap();
        assert!(registry
            .statuses()
            .iter()
            .all(|(_, state)| *state == SystemServiceState::Stopped));
    }
}
