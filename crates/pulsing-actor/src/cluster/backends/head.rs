//! Head node backend implementation.

use crate::actor::{ActorId, ActorPath, NodeId, StopReason};
use crate::cluster::{
    member::{MemberInfo, MemberStatus, NamedActorInfo, NamedActorInstance},
    NamingBackend,
};
use crate::transport::http2::{Http2Client, Http2Config};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Configuration for head node backend.
#[derive(Clone, Debug)]
pub struct HeadNodeConfig {
    pub sync_interval: Duration,
    pub heartbeat_interval: Duration,
    pub heartbeat_timeout: Duration,
}

impl Default for HeadNodeConfig {
    fn default() -> Self {
        Self {
            sync_interval: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Clone, Debug)]
enum NodeMode {
    Head,
    Worker { head_addr: SocketAddr },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NodeRegistration {
    node_id: NodeId,
    addr: SocketAddr,
    last_heartbeat: u64, // milliseconds since epoch
}

struct HeadNodeState {
    nodes: HashMap<NodeId, NodeRegistration>,
    named_actors: HashMap<String, NamedActorInfo>,
    actors: HashMap<ActorId, NodeId>,
}

impl HeadNodeState {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            named_actors: HashMap::new(),
            actors: HashMap::new(),
        }
    }

    fn register_node(&mut self, node_id: NodeId, addr: SocketAddr) {
        let now = HeadNodeBackend::now_millis();
        self.nodes.insert(
            node_id,
            NodeRegistration {
                node_id,
                addr,
                last_heartbeat: now,
            },
        );
    }

    fn update_heartbeat(&mut self, node_id: &NodeId) -> bool {
        if let Some(reg) = self.nodes.get_mut(node_id) {
            reg.last_heartbeat = HeadNodeBackend::now_millis();
            true
        } else {
            false
        }
    }

    fn remove_stale_nodes(&mut self, timeout_ms: u64) -> Vec<NodeId> {
        let now = HeadNodeBackend::now_millis();
        let mut removed = Vec::new();

        self.nodes.retain(|node_id, reg| {
            if now.saturating_sub(reg.last_heartbeat) > timeout_ms {
                removed.push(*node_id);
                false
            } else {
                true
            }
        });

        for node_id in &removed {
            self.named_actors.values_mut().for_each(|info| {
                info.remove_instance(node_id);
            });
            self.named_actors.retain(|_, info| !info.is_empty());
            self.actors.retain(|_, nid| nid != node_id);
        }

        removed
    }

    fn register_named_actor(
        &mut self,
        path: ActorPath,
        node_id: NodeId,
        actor_id: Option<ActorId>,
        metadata: HashMap<String, String>,
    ) {
        let key = path.as_str();
        if let Some(aid) = actor_id {
            let instance = NamedActorInstance::with_metadata(node_id, aid, metadata);
            self.named_actors
                .entry(key.clone())
                .and_modify(|info| info.add_full_instance(instance.clone()))
                .or_insert_with(|| NamedActorInfo::with_full_instance(path, instance));
        } else {
            self.named_actors
                .entry(key.clone())
                .and_modify(|info| info.add_instance(node_id))
                .or_insert_with(|| NamedActorInfo::with_instance(path, node_id));
        }
    }

    fn unregister_named_actor(&mut self, path: &ActorPath, node_id: &NodeId) {
        let key = path.as_str().to_string();
        if let Some(info) = self.named_actors.get_mut(&key) {
            info.remove_instance(node_id);
            if info.is_empty() {
                self.named_actors.remove(&key);
            }
        }
    }

    fn register_actor(&mut self, actor_id: ActorId, node_id: NodeId) {
        self.actors.insert(actor_id, node_id);
    }

    fn unregister_actor(&mut self, actor_id: &ActorId) {
        self.actors.remove(actor_id);
    }

    fn all_members(&self) -> Vec<MemberInfo> {
        self.nodes
            .values()
            .map(|reg| MemberInfo {
                node_id: reg.node_id,
                addr: reg.addr,
                gossip_addr: reg.addr,
                status: MemberStatus::Alive,
                incarnation: 0,
                last_update: None,
            })
            .collect()
    }

    fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        self.named_actors.values().cloned().collect()
    }
}

// ============================================================================
// Worker Node State
// ============================================================================

/// Worker node state (only used in worker mode)
struct WorkerNodeState {
    /// Cached members from head node
    members: HashMap<NodeId, MemberInfo>,
    /// Cached named actors from head node
    named_actors: HashMap<String, NamedActorInfo>,
    /// Cached actors from head node
    actors: HashMap<ActorId, NodeId>,
    /// Pending sync operations
    pending_sync: Vec<SyncOperation>,
}

#[derive(Clone, Debug)]
enum SyncOperation {
    RegisterNamed {
        path: ActorPath,
        actor_id: Option<ActorId>,
        metadata: HashMap<String, String>,
    },
    UnregisterNamed {
        path: ActorPath,
    },
    Register {
        actor_id: ActorId,
    },
    Unregister {
        actor_id: ActorId,
    },
}

impl WorkerNodeState {
    fn new() -> Self {
        Self {
            members: HashMap::new(),
            named_actors: HashMap::new(),
            actors: HashMap::new(),
            pending_sync: Vec::new(),
        }
    }

    /// Update state from sync response
    fn update_from_sync(&mut self, sync: SyncResponse) {
        // Update members
        self.members.clear();
        for member in sync.members {
            self.members.insert(member.node_id, member);
        }

        // Update named actors
        self.named_actors.clear();
        for info in sync.named_actors {
            self.named_actors
                .insert(info.path.as_str().to_string(), info);
        }

        // Update actors
        self.actors.clear();
        for (actor_id, node_id) in sync.actors {
            self.actors.insert(actor_id, node_id);
        }
    }

    /// Add pending sync operation
    fn add_pending_sync(&mut self, op: SyncOperation) {
        self.pending_sync.push(op);
    }

    /// Get and clear pending sync operations
    fn take_pending_sync(&mut self) -> Vec<SyncOperation> {
        std::mem::take(&mut self.pending_sync)
    }
}

// ============================================================================
// Shared State
// ============================================================================

enum BackendState {
    Head(HeadNodeState),
    Worker(WorkerNodeState),
}

// ============================================================================
// Head Node Backend
// ============================================================================

/// Head node based naming backend
#[derive(Clone)]
pub struct HeadNodeBackend {
    mode: NodeMode,
    local_node: NodeId,
    local_addr: SocketAddr,
    state: Arc<RwLock<BackendState>>,
    http_client: Option<Arc<Http2Client>>,
    config: HeadNodeConfig,
}

impl HeadNodeBackend {
    /// Create a new head node backend
    pub fn new(
        local_node: NodeId,
        local_addr: SocketAddr,
        is_head_node: bool,
        head_addr: Option<SocketAddr>,
        http_config: Http2Config,
    ) -> Self {
        Self::with_config(
            local_node,
            local_addr,
            is_head_node,
            head_addr,
            http_config,
            HeadNodeConfig::default(),
        )
    }

    pub fn with_config(
        local_node: NodeId,
        local_addr: SocketAddr,
        _is_head_node: bool,
        head_addr: Option<SocketAddr>,
        http_config: Http2Config,
        config: HeadNodeConfig,
    ) -> Self {
        let mode = if let Some(addr) = head_addr {
            NodeMode::Worker { head_addr: addr }
        } else {
            NodeMode::Head
        };

        let state = match &mode {
            NodeMode::Head => BackendState::Head(HeadNodeState::new()),
            NodeMode::Worker { .. } => BackendState::Worker(WorkerNodeState::new()),
        };

        let http_client = match &mode {
            NodeMode::Head => None,
            NodeMode::Worker { .. } => Some(Arc::new(Http2Client::new(http_config))),
        };

        if let Some(client) = &http_client {
            client.start_background_tasks();
        }

        Self {
            mode,
            local_node,
            local_addr,
            state: Arc::new(RwLock::new(state)),
            http_client,
            config,
        }
    }

    /// Get current timestamp in milliseconds
    fn now_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Check if this is a head node
    pub fn is_head(&self) -> bool {
        matches!(self.mode, NodeMode::Head)
    }

    /// Get head node address (if worker mode)
    pub fn head_addr(&self) -> Option<SocketAddr> {
        match &self.mode {
            NodeMode::Head => None,
            NodeMode::Worker { head_addr } => Some(*head_addr),
        }
    }

    /// Handle node registration (head node only)
    pub async fn handle_register_node(
        &self,
        node_id: NodeId,
        addr: SocketAddr,
    ) -> anyhow::Result<()> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let mut state = self.state.write().await;
        if let BackendState::Head(ref mut head_state) = *state {
            head_state.register_node(node_id, addr);
            tracing::info!(node_id = %node_id, addr = %addr, "Node registered");
        }
        Ok(())
    }

    /// Handle heartbeat (head node only)
    pub async fn handle_heartbeat(&self, node_id: &NodeId) -> anyhow::Result<()> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let mut state = self.state.write().await;
        if let BackendState::Head(ref mut head_state) = *state {
            if head_state.update_heartbeat(node_id) {
                Ok(())
            } else {
                Err(anyhow::anyhow!("Node not found: {}", node_id))
            }
        } else {
            Err(anyhow::anyhow!("Invalid state"))
        }
    }

    /// Handle register named actor (head node only)
    pub async fn handle_register_named_actor(
        &self,
        path: ActorPath,
        node_id: NodeId,
        actor_id: Option<ActorId>,
        metadata: HashMap<String, String>,
    ) -> anyhow::Result<()> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let mut state = self.state.write().await;
        if let BackendState::Head(ref mut head_state) = *state {
            head_state.register_named_actor(path.clone(), node_id, actor_id, metadata);
            tracing::debug!(path = %path, node_id = %node_id, "Named actor registered on head");
        }
        Ok(())
    }

    /// Handle unregister named actor (head node only)
    pub async fn handle_unregister_named_actor(
        &self,
        path: &ActorPath,
        node_id: &NodeId,
    ) -> anyhow::Result<()> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let mut state = self.state.write().await;
        if let BackendState::Head(ref mut head_state) = *state {
            head_state.unregister_named_actor(path, node_id);
            tracing::debug!(path = %path, node_id = %node_id, "Named actor unregistered from head");
        }
        Ok(())
    }

    /// Handle register actor (head node only)
    pub async fn handle_register_actor(
        &self,
        actor_id: ActorId,
        node_id: NodeId,
    ) -> anyhow::Result<()> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let mut state = self.state.write().await;
        if let BackendState::Head(ref mut head_state) = *state {
            head_state.register_actor(actor_id, node_id);
            tracing::debug!(actor_id = %actor_id, node_id = %node_id, "Actor registered on head");
        }
        Ok(())
    }

    /// Handle unregister actor (head node only)
    pub async fn handle_unregister_actor(
        &self,
        actor_id: &ActorId,
        node_id: &NodeId,
    ) -> anyhow::Result<()> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let mut state = self.state.write().await;
        if let BackendState::Head(ref mut head_state) = *state {
            // Only unregister if the actor belongs to this node
            if head_state.actors.get(actor_id) == Some(node_id) {
                head_state.unregister_actor(actor_id);
                tracing::debug!(actor_id = %actor_id, node_id = %node_id, "Actor unregistered from head");
            }
        }
        Ok(())
    }

    /// Handle sync request (head node only) - returns current state
    pub async fn handle_sync(&self) -> anyhow::Result<SyncResponse> {
        if !self.is_head() {
            return Err(anyhow::anyhow!("Not a head node"));
        }

        let state = self.state.read().await;
        if let BackendState::Head(ref head_state) = *state {
            let members = head_state.all_members();
            let named_actors = head_state.all_named_actors();
            let actors: Vec<_> = head_state.actors.iter().map(|(k, v)| (*k, *v)).collect();

            Ok(SyncResponse {
                members,
                named_actors,
                actors,
            })
        } else {
            Err(anyhow::anyhow!("Invalid state"))
        }
    }

    /// Make HTTP request to head node (worker mode only)
    async fn request_head<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        msg_type: &str,
        payload: Vec<u8>,
    ) -> anyhow::Result<T> {
        let head_addr = self.head_addr().ok_or_else(|| {
            anyhow::anyhow!("Cannot make request to head node: this is a head node")
        })?;
        let client = self
            .http_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HTTP client not available"))?;

        let response_bytes = client.ask(head_addr, path, msg_type, payload).await?;
        let result: T = bincode::deserialize(&response_bytes)?;
        Ok(result)
    }

    /// Send request to head node without response (worker mode only)
    async fn tell_head(&self, path: &str, msg_type: &str, payload: Vec<u8>) -> anyhow::Result<()> {
        let head_addr = self
            .head_addr()
            .ok_or_else(|| anyhow::anyhow!("Cannot send to head node: this is a head node"))?;
        let client = self
            .http_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("HTTP client not available"))?;

        tracing::debug!(
            head_addr = %head_addr,
            path = %path,
            msg_type = %msg_type,
            "Sending tell to head node"
        );

        client.tell(head_addr, path, msg_type, payload).await
    }

    /// Sync with head node (worker mode only)
    async fn sync_from_head(&self) -> anyhow::Result<()> {
        if self.is_head() {
            return Err(anyhow::anyhow!("Cannot sync: this is a head node"));
        }

        let sync: SyncResponse = self
            .request_head("/cluster/head/sync", "sync", Vec::new())
            .await?;

        let mut state = self.state.write().await;
        if let BackendState::Worker(ref mut worker_state) = *state {
            worker_state.update_from_sync(sync);
        }

        Ok(())
    }

    /// Register with head node (worker mode only)
    async fn register_with_head(&self) -> anyhow::Result<()> {
        if self.is_head() {
            return Err(anyhow::anyhow!("Cannot register: this is a head node"));
        }

        let req = RegisterNodeRequest {
            node_id: self.local_node,
            addr: self.local_addr,
        };
        let payload = bincode::serialize(&req)?;

        self.tell_head("/cluster/head/register", "register_node", payload)
            .await?;

        tracing::info!(
            node_id = %self.local_node,
            head_addr = ?self.head_addr(),
            "Registered with head node"
        );

        Ok(())
    }

    /// Send heartbeat to head node (worker mode only)
    async fn send_heartbeat(&self) -> anyhow::Result<()> {
        if self.is_head() {
            return Err(anyhow::anyhow!(
                "Cannot send heartbeat: this is a head node"
            ));
        }

        let req = HeartbeatRequest {
            node_id: self.local_node,
        };
        let payload = bincode::serialize(&req)?;

        self.tell_head("/cluster/head/heartbeat", "heartbeat", payload)
            .await?;

        Ok(())
    }

    /// Process pending sync operations (worker mode only)
    async fn process_pending_sync(&self) -> anyhow::Result<()> {
        if self.is_head() {
            return Ok(()); // No pending sync for head node
        }

        let pending = {
            let mut state = self.state.write().await;
            if let BackendState::Worker(ref mut worker_state) = *state {
                worker_state.take_pending_sync()
            } else {
                return Ok(());
            }
        };

        for op in pending {
            match op {
                SyncOperation::RegisterNamed {
                    path,
                    actor_id,
                    metadata,
                } => {
                    let req = RegisterNamedActorRequest {
                        path: path.clone(),
                        node_id: self.local_node,
                        actor_id,
                        metadata: metadata.clone(),
                    };
                    let payload = bincode::serialize(&req)?;
                    if let Err(e) = self
                        .tell_head(
                            "/cluster/head/named_actor/register",
                            "register_named_actor",
                            payload,
                        )
                        .await
                    {
                        tracing::warn!(path = %path, error = %e, "Failed to sync named actor registration");
                        // Re-add to pending queue
                        let mut state = self.state.write().await;
                        if let BackendState::Worker(ref mut worker_state) = *state {
                            worker_state.add_pending_sync(SyncOperation::RegisterNamed {
                                path,
                                actor_id,
                                metadata,
                            });
                        }
                    }
                }
                SyncOperation::UnregisterNamed { path } => {
                    let req = UnregisterNamedActorRequest {
                        path: path.clone(),
                        node_id: self.local_node,
                    };
                    let payload = bincode::serialize(&req)?;
                    if let Err(e) = self
                        .tell_head(
                            "/cluster/head/named_actor/unregister",
                            "unregister_named_actor",
                            payload,
                        )
                        .await
                    {
                        tracing::warn!(path = %path, error = %e, "Failed to sync named actor unregistration");
                        // Re-add to pending queue
                        let mut state = self.state.write().await;
                        if let BackendState::Worker(ref mut worker_state) = *state {
                            worker_state.add_pending_sync(SyncOperation::UnregisterNamed { path });
                        }
                    }
                }
                SyncOperation::Register { actor_id } => {
                    let req = RegisterActorRequest {
                        actor_id,
                        node_id: self.local_node,
                    };
                    let payload = bincode::serialize(&req)?;
                    if let Err(e) = self
                        .tell_head("/cluster/head/actor/register", "register_actor", payload)
                        .await
                    {
                        tracing::warn!(actor_id = %actor_id, error = %e, "Failed to sync actor registration");
                        let mut state = self.state.write().await;
                        if let BackendState::Worker(ref mut worker_state) = *state {
                            worker_state.add_pending_sync(SyncOperation::Register { actor_id });
                        }
                    }
                }
                SyncOperation::Unregister { actor_id } => {
                    let req = UnregisterActorRequest {
                        actor_id,
                        node_id: self.local_node,
                    };
                    let payload = bincode::serialize(&req)?;
                    if let Err(e) = self
                        .tell_head(
                            "/cluster/head/actor/unregister",
                            "unregister_actor",
                            payload,
                        )
                        .await
                    {
                        tracing::warn!(actor_id = %actor_id, error = %e, "Failed to sync actor unregistration");
                        // Re-add to pending queue
                        let mut state = self.state.write().await;
                        if let BackendState::Worker(ref mut worker_state) = *state {
                            worker_state.add_pending_sync(SyncOperation::Unregister { actor_id });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// NamingBackend Trait Implementation
// ============================================================================

#[async_trait]
impl NamingBackend for HeadNodeBackend {
    // ========================================================================
    // Node Management
    // ========================================================================

    async fn join(&self, _seeds: Vec<SocketAddr>) -> anyhow::Result<()> {
        if self.is_head() {
            // Head node: register itself
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.register_node(self.local_node, self.local_addr);
            }
            tracing::info!("Head node started and registered itself");
            Ok(())
        } else {
            // Worker node: register with head node
            self.register_with_head().await?;
            // Initial sync
            self.sync_from_head().await?;
            Ok(())
        }
    }

    async fn leave(&self) -> anyhow::Result<()> {
        if self.is_head() {
            // Head node: clear all registrations
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.nodes.clear();
                head_state.named_actors.clear();
                head_state.actors.clear();
            }
            tracing::info!("Head node leaving");
        } else {
            // Worker node: unregister from head (best effort)
            // Note: We don't have an unregister endpoint, so we just stop syncing
            tracing::info!(node_id = %self.local_node, "Worker node leaving");
        }
        Ok(())
    }

    async fn all_members(&self) -> Vec<MemberInfo> {
        if self.is_head() {
            let state = self.state.read().await;
            if let BackendState::Head(ref head_state) = *state {
                head_state.all_members()
            } else {
                Vec::new()
            }
        } else {
            // Worker: return cached members
            let state = self.state.read().await;
            if let BackendState::Worker(ref worker_state) = *state {
                worker_state.members.values().cloned().collect()
            } else {
                Vec::new()
            }
        }
    }

    async fn alive_members(&self) -> Vec<MemberInfo> {
        let all = self.all_members().await;
        all.into_iter()
            .filter(|m| m.status == MemberStatus::Alive && m.node_id != self.local_node)
            .collect()
    }

    async fn get_member(&self, node_id: &NodeId) -> Option<MemberInfo> {
        if self.is_head() {
            let state = self.state.read().await;
            if let BackendState::Head(ref head_state) = *state {
                head_state.nodes.get(node_id).map(|reg| MemberInfo {
                    node_id: reg.node_id,
                    addr: reg.addr,
                    gossip_addr: reg.addr,
                    status: MemberStatus::Alive,
                    incarnation: 0,
                    last_update: None,
                })
            } else {
                None
            }
        } else {
            // Worker: return from cache
            let state = self.state.read().await;
            if let BackendState::Worker(ref worker_state) = *state {
                worker_state.members.get(node_id).cloned()
            } else {
                None
            }
        }
    }

    // ========================================================================
    // Named Actor Registration
    // ========================================================================

    async fn register_named_actor(&self, path: ActorPath) {
        if self.is_head() {
            // Head node: register directly
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.register_named_actor(
                    path.clone(),
                    self.local_node,
                    None,
                    HashMap::new(),
                );
            }
        } else {
            // Worker: add to pending sync and update local cache
            {
                let mut state = self.state.write().await;
                if let BackendState::Worker(ref mut worker_state) = *state {
                    worker_state.add_pending_sync(SyncOperation::RegisterNamed {
                        path: path.clone(),
                        actor_id: None,
                        metadata: HashMap::new(),
                    });
                    // Update local cache immediately
                    worker_state
                        .named_actors
                        .entry(path.as_str().to_string())
                        .and_modify(|info| info.add_instance(self.local_node))
                        .or_insert_with(|| {
                            NamedActorInfo::with_instance(path.clone(), self.local_node)
                        });
                }
            }
            // Try to sync immediately
            let _ = self.process_pending_sync().await;
        }
    }

    async fn register_named_actor_full(
        &self,
        path: ActorPath,
        actor_id: ActorId,
        metadata: HashMap<String, String>,
    ) {
        if self.is_head() {
            // Head node: register directly
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.register_named_actor(
                    path.clone(),
                    self.local_node,
                    Some(actor_id),
                    metadata.clone(),
                );
            }
        } else {
            // Worker: add to pending sync and update local cache
            {
                let mut state = self.state.write().await;
                if let BackendState::Worker(ref mut worker_state) = *state {
                    worker_state.add_pending_sync(SyncOperation::RegisterNamed {
                        path: path.clone(),
                        actor_id: Some(actor_id),
                        metadata: metadata.clone(),
                    });
                    // Update local cache immediately
                    let instance =
                        NamedActorInstance::with_metadata(self.local_node, actor_id, metadata);
                    worker_state
                        .named_actors
                        .entry(path.as_str().to_string())
                        .and_modify(|info| info.add_full_instance(instance.clone()))
                        .or_insert_with(|| {
                            NamedActorInfo::with_full_instance(path.clone(), instance)
                        });
                }
            }
            // Try to sync immediately
            let _ = self.process_pending_sync().await;
        }
    }

    async fn unregister_named_actor(&self, path: &ActorPath) {
        if self.is_head() {
            // Head node: unregister directly
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.unregister_named_actor(path, &self.local_node);
            }
        } else {
            // Worker: add to pending sync and update local cache
            {
                let mut state = self.state.write().await;
                if let BackendState::Worker(ref mut worker_state) = *state {
                    worker_state
                        .add_pending_sync(SyncOperation::UnregisterNamed { path: path.clone() });
                    // Update local cache immediately
                    if let Some(info) = worker_state
                        .named_actors
                        .get_mut(&path.as_str().to_string())
                    {
                        info.remove_instance(&self.local_node);
                        if info.is_empty() {
                            worker_state.named_actors.remove(&path.as_str().to_string());
                        }
                    }
                }
            }
            // Try to sync immediately
            let _ = self.process_pending_sync().await;
        }
    }

    async fn broadcast_named_actor_failed(&self, path: &ActorPath, _reason: &StopReason) {
        // Same as unregister
        self.unregister_named_actor(path).await;
    }

    // ========================================================================
    // Named Actor Queries
    // ========================================================================

    async fn lookup_named_actor(&self, path: &ActorPath) -> Option<NamedActorInfo> {
        if self.is_head() {
            let state = self.state.read().await;
            if let BackendState::Head(ref head_state) = *state {
                head_state
                    .named_actors
                    .get(&path.as_str().to_string())
                    .cloned()
            } else {
                None
            }
        } else {
            // Worker: return from cache
            let state = self.state.read().await;
            if let BackendState::Worker(ref worker_state) = *state {
                worker_state
                    .named_actors
                    .get(&path.as_str().to_string())
                    .cloned()
            } else {
                None
            }
        }
    }

    async fn select_named_actor_instance(&self, path: &ActorPath) -> Option<MemberInfo> {
        let info = self.lookup_named_actor(path).await?;
        let node_id = info.select_instance()?;
        self.get_member(&node_id).await
    }

    async fn get_named_actor_instances(&self, path: &ActorPath) -> Vec<MemberInfo> {
        let info = match self.lookup_named_actor(path).await {
            Some(info) => info,
            None => return Vec::new(),
        };

        let mut members = Vec::new();
        for node_id in &info.instance_nodes {
            if let Some(member) = self.get_member(node_id).await {
                members.push(member);
            }
        }
        members
    }

    async fn get_named_actor_instances_detailed(
        &self,
        path: &ActorPath,
    ) -> Vec<(MemberInfo, Option<NamedActorInstance>)> {
        let info = match self.lookup_named_actor(path).await {
            Some(info) => info,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        for node_id in &info.instance_nodes {
            if let Some(member) = self.get_member(node_id).await {
                let instance = info.get_instance(node_id).cloned();
                result.push((member, instance));
            }
        }
        result
    }

    async fn all_named_actors(&self) -> Vec<NamedActorInfo> {
        if self.is_head() {
            let state = self.state.read().await;
            if let BackendState::Head(ref head_state) = *state {
                head_state.all_named_actors()
            } else {
                Vec::new()
            }
        } else {
            // Worker: return from cache
            let state = self.state.read().await;
            if let BackendState::Worker(ref worker_state) = *state {
                worker_state.named_actors.values().cloned().collect()
            } else {
                Vec::new()
            }
        }
    }

    // ========================================================================
    // Actor Registration
    // ========================================================================

    async fn register_actor(&self, actor_id: ActorId) {
        if self.is_head() {
            // Head node: register directly
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.register_actor(actor_id, self.local_node);
            }
        } else {
            // Worker: add to pending sync and update local cache
            {
                let mut state = self.state.write().await;
                if let BackendState::Worker(ref mut worker_state) = *state {
                    worker_state.add_pending_sync(SyncOperation::Register { actor_id });
                    worker_state.actors.insert(actor_id, self.local_node);
                }
            }
            // Try to sync immediately
            let _ = self.process_pending_sync().await;
        }
    }

    async fn unregister_actor(&self, actor_id: &ActorId) {
        if self.is_head() {
            // Head node: unregister directly
            let mut state = self.state.write().await;
            if let BackendState::Head(ref mut head_state) = *state {
                head_state.unregister_actor(actor_id);
            }
        } else {
            // Worker: add to pending sync and update local cache
            {
                let mut state = self.state.write().await;
                if let BackendState::Worker(ref mut worker_state) = *state {
                    worker_state.add_pending_sync(SyncOperation::Unregister {
                        actor_id: *actor_id,
                    });
                    worker_state.actors.remove(actor_id);
                }
            }
            // Try to sync immediately
            let _ = self.process_pending_sync().await;
        }
    }

    async fn lookup_actor(&self, actor_id: &ActorId) -> Option<MemberInfo> {
        let node_id = if self.is_head() {
            let state = self.state.read().await;
            if let BackendState::Head(ref head_state) = *state {
                head_state.actors.get(actor_id).copied()?
            } else {
                return None;
            }
        } else {
            // Worker: return from cache
            let state = self.state.read().await;
            if let BackendState::Worker(ref worker_state) = *state {
                worker_state.actors.get(actor_id).copied()?
            } else {
                return None;
            }
        };

        self.get_member(&node_id).await
    }

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    fn start(&self, cancel: CancellationToken) {
        if self.is_head() {
            // Head node: start heartbeat timeout detection
            let state = self.state.clone();
            let config = self.config.clone();
            tokio::spawn(async move {
                heartbeat_timeout_loop(state, config, cancel).await;
            });
        } else {
            // Worker node: start sync and heartbeat loops
            let sync_backend = Arc::new(self.clone());
            let heartbeat_backend = sync_backend.clone();
            let cancel_sync = cancel.clone();

            tokio::spawn(async move {
                worker_sync_loop(sync_backend, cancel_sync).await;
            });
            tokio::spawn(async move {
                worker_heartbeat_loop(heartbeat_backend, cancel).await;
            });
        }
    }

    fn local_node(&self) -> &NodeId {
        &self.local_node
    }

    fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ============================================================================
// Background Tasks
// ============================================================================

/// Head node: heartbeat timeout detection loop
async fn heartbeat_timeout_loop(
    state: Arc<RwLock<BackendState>>,
    config: HeadNodeConfig,
    cancel: CancellationToken,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                let timeout_ms = config.heartbeat_timeout.as_millis() as u64;
                let mut state_guard = state.write().await;
                if let BackendState::Head(ref mut head_state) = *state_guard {
                    let removed = head_state.remove_stale_nodes(timeout_ms);
                    if !removed.is_empty() {
                        tracing::info!(removed_count = removed.len(), "Removed stale nodes");
                    }
                }
            }
            _ = cancel.cancelled() => {
                tracing::info!("Heartbeat timeout loop shutting down");
                break;
            }
        }
    }
}

/// Worker node: sync loop
async fn worker_sync_loop(backend: Arc<HeadNodeBackend>, cancel: CancellationToken) {
    let mut interval = tokio::time::interval(backend.config.sync_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Sync from head
                if let Err(e) = backend.sync_from_head().await {
                    tracing::error!(error = %e, "Failed to sync from head node");
                    // Fast fail: if we can't sync, we should stop
                    cancel.cancel();
                    break;
                }
                // Process pending sync operations
                if let Err(e) = backend.process_pending_sync().await {
                    tracing::warn!(error = %e, "Failed to process pending sync");
                }
            }
            _ = cancel.cancelled() => {
                tracing::info!("Sync loop shutting down");
                break;
            }
        }
    }
}

/// Worker node: heartbeat loop
async fn worker_heartbeat_loop(backend: Arc<HeadNodeBackend>, cancel: CancellationToken) {
    let mut interval = tokio::time::interval(backend.config.heartbeat_interval);

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if let Err(e) = backend.send_heartbeat().await {
                    tracing::error!(error = %e, "Failed to send heartbeat to head node");
                    // Fast fail: if we can't heartbeat, we should stop
                    cancel.cancel();
                    break;
                }
            }
            _ = cancel.cancelled() => {
                tracing::info!("Heartbeat loop shutting down");
                break;
            }
        }
    }
}

// ============================================================================
// HTTP API Messages
// ============================================================================

#[derive(Serialize, Deserialize)]
struct RegisterNodeRequest {
    node_id: NodeId,
    addr: SocketAddr,
}

#[derive(Serialize, Deserialize)]
struct HeartbeatRequest {
    node_id: NodeId,
}

#[derive(Serialize, Deserialize)]
struct RegisterNamedActorRequest {
    path: ActorPath,
    node_id: NodeId,
    actor_id: Option<ActorId>,
    metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize)]
struct UnregisterNamedActorRequest {
    path: ActorPath,
    node_id: NodeId,
}

#[derive(Serialize, Deserialize)]
pub struct RegisterActorRequest {
    pub actor_id: ActorId,
    pub node_id: NodeId,
}

#[derive(Serialize, Deserialize)]
pub struct UnregisterActorRequest {
    pub actor_id: ActorId,
    pub node_id: NodeId,
}

/// Response for sync request from head node
#[derive(Serialize, Deserialize)]
pub struct SyncResponse {
    /// All registered members
    pub members: Vec<MemberInfo>,
    /// All named actors
    pub named_actors: Vec<NamedActorInfo>,
    /// Actor to node mappings
    pub actors: Vec<(ActorId, NodeId)>,
}
