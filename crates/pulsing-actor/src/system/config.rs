//! Configuration types for the Actor System

use crate::actor::{NodeId, DEFAULT_MAILBOX_SIZE};
use crate::cluster::GossipConfig;
use crate::policies::LoadBalancingPolicy;
use crate::supervision::SupervisionSpec;
use crate::transport::Http2Config;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

/// Actor System configuration
#[derive(Clone, Debug)]
pub struct SystemConfig {
    /// HTTP/2 address for all communication (actors + gossip)
    pub addr: SocketAddr,

    /// Seed nodes to join (HTTP/2 addresses)
    pub seed_nodes: Vec<SocketAddr>,

    /// Gossip configuration
    pub gossip_config: GossipConfig,

    /// HTTP/2 transport configuration
    pub http2_config: Http2Config,

    /// Default mailbox capacity for all actors
    pub default_mailbox_capacity: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:0".parse().unwrap(),
            seed_nodes: Vec::new(),
            gossip_config: GossipConfig::default(),
            http2_config: Http2Config::default(),
            default_mailbox_capacity: DEFAULT_MAILBOX_SIZE,
        }
    }
}

impl SystemConfig {
    /// Create config for a standalone node (no cluster)
    pub fn standalone() -> Self {
        Self::default()
    }

    /// Create config with specific address
    pub fn with_addr(addr: SocketAddr) -> Self {
        Self {
            addr,
            ..Default::default()
        }
    }

    /// Add seed nodes for cluster joining
    pub fn with_seeds(mut self, seeds: Vec<SocketAddr>) -> Self {
        self.seed_nodes = seeds;
        self
    }

    /// Set default mailbox capacity
    pub fn with_mailbox_capacity(mut self, capacity: usize) -> Self {
        self.default_mailbox_capacity = capacity;
        self
    }

    /// Enable TLS with passphrase-derived certificates
    ///
    /// All nodes using the same passphrase will be able to communicate securely.
    /// The passphrase is used to derive a shared CA certificate, enabling
    /// automatic mutual TLS authentication.
    #[cfg(feature = "tls")]
    pub fn with_tls(mut self, passphrase: &str) -> anyhow::Result<Self> {
        self.http2_config = self.http2_config.with_tls(passphrase)?;
        Ok(self)
    }

    /// Check if TLS is enabled
    pub fn is_tls_enabled(&self) -> bool {
        self.http2_config.is_tls_enabled()
    }
}

/// Options for spawning an actor
#[derive(Default, Clone, Debug)]
pub struct SpawnOptions {
    /// Override mailbox capacity (None = use system default)
    pub mailbox_capacity: Option<usize>,
    /// Whether this actor is public (can be resolved by name across cluster)
    pub public: bool,
    /// Supervision specification (restart policy)
    pub supervision: SupervisionSpec,
    /// Actor metadata (e.g., Python class, module, file path)
    pub metadata: HashMap<String, String>,
}

impl SpawnOptions {
    /// Create new spawn options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set mailbox capacity override
    pub fn mailbox_capacity(mut self, capacity: usize) -> Self {
        self.mailbox_capacity = Some(capacity);
        self
    }

    /// Set whether actor is public
    pub fn public(mut self, public: bool) -> Self {
        self.public = public;
        self
    }

    /// Set supervision specification
    pub fn supervision(mut self, supervision: SupervisionSpec) -> Self {
        self.supervision = supervision;
        self
    }

    /// Set actor metadata
    pub fn metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Options for resolving named actors
#[derive(Clone, Default)]
pub struct ResolveOptions {
    /// Target node ID (if specified, skip load balancing)
    pub node_id: Option<NodeId>,
    /// Load balancing policy (None = use system default)
    pub policy: Option<Arc<dyn LoadBalancingPolicy>>,
    /// Only select Alive nodes (default: true)
    pub filter_alive: bool,
}

impl std::fmt::Debug for ResolveOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolveOptions")
            .field("node_id", &self.node_id)
            .field("policy", &self.policy.as_ref().map(|p| p.name()))
            .field("filter_alive", &self.filter_alive)
            .finish()
    }
}

impl ResolveOptions {
    /// Create new resolve options with defaults
    pub fn new() -> Self {
        Self {
            filter_alive: true,
            ..Default::default()
        }
    }

    /// Set target node ID (bypasses load balancing)
    pub fn node_id(mut self, node_id: NodeId) -> Self {
        self.node_id = Some(node_id);
        self
    }

    /// Set load balance policy
    pub fn policy(mut self, policy: Arc<dyn LoadBalancingPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Set whether to filter only alive nodes
    pub fn filter_alive(mut self, filter: bool) -> Self {
        self.filter_alive = filter;
        self
    }
}
