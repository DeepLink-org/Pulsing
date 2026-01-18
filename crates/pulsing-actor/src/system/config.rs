//! Configuration types for the Actor System

use crate::actor::{NodeId, DEFAULT_MAILBOX_SIZE};
use crate::cluster::{GossipConfig, HeadNodeConfig};
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

    /// Head node mode: if true, this node acts as head node
    pub is_head_node: bool,

    /// Head node address: if Some, this node is a worker connecting to the head
    pub head_addr: Option<SocketAddr>,

    /// Head node configuration (only used when head node mode is enabled)
    pub head_node_config: Option<HeadNodeConfig>,
}

/// Default bind address for standalone mode (any interface, OS-assigned port)
const DEFAULT_BIND_ADDR: std::net::SocketAddr =
    std::net::SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::new(0, 0, 0, 0)), 0);

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            addr: DEFAULT_BIND_ADDR,
            seed_nodes: Vec::new(),
            gossip_config: GossipConfig::default(),
            http2_config: Http2Config::default(),
            default_mailbox_capacity: DEFAULT_MAILBOX_SIZE,
            is_head_node: false,
            head_addr: None,
            head_node_config: None,
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

    /// Enable head node mode
    pub fn with_head_node(mut self) -> Self {
        self.is_head_node = true;
        self.head_addr = None; // Clear head_addr if set
        self
    }

    /// Set head node address (makes this a worker node)
    pub fn with_head_addr(mut self, addr: SocketAddr) -> Self {
        self.head_addr = Some(addr);
        self.is_head_node = false; // Clear is_head_node if set
        self
    }

    /// Set head node configuration
    pub fn with_head_node_config(mut self, config: HeadNodeConfig) -> Self {
        self.head_node_config = Some(config);
        self
    }
}

// ============================================================================
// ActorSystem Builder
// ============================================================================

/// Builder for creating ActorSystem with fluent API
///
/// # Example
///
/// ```rust,ignore
/// // Standalone mode (simplest)
/// let system = ActorSystem::builder().build().await?;
///
/// // With custom address
/// let system = ActorSystem::builder()
///     .addr("0.0.0.0:8000")
///     .build()
///     .await?;
///
/// // Cluster mode with seeds
/// let system = ActorSystem::builder()
///     .addr("0.0.0.0:8000")
///     .seeds(["192.168.1.1:8000", "192.168.1.2:8000"])
///     .mailbox_capacity(512)
///     .build()
///     .await?;
/// ```
#[derive(Default)]
pub struct ActorSystemBuilder {
    /// Bind address (stored as Result for deferred error handling)
    addr: Option<Result<SocketAddr, String>>,
    /// Seed nodes (stored as Results for deferred error handling)
    seeds: Vec<Result<SocketAddr, String>>,
    /// Mailbox capacity
    mailbox_capacity: Option<usize>,
    /// Gossip configuration
    gossip_config: Option<GossipConfig>,
    /// HTTP/2 configuration
    http2_config: Option<Http2Config>,
    /// Head node mode
    is_head_node: bool,
    /// Head node address (if set, makes this a worker)
    head_addr: Option<Result<SocketAddr, String>>,
    /// Head node configuration
    head_node_config: Option<HeadNodeConfig>,
}

impl ActorSystemBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the bind address
    ///
    /// Accepts `&str`, `String`, or `SocketAddr`.
    /// Address parsing errors are deferred to `build()`.
    pub fn addr(mut self, addr: impl Into<AddrInput>) -> Self {
        self.addr = Some(addr.into().0);
        self
    }

    /// Add seed nodes for cluster joining
    ///
    /// Accepts `[&str]`, `Vec<String>`, or `Vec<SocketAddr>`.
    /// Address parsing errors are deferred to `build()`.
    pub fn seeds(mut self, seeds: impl IntoIterator<Item = impl Into<AddrInput>>) -> Self {
        self.seeds = seeds.into_iter().map(|s| s.into().0).collect();
        self
    }

    /// Set default mailbox capacity for actors
    pub fn mailbox_capacity(mut self, capacity: usize) -> Self {
        self.mailbox_capacity = Some(capacity);
        self
    }

    /// Enable TLS with passphrase
    #[cfg(feature = "tls")]
    pub fn tls(mut self, passphrase: &str) -> anyhow::Result<Self> {
        let http2_config = self
            .http2_config
            .take()
            .unwrap_or_default()
            .with_tls(passphrase)?;
        self.http2_config = Some(http2_config);
        Ok(self)
    }

    /// Set gossip configuration
    pub fn gossip(mut self, config: GossipConfig) -> Self {
        self.gossip_config = Some(config);
        self
    }

    /// Enable head node mode
    pub fn head_node(mut self) -> Self {
        self.is_head_node = true;
        self.head_addr = None; // Clear head_addr if set
        self
    }

    /// Set head node address (makes this a worker node)
    ///
    /// Accepts `&str`, `String`, or `SocketAddr`.
    /// Address parsing errors are deferred to `build()`.
    pub fn head_addr(mut self, addr: impl Into<AddrInput>) -> Self {
        self.head_addr = Some(addr.into().0);
        self.is_head_node = false; // Clear is_head_node if set
        self
    }

    /// Set head node configuration
    pub fn head_node_config(mut self, config: HeadNodeConfig) -> Self {
        self.head_node_config = Some(config);
        self
    }

    /// Build the ActorSystem
    ///
    /// Returns an error if any address parsing failed.
    pub async fn build(self) -> anyhow::Result<Arc<crate::system::ActorSystem>> {
        // Parse bind address (use default if not specified)
        let addr = match self.addr {
            Some(Ok(addr)) => addr,
            Some(Err(invalid)) => {
                return Err(anyhow::anyhow!("Invalid bind address: {}", invalid));
            }
            None => DEFAULT_BIND_ADDR,
        };

        // Parse seed nodes
        let mut seed_nodes = Vec::with_capacity(self.seeds.len());
        for (i, seed) in self.seeds.into_iter().enumerate() {
            match seed {
                Ok(addr) => seed_nodes.push(addr),
                Err(invalid) => {
                    return Err(anyhow::anyhow!(
                        "Invalid seed address at index {}: {}",
                        i,
                        invalid
                    ));
                }
            }
        }

        // Parse head node address if specified
        let head_addr = match self.head_addr {
            Some(Ok(addr)) => Some(addr),
            Some(Err(invalid)) => {
                return Err(anyhow::anyhow!("Invalid head node address: {}", invalid));
            }
            None => None,
        };

        let config = SystemConfig {
            addr,
            seed_nodes,
            gossip_config: self.gossip_config.unwrap_or_default(),
            http2_config: self.http2_config.unwrap_or_default(),
            default_mailbox_capacity: self.mailbox_capacity.unwrap_or(DEFAULT_MAILBOX_SIZE),
            is_head_node: self.is_head_node,
            head_addr,
            head_node_config: self.head_node_config,
        };

        crate::system::ActorSystem::new(config).await
    }
}

/// Helper type for flexible address input (defers parsing errors)
pub struct AddrInput(Result<SocketAddr, String>);

impl From<SocketAddr> for AddrInput {
    fn from(addr: SocketAddr) -> Self {
        AddrInput(Ok(addr))
    }
}

impl From<&str> for AddrInput {
    fn from(s: &str) -> Self {
        AddrInput(s.parse().map_err(|_| s.to_string()))
    }
}

impl From<String> for AddrInput {
    fn from(s: String) -> Self {
        AddrInput(s.parse().map_err(|_| s.clone()))
    }
}

impl From<&&str> for AddrInput {
    fn from(s: &&str) -> Self {
        AddrInput(s.parse().map_err(|_| s.to_string()))
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
