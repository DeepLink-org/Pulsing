//! Configuration types for the Actor System.
//!
//! Core types:
//! - [`SystemConfig`]
//! - [`ActorSystemBuilder`]
//! - [`SpawnOptions`]
//! - [`ResolveOptions`]

use crate::actor::{NodeId, DEFAULT_MAILBOX_SIZE};
use crate::cluster::{GossipConfig, HeadNodeConfig};
use crate::error::{PulsingError, Result, RuntimeError};
use crate::policies::LoadBalancingPolicy;
use crate::supervision::SupervisionSpec;
use crate::transport::Http2Config;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

/// Minimum mailbox capacity (prevents performance issues)
const MIN_MAILBOX_CAPACITY: usize = 16;

/// Maximum mailbox capacity (prevents memory exhaustion)
const MAX_MAILBOX_CAPACITY: usize = 1_000_000;

/// Actor System configuration
///
/// This struct holds all configuration options for the actor system.
/// Use the builder pattern via [`ActorSystem::builder()`](crate::system::ActorSystem::builder)
/// for a more ergonomic API.
///
/// # Validation
///
/// Call [`validate()`](Self::validate) to check configuration validity before use.
/// The builder automatically validates during `build()`.
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
    pub fn with_tls(mut self, passphrase: &str) -> Result<Self> {
        self.http2_config = self
            .http2_config
            .with_tls(passphrase)
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())))?;
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

    /// Validate the configuration
    ///
    /// Returns a list of validation errors, or an empty list if valid.
    /// The builder calls this automatically during `build()`.
    ///
    /// # Validation Rules
    /// - Mailbox capacity must be between 16 and 1,000,000
    /// - Cannot be both head node and have head_addr set
    /// - Seed nodes should not be empty when head_addr is not set (for cluster mode)
    pub fn validate(&self) -> Vec<ConfigValidationError> {
        let mut errors = Vec::new();

        // Validate mailbox capacity
        if self.default_mailbox_capacity < MIN_MAILBOX_CAPACITY {
            errors.push(ConfigValidationError::MailboxTooSmall {
                value: self.default_mailbox_capacity,
                min: MIN_MAILBOX_CAPACITY,
            });
        }
        if self.default_mailbox_capacity > MAX_MAILBOX_CAPACITY {
            errors.push(ConfigValidationError::MailboxTooLarge {
                value: self.default_mailbox_capacity,
                max: MAX_MAILBOX_CAPACITY,
            });
        }

        // Validate head node configuration
        if self.is_head_node && self.head_addr.is_some() {
            errors.push(ConfigValidationError::ConflictingHeadNodeConfig);
        }

        errors
    }

    /// Check if configuration is valid
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }
}

/// Configuration validation error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigValidationError {
    /// Mailbox capacity is too small
    MailboxTooSmall { value: usize, min: usize },
    /// Mailbox capacity is too large
    MailboxTooLarge { value: usize, max: usize },
    /// Conflicting head node configuration (both is_head_node and head_addr set)
    ConflictingHeadNodeConfig,
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MailboxTooSmall { value, min } => {
                write!(
                    f,
                    "Mailbox capacity {} is too small (minimum: {})",
                    value, min
                )
            }
            Self::MailboxTooLarge { value, max } => {
                write!(
                    f,
                    "Mailbox capacity {} is too large (maximum: {})",
                    value, max
                )
            }
            Self::ConflictingHeadNodeConfig => {
                write!(
                    f,
                    "Cannot set both head_node mode and head_addr (conflicting options)"
                )
            }
        }
    }
}

impl std::error::Error for ConfigValidationError {}

// ============================================================================
// ActorSystem Builder
// ============================================================================

/// Builder for creating ActorSystem with fluent API
#[derive(Default)]
pub struct ActorSystemBuilder {
    /// Bind address (stored as Result for deferred error handling)
    addr: Option<std::result::Result<SocketAddr, String>>,
    /// Seed nodes (stored as Results for deferred error handling)
    seeds: Vec<std::result::Result<SocketAddr, String>>,
    /// Mailbox capacity
    mailbox_capacity: Option<usize>,
    /// Gossip configuration
    gossip_config: Option<GossipConfig>,
    /// HTTP/2 configuration
    http2_config: Option<Http2Config>,
    /// Head node mode
    is_head_node: bool,
    /// Head node address (if set, makes this a worker)
    head_addr: Option<std::result::Result<SocketAddr, String>>,
    /// Head node configuration
    head_node_config: Option<HeadNodeConfig>,
}

impl ActorSystemBuilder {
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
    pub fn tls(mut self, passphrase: &str) -> Result<Self> {
        let http2_config = self
            .http2_config
            .take()
            .unwrap_or_default()
            .with_tls(passphrase)
            .map_err(|e| PulsingError::from(RuntimeError::Other(e.to_string())))?;
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
    /// Returns an error if any address parsing or validation failed.
    pub async fn build(self) -> crate::error::Result<Arc<crate::system::ActorSystem>> {
        let addr =
            Self::parse_optional_addr("bind address", self.addr)?.unwrap_or(DEFAULT_BIND_ADDR);

        let seed_nodes = Self::parse_addr_list("seed address", self.seeds)?;

        let head_addr = Self::parse_optional_addr("head node address", self.head_addr)?;

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

        Self::validate_config(&config)?;

        crate::system::ActorSystem::new(config).await
    }

    fn parse_optional_addr(
        label: &str,
        input: Option<std::result::Result<SocketAddr, String>>,
    ) -> Result<Option<SocketAddr>> {
        match input {
            Some(Ok(addr)) => Ok(Some(addr)),
            Some(Err(invalid)) => Err(PulsingError::from(RuntimeError::Other(format!(
                "Invalid {}: {}",
                label, invalid
            )))),
            None => Ok(None),
        }
    }

    fn parse_addr_list(
        label: &str,
        seeds: Vec<std::result::Result<SocketAddr, String>>,
    ) -> Result<Vec<SocketAddr>> {
        let mut addrs = Vec::with_capacity(seeds.len());
        for (i, seed) in seeds.into_iter().enumerate() {
            match seed {
                Ok(addr) => addrs.push(addr),
                Err(invalid) => {
                    return Err(PulsingError::from(RuntimeError::Other(format!(
                        "Invalid {} at index {}: {}",
                        label, i, invalid
                    ))));
                }
            }
        }
        Ok(addrs)
    }

    fn validate_config(config: &SystemConfig) -> Result<()> {
        let errors = config.validate();
        if errors.is_empty() {
            return Ok(());
        }
        let error_msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        Err(PulsingError::from(RuntimeError::Other(format!(
            "Configuration validation failed:\n  - {}",
            error_msgs.join("\n  - ")
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_config_default() {
        let config = SystemConfig::default();
        assert!(config.is_valid());
        assert_eq!(config.default_mailbox_capacity, DEFAULT_MAILBOX_SIZE);
        assert!(!config.is_head_node);
        assert!(config.head_addr.is_none());
    }

    #[test]
    fn test_system_config_standalone() {
        let config = SystemConfig::standalone();
        assert!(config.is_valid());
    }

    #[test]
    fn test_system_config_with_addr() {
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let config = SystemConfig::with_addr(addr);
        assert_eq!(config.addr, addr);
        assert!(config.is_valid());
    }

    #[test]
    fn test_system_config_validation_mailbox_too_small() {
        let config = SystemConfig {
            default_mailbox_capacity: 1,
            ..Default::default()
        };
        let errors = config.validate();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ConfigValidationError::MailboxTooSmall { .. }
        ));
    }

    #[test]
    fn test_system_config_validation_mailbox_too_large() {
        let config = SystemConfig {
            default_mailbox_capacity: 10_000_000,
            ..Default::default()
        };
        let errors = config.validate();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ConfigValidationError::MailboxTooLarge { .. }
        ));
    }

    #[test]
    fn test_system_config_validation_conflicting_head_node() {
        let addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let config = SystemConfig {
            is_head_node: true,
            head_addr: Some(addr),
            ..Default::default()
        };
        let errors = config.validate();
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ConfigValidationError::ConflictingHeadNodeConfig
        ));
    }

    #[test]
    fn test_system_config_builder_methods() {
        let config = SystemConfig::standalone()
            .with_mailbox_capacity(256)
            .with_head_node();

        assert_eq!(config.default_mailbox_capacity, 256);
        assert!(config.is_head_node);
        assert!(config.is_valid());
    }

    #[test]
    fn test_spawn_options_default() {
        let options = SpawnOptions::default();
        assert!(options.mailbox_capacity.is_none());
        assert!(options.metadata.is_empty());
    }

    #[test]
    fn test_spawn_options_builder() {
        let options = SpawnOptions::default()
            .mailbox_capacity(512)
            .metadata([("key".to_string(), "value".to_string())].into());

        assert_eq!(options.mailbox_capacity, Some(512));
        assert_eq!(options.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_resolve_options_default() {
        let options = ResolveOptions::default();
        assert!(options.node_id.is_none());
        assert!(options.policy.is_none());
        assert!(options.filter_alive);
    }

    #[test]
    fn test_resolve_options_builder() {
        let node_id = NodeId::new(123);
        let options = ResolveOptions::default()
            .node_id(node_id)
            .filter_alive(false);

        assert_eq!(options.node_id, Some(node_id));
        assert!(!options.filter_alive);
    }

    #[test]
    fn test_config_validation_error_display() {
        let err = ConfigValidationError::MailboxTooSmall { value: 5, min: 16 };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("16"));

        let err = ConfigValidationError::ConflictingHeadNodeConfig;
        assert!(err.to_string().contains("head_node"));
    }

    // --- 配置解析 ---

    #[test]
    fn test_config_with_seeds() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let config = SystemConfig::standalone().with_seeds(vec![addr]);
        assert_eq!(config.seed_nodes.len(), 1);
        assert_eq!(config.seed_nodes[0], addr);
    }

    #[test]
    fn test_config_with_head_node_and_head_addr() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let config = SystemConfig::standalone().with_head_node();
        assert!(config.is_head_node);
        assert!(config.head_addr.is_none());

        let config = SystemConfig::standalone().with_head_addr(addr);
        assert!(!config.is_head_node);
        assert_eq!(config.head_addr, Some(addr));
    }

    #[tokio::test]
    async fn test_builder_invalid_addr_parse() {
        let result = ActorSystemBuilder::default()
            .addr("not-a-valid-address")
            .build()
            .await;
        let err = match result {
            Ok(_) => panic!("expected build to fail"),
            Err(e) => e,
        };
        assert!(err.to_string().to_lowercase().contains("invalid"));
    }

    #[tokio::test]
    async fn test_builder_invalid_seed_parse() {
        let result = ActorSystemBuilder::default()
            .seeds(vec!["127.0.0.1:0", "invalid-seed"])
            .build()
            .await;
        let err = match result {
            Ok(_) => panic!("expected build to fail"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("Invalid") && msg.contains("index"));
    }

    #[tokio::test]
    async fn test_builder_validation_mailbox_too_small() {
        let result = ActorSystemBuilder::default()
            .addr("127.0.0.1:0")
            .mailbox_capacity(1)
            .build()
            .await;
        let err = match result {
            Ok(_) => panic!("expected validation to fail"),
            Err(e) => e,
        };
        assert!(err.to_string().to_lowercase().contains("mailbox"));
    }

    #[tokio::test]
    async fn test_builder_validation_mailbox_too_large() {
        let result = ActorSystemBuilder::default()
            .addr("127.0.0.1:0")
            .mailbox_capacity(10_000_000)
            .build()
            .await;
        let err = match result {
            Ok(_) => panic!("expected validation to fail"),
            Err(e) => e,
        };
        assert!(err.to_string().to_lowercase().contains("mailbox"));
    }

    #[test]
    fn test_validation_multiple_errors() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let config = SystemConfig {
            default_mailbox_capacity: 1,
            is_head_node: true,
            head_addr: Some(addr),
            ..Default::default()
        };
        let errors = config.validate();
        assert_eq!(errors.len(), 2);
        let has_mailbox = errors
            .iter()
            .any(|e| matches!(e, ConfigValidationError::MailboxTooSmall { .. }));
        let has_conflict = errors
            .iter()
            .any(|e| matches!(e, ConfigValidationError::ConflictingHeadNodeConfig));
        assert!(has_mailbox);
        assert!(has_conflict);
    }

    #[tokio::test]
    async fn test_builder_valid_build() {
        let result = ActorSystemBuilder::default()
            .addr("127.0.0.1:0")
            .build()
            .await;
        let system = result.unwrap();
        system.shutdown().await.unwrap();
    }
}

/// Helper type for flexible address input (defers parsing errors)
pub struct AddrInput(std::result::Result<SocketAddr, String>);

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
    /// Supervision specification (restart policy)
    pub supervision: SupervisionSpec,
    /// Actor metadata (e.g., Python class, module, file path)
    pub metadata: HashMap<String, String>,
}

impl SpawnOptions {
    /// Set mailbox capacity override
    pub fn mailbox_capacity(mut self, capacity: usize) -> Self {
        self.mailbox_capacity = Some(capacity);
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
#[derive(Clone)]
pub struct ResolveOptions {
    /// Target node ID (if specified, skip load balancing)
    pub node_id: Option<NodeId>,
    /// Load balancing policy (None = use system default)
    pub policy: Option<Arc<dyn LoadBalancingPolicy>>,
    /// Only select Alive nodes (default: true)
    pub filter_alive: bool,
}

impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            node_id: None,
            policy: None,
            filter_alive: true,
        }
    }
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
