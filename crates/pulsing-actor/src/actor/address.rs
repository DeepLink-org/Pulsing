//! Actor addressing - URI-based actor addressing scheme
//!
//! This module implements the actor addressing scheme as defined in the design document.
//!
//! ## Address Types
//!
//! 1. Named Actor Service Address: `actor:///namespace/path/name`
//! 2. Named Actor Instance Address: `actor:///namespace/path/name@node_id`
//! 3. Global Actor Address: `actor://node_id/actor_id`
//! 4. Local Reference: `actor://0/actor_id` (node_id=0 means local)

use super::traits::NodeId;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

/// Address parsing error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddressParseError {
    /// Invalid URI scheme (must be "actor://")
    InvalidScheme,
    /// Invalid address format
    InvalidFormat,
    /// Missing namespace in named actor path
    MissingNamespace,
    /// Empty path segment
    EmptySegment,
    /// Invalid character in path
    InvalidCharacter,
    /// Path exceeds maximum length
    PathTooLong,
    /// Single segment exceeds maximum length
    SegmentTooLong,
    /// Attempted to use a reserved system namespace
    ReservedNamespace,
}

impl fmt::Display for AddressParseError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScheme => write!(f, "Invalid scheme, expected 'actor://'"),
            Self::InvalidFormat => write!(f, "Invalid address format"),
            Self::MissingNamespace => {
                write!(f, "Named actor path must have at least namespace/name")
            }
            Self::EmptySegment => write!(f, "Path segment cannot be empty"),
            Self::InvalidCharacter => write!(f, "Invalid character in path"),
            Self::PathTooLong => write!(
                f,
                "Path exceeds maximum length of {} characters",
                ActorPath::MAX_PATH_LENGTH
            ),
            Self::SegmentTooLong => write!(
                f,
                "Segment exceeds maximum length of {} characters",
                ActorPath::MAX_SEGMENT_LENGTH
            ),
            Self::ReservedNamespace => {
                write!(f, "Cannot use reserved system namespace for user actors")
            }
        }
    }
}

impl std::error::Error for AddressParseError {}

/// Actor path for named actors (namespace + hierarchical path + name)
///
/// A path must have at least two segments: namespace and name.
/// Additional segments can be used for logical grouping.
///
/// Examples:
/// - `services/llm/router` (namespace: services, name: router)
/// - `workers/inference/pool` (namespace: workers, name: pool)
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActorPath {
    /// Path segments, e.g., ["services", "llm", "router"]
    segments: Vec<String>,
}

impl ActorPath {
    /// Reserved system namespaces that cannot be used by user actors
    pub const SYSTEM_NAMESPACES: &'static [&'static str] = &["system"];

    /// Maximum total path length (prevents DoS and memory issues)
    pub const MAX_PATH_LENGTH: usize = 256;

    /// Maximum single segment length
    pub const MAX_SEGMENT_LENGTH: usize = 64;

    /// Create a new actor path from a string
    ///
    /// The path must have at least two segments (namespace/name).
    ///
    /// # Validation Rules
    /// - Path cannot be empty
    /// - Path cannot exceed 256 characters
    /// - Each segment cannot exceed 64 characters
    /// - Each segment can only contain alphanumeric characters, `_`, and `-`
    /// - Path must have at least two segments (namespace/name)
    /// - User code cannot use reserved system namespaces (use `new_system` for internal use)
    ///
    /// # Examples
    /// ```
    /// use pulsing_actor::actor::ActorPath;
    ///
    /// let path = ActorPath::new("services/llm/router").unwrap();
    /// assert_eq!(path.namespace(), "services");
    /// assert_eq!(path.name(), "router");
    ///
    /// // These will fail:
    /// // ActorPath::new("system/internal").unwrap(); // Reserved namespace
    /// // ActorPath::new("a".repeat(300)).unwrap();   // Too long
    /// ```
    pub fn new(path: impl AsRef<str>) -> Result<Self, AddressParseError> {
        let path = path.as_ref().trim_matches('/');

        // Check total path length
        if path.len() > Self::MAX_PATH_LENGTH {
            return Err(AddressParseError::PathTooLong);
        }

        if path.is_empty() {
            return Err(AddressParseError::MissingNamespace);
        }

        let segments: Vec<String> = path.split('/').map(|s| s.trim().to_string()).collect();

        // Validate segments
        for segment in &segments {
            if segment.is_empty() {
                return Err(AddressParseError::EmptySegment);
            }
            if segment.len() > Self::MAX_SEGMENT_LENGTH {
                return Err(AddressParseError::SegmentTooLong);
            }
            if !Self::is_valid_segment(segment) {
                return Err(AddressParseError::InvalidCharacter);
            }
        }

        // Must have at least namespace/name
        if segments.len() < 2 {
            return Err(AddressParseError::MissingNamespace);
        }

        // Check for reserved system namespaces (user code cannot use these)
        if Self::SYSTEM_NAMESPACES.contains(&segments[0].as_str()) {
            return Err(AddressParseError::ReservedNamespace);
        }

        Ok(Self { segments })
    }

    /// Create a system actor path (bypasses reserved namespace check)
    ///
    /// # Warning
    /// This method is intended for framework internals (actor system, Python bindings).
    /// Application code should use `new()` which enforces namespace restrictions.
    ///
    /// # Use Cases
    /// - Creating paths for built-in system actors like `system/core`
    /// - Python bindings for `PythonActorService` at `system/python_actor_service`
    #[doc(hidden)]
    pub fn new_system(path: impl AsRef<str>) -> Result<Self, AddressParseError> {
        let path = path.as_ref().trim_matches('/');

        // Check total path length
        if path.len() > Self::MAX_PATH_LENGTH {
            return Err(AddressParseError::PathTooLong);
        }

        if path.is_empty() {
            return Err(AddressParseError::MissingNamespace);
        }

        let segments: Vec<String> = path.split('/').map(|s| s.trim().to_string()).collect();

        // Validate segments
        for segment in &segments {
            if segment.is_empty() {
                return Err(AddressParseError::EmptySegment);
            }
            if segment.len() > Self::MAX_SEGMENT_LENGTH {
                return Err(AddressParseError::SegmentTooLong);
            }
            if !Self::is_valid_segment(segment) {
                return Err(AddressParseError::InvalidCharacter);
            }
        }

        // Must have at least namespace/name
        if segments.len() < 2 {
            return Err(AddressParseError::MissingNamespace);
        }

        Ok(Self { segments })
    }

    /// Check if a segment contains only valid characters
    fn is_valid_segment(s: &str) -> bool {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    }

    /// Get the namespace (first segment)
    pub fn namespace(&self) -> &str {
        &self.segments[0]
    }

    /// Get the name (last segment)
    ///
    /// # Safety
    /// This will never panic because ActorPath::new() ensures at least 2 segments exist.
    pub fn name(&self) -> &str {
        // SAFETY: ActorPath invariant guarantees segments.len() >= 2
        // This is enforced in ActorPath::new() which is the only way to construct an ActorPath
        &self.segments[self.segments.len() - 1]
    }

    /// Get all segments
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Get the full path as a string
    pub fn as_str(&self) -> String {
        self.segments.join("/")
    }

    /// Check if this is a system namespace
    pub fn is_system(&self) -> bool {
        Self::SYSTEM_NAMESPACES.contains(&self.namespace())
    }

    /// Get the parent path (without the last segment)
    pub fn parent(&self) -> Option<String> {
        if self.segments.len() > 2 {
            Some(self.segments[..self.segments.len() - 1].join("/"))
        } else {
            None
        }
    }

    /// Create a child path by appending a segment
    pub fn child(&self, name: impl AsRef<str>) -> Result<Self, AddressParseError> {
        let mut segments = self.segments.clone();
        let name = name.as_ref();

        if !Self::is_valid_segment(name) {
            return Err(AddressParseError::InvalidCharacter);
        }

        segments.push(name.to_string());
        Ok(Self { segments })
    }
}

impl fmt::Display for ActorPath {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl TryFrom<&str> for ActorPath {
    type Error = AddressParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl TryFrom<String> for ActorPath {
    type Error = AddressParseError;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

/// Trait for types that can be converted to ActorPath
pub trait IntoActorPath {
    fn into_actor_path(self) -> anyhow::Result<ActorPath>;
}

impl IntoActorPath for &str {
    fn into_actor_path(self) -> anyhow::Result<ActorPath> {
        ActorPath::new(self).map_err(Into::into)
    }
}

impl IntoActorPath for String {
    fn into_actor_path(self) -> anyhow::Result<ActorPath> {
        ActorPath::new(self).map_err(Into::into)
    }
}

impl IntoActorPath for ActorPath {
    fn into_actor_path(self) -> anyhow::Result<ActorPath> {
        Ok(self)
    }
}

impl IntoActorPath for &ActorPath {
    fn into_actor_path(self) -> anyhow::Result<ActorPath> {
        Ok(self.clone())
    }
}

/// Actor address - unified addressing for all actor types
///
/// Supports three address formats:
/// 1. Named service: `actor:///namespace/path/name` - load-balanced access to named actor
/// 2. Named instance: `actor:///namespace/path/name@node_id` - specific instance
/// 3. Global: `actor://node_id/actor_id` - direct address to any actor (node_id=0 means local)
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActorAddress {
    /// Named Actor - registered via Gossip, supports multiple instances
    /// Format: `actor:///namespace/path/name[@node_id]`
    Named {
        /// The actor path (namespace + hierarchical path)
        path: ActorPath,
        /// Optional instance locator (specific node)
        instance: Option<NodeId>,
    },

    /// Global Actor Address - direct addressing without Gossip registration
    /// Format: `actor://node_id/actor_id`
    Global {
        /// The node where the actor resides (0 = local)
        node_id: NodeId,
        /// The actor's local identifier
        actor_id: u64,
    },
}

impl ActorAddress {
    /// Parse an actor address from a URI string
    ///
    /// # Examples
    /// ```
    /// use pulsing_actor::actor::ActorAddress;
    ///
    /// // Named actor (service address)
    /// let addr = ActorAddress::parse("actor:///services/llm/router").unwrap();
    ///
    /// // Named actor (specific instance)
    /// let addr = ActorAddress::parse("actor:///services/llm/router@123").unwrap();
    ///
    /// // Global address
    /// let addr = ActorAddress::parse("actor://123/456").unwrap();
    ///
    /// // Local reference (node_id = 0)
    /// let addr = ActorAddress::parse("actor://0/456").unwrap();
    /// ```
    pub fn parse(uri: &str) -> Result<Self, AddressParseError> {
        if !uri.starts_with("actor://") {
            return Err(AddressParseError::InvalidScheme);
        }

        let rest = &uri[8..]; // Remove "actor://"

        if let Some(path_part) = rest.strip_prefix('/') {
            // Named actor: actor:///namespace/path/name[@node]
            if path_part.is_empty() {
                return Err(AddressParseError::MissingNamespace);
            }

            if let Some((path, node)) = path_part.rsplit_once('@') {
                // With instance specifier
                let node_id = node
                    .parse::<u64>()
                    .map_err(|_| AddressParseError::InvalidFormat)?;
                Ok(Self::Named {
                    path: ActorPath::new(path)?,
                    instance: Some(NodeId::new(node_id)),
                })
            } else {
                // Service address (no instance)
                Ok(Self::Named {
                    path: ActorPath::new(path_part)?,
                    instance: None,
                })
            }
        } else {
            // Global: actor://node_id/actor_id
            let (node_id_str, actor_id_str) = rest
                .split_once('/')
                .ok_or(AddressParseError::InvalidFormat)?;

            if node_id_str.is_empty() || actor_id_str.is_empty() {
                return Err(AddressParseError::InvalidFormat);
            }

            let node_id = node_id_str
                .parse::<u64>()
                .map_err(|_| AddressParseError::InvalidFormat)?;
            let actor_id = actor_id_str
                .parse::<u64>()
                .map_err(|_| AddressParseError::InvalidFormat)?;

            Ok(Self::Global {
                node_id: NodeId::new(node_id),
                actor_id,
            })
        }
    }

    /// Create a named actor service address
    pub fn named(path: ActorPath) -> Self {
        Self::Named {
            path,
            instance: None,
        }
    }

    /// Create a named actor instance address
    pub fn named_instance(path: ActorPath, node_id: NodeId) -> Self {
        Self::Named {
            path,
            instance: Some(node_id),
        }
    }

    /// Create a global actor address
    pub fn global(node_id: NodeId, actor_id: u64) -> Self {
        Self::Global { node_id, actor_id }
    }

    /// Create a local actor reference (node_id = 0)
    pub fn local(actor_id: u64) -> Self {
        Self::Global {
            node_id: NodeId::LOCAL,
            actor_id,
        }
    }

    /// Convert to URI string
    pub fn to_uri(&self) -> String {
        match self {
            Self::Named {
                path,
                instance: None,
            } => {
                format!("actor:///{}", path.as_str())
            }
            Self::Named {
                path,
                instance: Some(node),
            } => {
                format!("actor:///{}@{}", path.as_str(), node.0)
            }
            Self::Global { node_id, actor_id } => {
                format!("actor://{}/{}", node_id.0, actor_id)
            }
        }
    }

    /// Check if this is a local reference (node_id = 0)
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Global { node_id, .. } if node_id.is_local())
    }

    /// Check if this is a named actor address
    pub fn is_named(&self) -> bool {
        matches!(self, Self::Named { .. })
    }

    /// Check if this is a global address
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global { .. })
    }

    /// Resolve local node id to actual node ID
    pub fn resolve_local(self, current_node: NodeId) -> Self {
        match self {
            Self::Global { node_id, actor_id } if node_id.is_local() => Self::Global {
                node_id: current_node,
                actor_id,
            },
            other => other,
        }
    }

    /// Add instance specifier to a named address
    pub fn with_instance(self, node_id: NodeId) -> Self {
        match self {
            Self::Named { path, .. } => Self::Named {
                path,
                instance: Some(node_id),
            },
            other => other,
        }
    }

    /// Get the path for named actors
    pub fn path(&self) -> Option<&ActorPath> {
        match self {
            Self::Named { path, .. } => Some(path),
            _ => None,
        }
    }

    /// Get the node ID
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::Global { node_id, .. } => Some(*node_id),
            Self::Named { instance, .. } => *instance,
        }
    }

    /// Get the actor ID for global addresses
    pub fn actor_id(&self) -> Option<u64> {
        match self {
            Self::Global { actor_id, .. } => Some(*actor_id),
            _ => None,
        }
    }
}

impl fmt::Display for ActorAddress {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uri())
    }
}

impl TryFrom<&str> for ActorAddress {
    type Error = AddressParseError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_path_new() {
        let path = ActorPath::new("services/llm/router").unwrap();
        assert_eq!(path.namespace(), "services");
        assert_eq!(path.name(), "router");
        assert_eq!(path.segments().len(), 3);
        assert_eq!(path.as_str(), "services/llm/router");
    }

    #[test]
    fn test_actor_path_two_segments() {
        let path = ActorPath::new("workers/pool").unwrap();
        assert_eq!(path.namespace(), "workers");
        assert_eq!(path.name(), "pool");
    }

    #[test]
    fn test_actor_path_invalid() {
        // Single segment (missing namespace)
        assert!(ActorPath::new("single").is_err());

        // Empty path
        assert!(ActorPath::new("").is_err());

        // Empty segment
        assert!(ActorPath::new("services//router").is_err());
    }

    #[test]
    fn test_actor_path_system_namespace() {
        // System namespace is reserved - regular new() should fail
        assert!(matches!(
            ActorPath::new("system/cluster/monitor"),
            Err(AddressParseError::ReservedNamespace)
        ));

        // Use new_system for internal system actors
        let path = ActorPath::new_system("system/cluster/monitor").unwrap();
        assert!(path.is_system());

        // Regular namespaces work normally
        let path = ActorPath::new("services/api").unwrap();
        assert!(!path.is_system());
    }

    #[test]
    fn test_actor_path_length_limits() {
        // Path too long
        let long_path = format!("services/{}", "a".repeat(300));
        assert!(matches!(
            ActorPath::new(&long_path),
            Err(AddressParseError::PathTooLong)
        ));

        // Single segment too long
        let long_segment = format!("services/{}", "b".repeat(100));
        assert!(matches!(
            ActorPath::new(&long_segment),
            Err(AddressParseError::SegmentTooLong)
        ));

        // Valid length path
        let valid_path = format!("services/{}", "c".repeat(50));
        assert!(ActorPath::new(&valid_path).is_ok());
    }

    #[test]
    fn test_address_parse_named_service() {
        let addr = ActorAddress::parse("actor:///services/llm/router").unwrap();
        match addr {
            ActorAddress::Named { path, instance } => {
                assert_eq!(path.as_str(), "services/llm/router");
                assert!(instance.is_none());
            }
            _ => panic!("Expected Named address"),
        }
    }

    #[test]
    fn test_address_parse_named_instance() {
        let addr = ActorAddress::parse("actor:///services/llm/router@123").unwrap();
        match addr {
            ActorAddress::Named { path, instance } => {
                assert_eq!(path.as_str(), "services/llm/router");
                assert_eq!(instance.unwrap().0, 123);
            }
            _ => panic!("Expected Named address"),
        }
    }

    #[test]
    fn test_address_parse_global() {
        let addr = ActorAddress::parse("actor://123/456").unwrap();
        match addr {
            ActorAddress::Global { node_id, actor_id } => {
                assert_eq!(node_id.0, 123);
                assert_eq!(actor_id, 456);
            }
            _ => panic!("Expected Global address"),
        }
    }

    #[test]
    fn test_address_parse_local() {
        let addr = ActorAddress::parse("actor://0/456").unwrap();
        assert!(addr.is_local());

        match addr {
            ActorAddress::Global { node_id, actor_id } => {
                assert_eq!(node_id.0, 0);
                assert_eq!(actor_id, 456);
            }
            _ => panic!("Expected Global address"),
        }
    }

    #[test]
    fn test_address_resolve_local() {
        let addr = ActorAddress::parse("actor://0/456").unwrap();
        let current_node = NodeId::new(123);
        let resolved = addr.resolve_local(current_node);

        match resolved {
            ActorAddress::Global { node_id, actor_id } => {
                assert_eq!(node_id.0, 123);
                assert_eq!(actor_id, 456);
            }
            _ => panic!("Expected Global address"),
        }
    }

    #[test]
    fn test_address_to_uri() {
        // Named service
        let addr = ActorAddress::named(ActorPath::new("services/api").unwrap());
        assert_eq!(addr.to_uri(), "actor:///services/api");

        // Named instance
        let addr =
            ActorAddress::named_instance(ActorPath::new("services/api").unwrap(), NodeId::new(123));
        assert_eq!(addr.to_uri(), "actor:///services/api@123");

        // Global
        let addr = ActorAddress::global(NodeId::new(123), 456);
        assert_eq!(addr.to_uri(), "actor://123/456");

        // Local
        let addr = ActorAddress::local(456);
        assert_eq!(addr.to_uri(), "actor://0/456");
    }

    #[test]
    fn test_address_invalid_scheme() {
        assert!(ActorAddress::parse("http://localhost/actor").is_err());
        assert!(ActorAddress::parse("actors://localhost/actor").is_err());
    }

    #[test]
    fn test_address_roundtrip() {
        let cases = vec![
            "actor:///services/llm/router",
            "actor:///services/llm/router@123",
            "actor://123/456",
            "actor://0/789",
        ];

        for uri in cases {
            let addr = ActorAddress::parse(uri).unwrap();
            assert_eq!(addr.to_uri(), uri);
        }
    }
}
