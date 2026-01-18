//! Cluster module - Gossip-based service discovery
//!
//! Implements a SWIM-like protocol for:
//! - Cluster membership management
//! - Actor location discovery (named actors with multi-instance support)
//! - Failure detection

mod gossip;
mod member;
mod naming;
pub mod swim;

pub mod backends;

pub use gossip::{GossipCluster, GossipConfig, GossipMessage};
pub use member::{
    ActorLocation, ClusterNode, FailureInfo, MemberInfo, MemberStatus, NamedActorInfo,
    NamedActorInstance, NodeStatus,
};
pub use naming::NamingBackend;
pub use swim::{SwimConfig, SwimDetector, SwimMessage};

// Re-export backends for convenience
pub use backends::GossipBackend;
// Re-export head node backend types (via backends module's re-exports)
pub use backends::{HeadNodeBackend, HeadNodeConfig};
