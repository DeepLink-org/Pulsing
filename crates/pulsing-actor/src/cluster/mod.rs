//! Cluster module - Gossip-based service discovery
//!
//! Implements a SWIM-like protocol for:
//! - Cluster membership management
//! - Actor location discovery (named actors with multi-instance support)
//! - Failure detection

mod gossip;
mod member;
pub mod swim;

pub use gossip::{GossipCluster, GossipConfig, GossipMessage};
pub use member::{
    ActorLocation, ClusterNode, FailureInfo, MemberInfo, MemberStatus, NamedActorInfo,
    NamedActorInstance, NodeStatus,
};
pub use swim::{SwimConfig, SwimDetector, SwimMessage};
