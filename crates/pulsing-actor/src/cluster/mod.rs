//! Cluster module.

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

pub use backends::GossipBackend;
pub use backends::{HeadNodeBackend, HeadNodeConfig};
