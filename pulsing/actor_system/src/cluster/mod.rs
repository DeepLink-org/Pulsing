//! Cluster module - Gossip-based service discovery
//!
//! Implements a SWIM-like protocol for:
//! - Cluster membership management
//! - Actor location discovery
//! - Failure detection

mod gossip;
mod member;
pub mod swim;

pub use gossip::{GossipCluster, GossipConfig};
pub use member::{ActorLocation, MemberInfo, MemberStatus};
pub use swim::{SwimConfig, SwimDetector, SwimMessage};

