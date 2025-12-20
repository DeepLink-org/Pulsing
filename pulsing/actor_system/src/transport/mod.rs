//! Transport module - network communication layer
//!
//! Provides reliable message transport between nodes.
//!
//! The primary transport is HTTP-based, providing a unified interface
//! for both actor messages and cluster gossip protocol.

pub mod codec;
pub mod http;
pub mod tcp;

pub use codec::{MessageCodec, TransportMessage};
pub use http::{
    ActorRequest, ActorResponse, GossipRequest, GossipResponse, HttpMessageHandler,
    HttpRemoteTransport, HttpTransport, HttpTransportConfig,
};
pub use tcp::{TcpRemoteTransport, TcpTransport, TcpTransportConfig};
