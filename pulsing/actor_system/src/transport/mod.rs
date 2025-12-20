//! Transport module - network communication layer
//!
//! Provides reliable message transport between nodes.

pub mod codec;
pub mod tcp;

pub use codec::{MessageCodec, TransportMessage};
pub use tcp::{TcpRemoteTransport, TcpTransport, TcpTransportConfig};

