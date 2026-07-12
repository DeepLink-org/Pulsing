//! Shared binding logic for Path A (PyO3) and Path B (RustPython).
//!
//! Keep Python-facing behavior identical by centralizing wire formats, IDs, and
//! config helpers here; each binding crate only adapts to its VM.

pub mod ids;
pub mod message;
pub mod zerocopy;

pub use ids::{parse_actor_id, parse_node_id, PyActorIdView, PyNodeIdView};
pub use message::{
    ZeroCopyDescriptorHeader, SEALED_PY_MSG_TYPE, SEALED_ZEROCOPY_MSG_TYPE, ZC_CHUNK_MSG_TYPE,
    ZC_DESCRIPTOR_MSG_TYPE,
};
pub use zerocopy::{chunk_len, reassemble_zerocopy_stream, zerocopy_mode};
