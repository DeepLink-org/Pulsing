//! Message wire constants and zerocopy header — shared by both binding paths.

use serde::{Deserialize, Serialize};

pub const SEALED_PY_MSG_TYPE: &str = "__sealed_py_message__";
pub const SEALED_ZEROCOPY_MSG_TYPE: &str = "__sealed_zerocopy_message__";
pub const ZC_DESCRIPTOR_MSG_TYPE: &str = "__zc_descriptor__";
pub const ZC_CHUNK_MSG_TYPE: &str = "__zc_chunk__";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyDescriptorHeader {
    pub version: u32,
    pub buffer_count: usize,
    pub buffer_lengths: Vec<usize>,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub strides: Option<Vec<isize>>,
    pub transport: Option<String>,
    pub checksum: Option<String>,
}

pub fn zerocopy_chunk_bytes() -> usize {
    const DEFAULT: usize = 1024 * 1024;
    const MIN: usize = 4 * 1024;
    std::env::var("PULSING_ZEROCOPY_CHUNK_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.max(MIN))
        .unwrap_or(DEFAULT)
}

pub fn zerocopy_stream_threshold() -> usize {
    const DEFAULT: usize = 64 * 1024;
    std::env::var("PULSING_ZEROCOPY_STREAM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT)
}
