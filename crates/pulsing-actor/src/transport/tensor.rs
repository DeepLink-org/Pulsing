//! Tensor data-plane abstraction.
//!
//! Clear-text tensor traffic uses a pooled, long-lived raw TCP connection.
//! `write_vectored` writes the small header, opaque metadata, and the original
//! buffers without first packing them into a combined body; `read_exact` reads
//! each payload directly into its final owned receive allocation. TLS and the
//! explicit `PULSING_TENSOR_TRANSPORT=http2` mode retain a compatibility HTTP/2
//! backend, which packs the buffers and therefore has additional payload copies.
//! A future same-host shared-memory implementation can implement the same trait
//! without changing PulsingQueue.

use crate::actor::{
    max_tensor_buffers, max_tensor_metadata_bytes, max_tensor_wire_bytes, ActorId, TensorMessage,
};
use crate::error::{PulsingError, Result, RuntimeError};
use std::io::{ErrorKind, IoSlice};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub(crate) const RAW_TENSOR_MAGIC: &[u8; 4] = b"PTR1";
const RAW_FIXED_HEADER_LEN: usize = 4 + 1 + 3 + 4 + 4 + 8 + 4;
const MAX_RAW_PATH_BYTES: usize = 64 * 1024;
const MAX_VECTORED_SLICES: usize = 64;

static RAW_FRAMES_SENT: AtomicU64 = AtomicU64::new(0);
static RAW_FRAMES_RECEIVED: AtomicU64 = AtomicU64::new(0);
static RAW_BYTES_SENT: AtomicU64 = AtomicU64::new(0);
static RAW_BYTES_RECEIVED: AtomicU64 = AtomicU64::new(0);
static HTTP2_FALLBACK_FRAMES: AtomicU64 = AtomicU64::new(0);
static HTTP2_FALLBACK_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_COPY_MODEL: AtomicU64 = AtomicU64::new(0);
static RAW_CONNECTIONS_ACCEPTED: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawTensorTransportStats {
    pub frames_sent: u64,
    pub frames_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub http2_fallback_frames: u64,
    pub http2_fallback_bytes: u64,
    pub active_copy_model: &'static str,
    pub raw_connections_accepted: u64,
}

pub fn raw_tensor_transport_stats() -> RawTensorTransportStats {
    let active_copy_model = match LAST_COPY_MODEL.load(Ordering::Relaxed) {
        1 => "direct_tcp",
        2 => "packed_http2_compatibility",
        _ => "unused",
    };
    RawTensorTransportStats {
        frames_sent: RAW_FRAMES_SENT.load(Ordering::Relaxed),
        frames_received: RAW_FRAMES_RECEIVED.load(Ordering::Relaxed),
        bytes_sent: RAW_BYTES_SENT.load(Ordering::Relaxed),
        bytes_received: RAW_BYTES_RECEIVED.load(Ordering::Relaxed),
        http2_fallback_frames: HTTP2_FALLBACK_FRAMES.load(Ordering::Relaxed),
        http2_fallback_bytes: HTTP2_FALLBACK_BYTES.load(Ordering::Relaxed),
        active_copy_model,
        raw_connections_accepted: RAW_CONNECTIONS_ACCEPTED.load(Ordering::Relaxed),
    }
}

pub(crate) fn record_tensor_http2_fallback(bytes: usize) {
    HTTP2_FALLBACK_FRAMES.fetch_add(1, Ordering::Relaxed);
    HTTP2_FALLBACK_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    LAST_COPY_MODEL.store(2, Ordering::Relaxed);
}

pub(crate) fn record_raw_tensor_connection() {
    RAW_CONNECTIONS_ACCEPTED.fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum RawTensorKind {
    Ask = 1,
    Tell = 2,
    TensorResponse = 3,
    Ack = 4,
    Error = 5,
    SingleResponse = 6,
}

impl TryFrom<u8> for RawTensorKind {
    type Error = PulsingError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Ask),
            2 => Ok(Self::Tell),
            3 => Ok(Self::TensorResponse),
            4 => Ok(Self::Ack),
            5 => Ok(Self::Error),
            6 => Ok(Self::SingleResponse),
            _ => Err(protocol_error(format!(
                "unknown raw tensor frame kind {value}"
            ))),
        }
    }
}

#[derive(Debug)]
pub(crate) struct RawTensorFrame {
    pub kind: RawTensorKind,
    pub path: String,
    pub message: TensorMessage,
}

fn protocol_error(message: impl Into<String>) -> PulsingError {
    PulsingError::from(RuntimeError::Other(format!(
        "Raw tensor protocol error: {}",
        message.into()
    )))
}

/// Write header, metadata and each tensor buffer directly to the socket.
/// No combined payload allocation is constructed on this path.
pub(crate) async fn write_raw_tensor_frame<W>(
    writer: &mut W,
    kind: RawTensorKind,
    path: &str,
    message: &TensorMessage,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let path_bytes = path.as_bytes();
    if path_bytes.len() > MAX_RAW_PATH_BYTES {
        return Err(protocol_error("actor path is too large"));
    }
    if message.metadata.len() > max_tensor_metadata_bytes() {
        return Err(protocol_error("metadata exceeds configured maximum"));
    }
    if message.buffers.len() > max_tensor_buffers() {
        return Err(protocol_error("buffer count exceeds configured maximum"));
    }

    let lengths_bytes = message
        .buffers
        .len()
        .checked_mul(8)
        .ok_or_else(|| protocol_error("length table overflow"))?;
    let total = RAW_FIXED_HEADER_LEN
        .checked_add(lengths_bytes)
        .and_then(|size| size.checked_add(path_bytes.len()))
        .and_then(|size| size.checked_add(message.metadata.len()))
        .and_then(|size| {
            message
                .buffers
                .iter()
                .try_fold(size, |size, buffer| size.checked_add(buffer.len()))
        })
        .ok_or_else(|| protocol_error("frame size overflow"))?;
    if total > max_tensor_wire_bytes() {
        return Err(protocol_error("frame exceeds configured maximum"));
    }

    let mut header = Vec::with_capacity(RAW_FIXED_HEADER_LEN + lengths_bytes);
    header.extend_from_slice(RAW_TENSOR_MAGIC);
    header.push(kind as u8);
    header.extend_from_slice(&[0; 3]);
    header.extend_from_slice(&message.version.to_le_bytes());
    header.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
    header.extend_from_slice(&(message.metadata.len() as u64).to_le_bytes());
    header.extend_from_slice(&(message.buffers.len() as u32).to_le_bytes());
    for buffer in &message.buffers {
        header.extend_from_slice(&(buffer.len() as u64).to_le_bytes());
    }

    let mut slices = Vec::with_capacity(3 + message.buffers.len());
    slices.push(IoSlice::new(&header));
    if !path_bytes.is_empty() {
        slices.push(IoSlice::new(path_bytes));
    }
    if !message.metadata.is_empty() {
        slices.push(IoSlice::new(&message.metadata));
    }
    slices.extend(
        message
            .buffers
            .iter()
            .filter(|buffer| !buffer.is_empty())
            .map(|buffer| IoSlice::new(buffer)),
    );
    write_all_vectored(writer, &mut slices).await?;
    writer
        .flush()
        .await
        .map_err(|error| protocol_error(error.to_string()))?;
    RAW_FRAMES_SENT.fetch_add(1, Ordering::Relaxed);
    RAW_BYTES_SENT.fetch_add(total as u64, Ordering::Relaxed);
    LAST_COPY_MODEL.store(1, Ordering::Relaxed);
    Ok(())
}

async fn write_all_vectored<W>(writer: &mut W, slices: &mut [IoSlice<'_>]) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut remaining = slices;
    while !remaining.is_empty() {
        let batch_len = remaining.len().min(MAX_VECTORED_SLICES);
        let written = writer
            .write_vectored(&remaining[..batch_len])
            .await
            .map_err(|error| protocol_error(error.to_string()))?;
        if written == 0 {
            return Err(protocol_error(
                std::io::Error::new(ErrorKind::WriteZero, "failed to write tensor frame")
                    .to_string(),
            ));
        }
        IoSlice::advance_slices(&mut remaining, written);
    }
    Ok(())
}

/// Read directly into the final per-buffer allocations. The returned
/// TensorMessage moves those Vec allocations into Bytes without a payload copy.
pub(crate) async fn read_raw_tensor_frame<R>(reader: &mut R) -> Result<RawTensorFrame>
where
    R: AsyncRead + Unpin,
{
    let mut fixed = [0u8; RAW_FIXED_HEADER_LEN];
    reader
        .read_exact(&mut fixed)
        .await
        .map_err(|error| protocol_error(error.to_string()))?;
    if &fixed[..4] != RAW_TENSOR_MAGIC {
        return Err(protocol_error("invalid magic"));
    }
    let kind = RawTensorKind::try_from(fixed[4])?;
    let version = u32::from_le_bytes(fixed[8..12].try_into().expect("fixed-width slice"));
    let path_len =
        u32::from_le_bytes(fixed[12..16].try_into().expect("fixed-width slice")) as usize;
    let metadata_len_u64 = u64::from_le_bytes(fixed[16..24].try_into().expect("fixed-width slice"));
    let metadata_len = usize::try_from(metadata_len_u64)
        .map_err(|_| protocol_error("metadata length exceeds this platform"))?;
    let buffer_count =
        u32::from_le_bytes(fixed[24..28].try_into().expect("fixed-width slice")) as usize;

    if path_len > MAX_RAW_PATH_BYTES {
        return Err(protocol_error("actor path exceeds configured maximum"));
    }
    if metadata_len > max_tensor_metadata_bytes() {
        return Err(protocol_error("metadata exceeds configured maximum"));
    }
    if buffer_count > max_tensor_buffers() {
        return Err(protocol_error("buffer count exceeds configured maximum"));
    }

    let lengths_bytes = buffer_count
        .checked_mul(8)
        .ok_or_else(|| protocol_error("length table overflow"))?;
    let mut length_table = vec![0u8; lengths_bytes];
    reader
        .read_exact(&mut length_table)
        .await
        .map_err(|error| protocol_error(error.to_string()))?;
    let mut lengths = Vec::with_capacity(buffer_count);
    let mut total = RAW_FIXED_HEADER_LEN
        .checked_add(lengths_bytes)
        .and_then(|size| size.checked_add(path_len))
        .and_then(|size| size.checked_add(metadata_len))
        .ok_or_else(|| protocol_error("frame size overflow"))?;
    for index in 0..buffer_count {
        let offset = index * 8;
        let len_u64 = u64::from_le_bytes(
            length_table[offset..offset + 8]
                .try_into()
                .expect("fixed-width slice"),
        );
        let len = usize::try_from(len_u64)
            .map_err(|_| protocol_error("buffer length exceeds this platform"))?;
        total = total
            .checked_add(len)
            .ok_or_else(|| protocol_error("frame size overflow"))?;
        lengths.push(len);
    }
    if total > max_tensor_wire_bytes() {
        return Err(protocol_error("frame exceeds configured maximum"));
    }

    let mut path_bytes = vec![0u8; path_len];
    reader
        .read_exact(&mut path_bytes)
        .await
        .map_err(|error| protocol_error(error.to_string()))?;
    let path = String::from_utf8(path_bytes)
        .map_err(|error| protocol_error(format!("invalid actor path: {error}")))?;

    let mut metadata = vec![0u8; metadata_len];
    reader
        .read_exact(&mut metadata)
        .await
        .map_err(|error| protocol_error(error.to_string()))?;
    let mut buffers = Vec::with_capacity(buffer_count);
    for len in lengths {
        let mut buffer = vec![0u8; len];
        reader
            .read_exact(&mut buffer)
            .await
            .map_err(|error| protocol_error(error.to_string()))?;
        buffers.push(buffer);
    }

    RAW_FRAMES_RECEIVED.fetch_add(1, Ordering::Relaxed);
    RAW_BYTES_RECEIVED.fetch_add(total as u64, Ordering::Relaxed);
    Ok(RawTensorFrame {
        kind,
        path,
        message: TensorMessage::from_owned_receive(version, metadata, buffers)?,
    })
}

pub(crate) fn raw_tensor_transport_requested() -> bool {
    !matches!(
        std::env::var("PULSING_TENSOR_TRANSPORT")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "http2" | "legacy" | "off"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct PartialVectoredWriter {
        data: Vec<u8>,
        max_write: usize,
        vectored_calls: usize,
        max_slices_seen: usize,
    }

    impl PartialVectoredWriter {
        fn new(max_write: usize) -> Self {
            Self {
                data: Vec::new(),
                max_write,
                vectored_calls: 0,
                max_slices_seen: 0,
            }
        }
    }

    impl AsyncWrite for PartialVectoredWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            let count = buffer.len().min(self.max_write);
            self.data.extend_from_slice(&buffer[..count]);
            Poll::Ready(Ok(count))
        }

        fn poll_write_vectored(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buffers: &[IoSlice<'_>],
        ) -> Poll<std::io::Result<usize>> {
            self.vectored_calls += 1;
            self.max_slices_seen = self.max_slices_seen.max(buffers.len());
            let mut remaining = self.max_write;
            let mut written = 0;
            for buffer in buffers {
                if remaining == 0 {
                    break;
                }
                let count = buffer.len().min(remaining);
                self.data.extend_from_slice(&buffer[..count]);
                written += count;
                remaining -= count;
            }
            Poll::Ready(Ok(written))
        }

        fn is_write_vectored(&self) -> bool {
            true
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn vectored_writer_handles_partial_writes_and_many_buffers() {
        let buffers = (0..200)
            .map(|index| Bytes::from(vec![index as u8; 3]))
            .collect();
        let message = TensorMessage::new(9, Bytes::from_static(b"manifest"), buffers);
        let mut writer = PartialVectoredWriter::new(7);

        write_raw_tensor_frame(&mut writer, RawTensorKind::Ask, "/actors/test", &message)
            .await
            .unwrap();

        assert!(writer.vectored_calls > 1);
        assert_eq!(writer.max_slices_seen, MAX_VECTORED_SLICES);

        let mut wire = writer.data.as_slice();
        let decoded = read_raw_tensor_frame(&mut wire).await.unwrap();
        assert_eq!(decoded.kind, RawTensorKind::Ask);
        assert_eq!(decoded.path, "/actors/test");
        assert_eq!(decoded.message.version, 9);
        assert_eq!(&decoded.message.metadata[..], b"manifest");
        assert_eq!(decoded.message.buffers.len(), 200);
        for (index, buffer) in decoded.message.buffers.iter().enumerate() {
            assert_eq!(&buffer[..], &[index as u8; 3]);
        }
    }

    #[tokio::test]
    async fn reader_rejects_excessive_buffer_count_before_allocating_table() {
        let mut header = Vec::with_capacity(RAW_FIXED_HEADER_LEN);
        header.extend_from_slice(RAW_TENSOR_MAGIC);
        header.push(RawTensorKind::Ask as u8);
        header.extend_from_slice(&[0; 3]);
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        header.extend_from_slice(&((max_tensor_buffers() + 1) as u32).to_le_bytes());

        let mut input = header.as_slice();
        let error = read_raw_tensor_frame(&mut input).await.unwrap_err();
        assert!(error.to_string().contains("buffer count exceeds"));
    }

    #[tokio::test]
    async fn reader_rejects_excessive_metadata_before_allocating_payload() {
        let mut header = Vec::with_capacity(RAW_FIXED_HEADER_LEN);
        header.extend_from_slice(RAW_TENSOR_MAGIC);
        header.push(RawTensorKind::Ask as u8);
        header.extend_from_slice(&[0; 3]);
        header.extend_from_slice(&1u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&((max_tensor_metadata_bytes() + 1) as u64).to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());

        let mut input = header.as_slice();
        let error = read_raw_tensor_frame(&mut input).await.unwrap_err();
        assert!(error.to_string().contains("metadata exceeds"));
    }

    #[tokio::test]
    async fn reader_rejects_excessive_total_before_allocating_payload() {
        let mut input = Vec::with_capacity(RAW_FIXED_HEADER_LEN + 8);
        input.extend_from_slice(RAW_TENSOR_MAGIC);
        input.push(RawTensorKind::Ask as u8);
        input.extend_from_slice(&[0; 3]);
        input.extend_from_slice(&1u32.to_le_bytes());
        input.extend_from_slice(&0u32.to_le_bytes());
        input.extend_from_slice(&0u64.to_le_bytes());
        input.extend_from_slice(&1u32.to_le_bytes());
        // No payload follows. The advertised size must be rejected after the
        // tiny length table is read and before any payload Vec is allocated.
        input.extend_from_slice(&(max_tensor_wire_bytes() as u64).to_le_bytes());

        let error = read_raw_tensor_frame(&mut input.as_slice())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("frame exceeds"));
    }
}

/// Observable payload-copy contract of a tensor transport backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorCopyModel {
    /// Compatibility fallback: one sender-side wire packing copy, plus possible
    /// HTTP/2 receive-body coalescing before zero-copy Bytes slices are exposed.
    PackedHttp2Compatibility,
    /// Raw TCP data plane: application buffer to kernel and kernel to final
    /// owned receive buffer only.
    DirectTcp,
    /// Reserved same-host process transport.
    SharedMemory,
}

/// Backend boundary for opaque metadata plus contiguous tensor buffers.
#[async_trait::async_trait]
pub trait TensorTransport: Send + Sync {
    async fn request_tensor(
        &self,
        actor_id: &ActorId,
        message: TensorMessage,
    ) -> Result<TensorMessage>;

    async fn send_tensor(&self, actor_id: &ActorId, message: TensorMessage) -> Result<()>;

    fn copy_model(&self) -> TensorCopyModel;
}
