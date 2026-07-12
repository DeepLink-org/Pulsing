//! Zerocopy wire helpers (Python-independent).

use std::cmp::min;

use pulsing_actor::prelude::Message;

use crate::message::{
    zerocopy_chunk_bytes, ZeroCopyDescriptorHeader, ZC_CHUNK_MSG_TYPE, ZC_DESCRIPTOR_MSG_TYPE,
};

pub async fn reassemble_zerocopy_stream(
    header: ZeroCopyDescriptorHeader,
    stream: &mut std::pin::Pin<
        Box<dyn futures::Stream<Item = pulsing_actor::error::Result<Message>> + Send>,
    >,
) -> pulsing_actor::error::Result<(ZeroCopyDescriptorHeader, Vec<Vec<u8>>)> {
    use futures::StreamExt;

    let mut raw_buffers: Vec<Vec<u8>> = header
        .buffer_lengths
        .iter()
        .map(|&len| Vec::with_capacity(len))
        .collect();
    let total_expected: usize = header.buffer_lengths.iter().sum();

    let mut buf_idx = 0;
    let mut received = 0usize;

    while received < total_expected {
        let frame = stream.next().await.ok_or_else(|| {
            pulsing_actor::error::PulsingError::from(pulsing_actor::error::RuntimeError::Other(
                "Zerocopy stream ended before all data received".into(),
            ))
        })??;

        match frame {
            Message::Single {
                ref msg_type,
                ref data,
            } if msg_type == ZC_CHUNK_MSG_TYPE => {
                let remaining_in_buf = header.buffer_lengths[buf_idx] - raw_buffers[buf_idx].len();
                if data.len() <= remaining_in_buf {
                    raw_buffers[buf_idx].extend_from_slice(data);
                } else {
                    let first_part = &data[..remaining_in_buf];
                    raw_buffers[buf_idx].extend_from_slice(first_part);
                    let mut rest = &data[remaining_in_buf..];
                    buf_idx += 1;
                    while !rest.is_empty() && buf_idx < raw_buffers.len() {
                        let can_take = min(
                            rest.len(),
                            header.buffer_lengths[buf_idx] - raw_buffers[buf_idx].len(),
                        );
                        raw_buffers[buf_idx].extend_from_slice(&rest[..can_take]);
                        rest = &rest[can_take..];
                        if raw_buffers[buf_idx].len() == header.buffer_lengths[buf_idx] {
                            buf_idx += 1;
                        }
                    }
                }
                received += data.len();
                if buf_idx < raw_buffers.len()
                    && raw_buffers[buf_idx].len() == header.buffer_lengths[buf_idx]
                {
                    buf_idx += 1;
                }
            }
            _ => {
                return Err(pulsing_actor::error::PulsingError::from(
                    pulsing_actor::error::RuntimeError::Other(format!(
                        "Unexpected frame in zerocopy stream: {:?}",
                        frame.msg_type()
                    )),
                ));
            }
        }
    }

    Ok((header, raw_buffers))
}

pub fn zerocopy_mode() -> String {
    std::env::var("PULSING_ZEROCOPY")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
}

pub fn chunk_len() -> usize {
    zerocopy_chunk_bytes()
}
