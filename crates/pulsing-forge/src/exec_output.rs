//! Codex-compatible exec output shapes and output buffering.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const DEFAULT_SHELL_TIMEOUT_MS: u64 = 10_000;
pub const DEFAULT_YIELD_TIME_MS: u64 = 250;
pub const MIN_YIELD_TIME_MS: u64 = 250;
pub const MAX_YIELD_TIME_MS: u64 = 30_000;
pub const DEFAULT_MAX_OUTPUT_TOKENS: usize = 10_000;
pub const SHELL_MAX_BYTES: usize = 256 * 1024;
/// Max bytes accepted per `write_stdin` call — guards against unbounded input growth.
pub const MAX_STDIN_BYTES: usize = 1024 * 1024;
/// Atomic sentinel while PTY / pipe session is still running.
pub const RUNNING_EXIT_SENTINEL: i32 = i32::MIN;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecStream {
    Stdout,
    Stderr,
    Pty,
}

/// Streaming chunk emitted while a unified exec session is running.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecOutputDelta {
    pub session_id: i32,
    pub stream: ExecStream,
    pub chunk: String,
}

/// Structured payload returned by `exec_command` / `write_stdin`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecCommandOutput {
    pub chunk_id: String,
    pub wall_time_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_token_count: Option<usize>,
    pub output: String,
}

impl ExecCommandOutput {
    pub fn new(
        output: String,
        wall_time_seconds: f64,
        exit_code: Option<i32>,
        session_id: Option<i32>,
    ) -> Self {
        let original_token_count = estimate_tokens(&output);
        Self {
            chunk_id: Uuid::new_v4().to_string(),
            wall_time_seconds,
            exit_code,
            session_id,
            original_token_count: Some(original_token_count),
            output,
        }
    }
}

/// Rolling buffer for unified exec sessions (head+tail truncation).
#[derive(Clone, Debug, Default)]
pub struct OutputBuffer {
    max_bytes: usize,
    data: String,
}

impl OutputBuffer {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            data: String::new(),
        }
    }

    pub fn push(&mut self, chunk: &str) {
        if chunk.is_empty() {
            return;
        }
        self.data.push_str(chunk);
        if self.data.len() > self.max_bytes {
            let keep = self.max_bytes / 2;
            let tail = tail_at(&self.data, keep);
            self.data = format!("...[output truncated]...\n{tail}");
        }
    }

    pub fn snapshot(&self) -> String {
        self.data.clone()
    }

    pub fn truncate_to_tokens(&mut self, max_tokens: usize) {
        let est = estimate_tokens(&self.data);
        if est <= max_tokens {
            return;
        }
        let ratio = max_tokens as f64 / est as f64;
        let keep = ((self.data.len() as f64) * ratio) as usize;
        let tail = tail_at(&self.data, keep);
        self.data = format!("...[token limit]...\n{tail}");
    }
}

/// Returns the trailing slice of `s` that keeps at least `keep` bytes,
/// snapped outward to the nearest UTF-8 char boundary. Naive byte-offset
/// slicing (`s[len - keep..]`) panics whenever the cut point lands inside a
/// multi-byte character (e.g. CJK output), so callers must go through this.
fn tail_at(s: &str, keep: usize) -> &str {
    let mut start = s.len().saturating_sub(keep);
    while start > 0 && !s.is_char_boundary(start) {
        start -= 1;
    }
    &s[start..]
}

/// Incrementally decodes a raw byte stream (PTY/pipe reads) into UTF-8,
/// holding back a possibly-incomplete trailing multi-byte sequence instead of
/// rendering it as `U+FFFD` when a codepoint is split across two reads.
#[derive(Debug, Default)]
pub struct Utf8ChunkDecoder {
    pending: Vec<u8>,
}

impl Utf8ChunkDecoder {
    /// Decodes `bytes` combined with any carry-over from the previous call.
    pub fn decode(&mut self, bytes: &[u8]) -> String {
        if !bytes.is_empty() {
            self.pending.extend_from_slice(bytes);
        }
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid_len = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid_len]).into_owned();
                let leftover = self.pending.len() - valid_len;
                // A genuine UTF-8 codepoint is at most 4 bytes; a longer
                // leftover means the bytes are simply invalid rather than a
                // split codepoint, so flush them lossily instead of
                // buffering forever.
                if leftover > 4 {
                    let rest = String::from_utf8_lossy(&self.pending[valid_len..]).into_owned();
                    self.pending.clear();
                    return out + &rest;
                }
                self.pending.drain(..valid_len);
                out
            }
        }
    }

    /// Flushes any buffered bytes at end-of-stream (best-effort lossy decode).
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

pub fn estimate_tokens(text: &str) -> usize {
    // Cheap proxy: ~4 chars per token for mixed shell output.
    (text.len() / 4).max(1)
}

pub fn clamp_yield_ms(raw: Option<u64>) -> u64 {
    raw.unwrap_or(DEFAULT_YIELD_TIME_MS)
        .clamp(MIN_YIELD_TIME_MS, MAX_YIELD_TIME_MS)
}

pub fn shell_timeout_ms(args: &serde_json::Value) -> u64 {
    if let Some(ms) = args.get("timeout_ms").and_then(|v| v.as_u64()) {
        return ms.max(1);
    }
    if let Some(sec) = args.get("timeout_sec").and_then(|v| v.as_u64()) {
        return (sec * 1000).max(1);
    }
    DEFAULT_SHELL_TIMEOUT_MS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_truncation_does_not_split_multibyte_chars() {
        // Regression test: naive `s[len - keep..]` byte slicing panics (or
        // corrupts output) when the cut point lands mid-codepoint. Every "中"
        // is 3 bytes, so a `max_bytes` that isn't a multiple of 3 forces the
        // cut through the middle of a character.
        let mut buf = OutputBuffer::new(10);
        buf.push(&"中".repeat(20));
        let snapshot = buf.snapshot();
        assert!(snapshot.contains("...[output truncated]..."));
        // If this didn't panic, the buffer is guaranteed valid UTF-8 already
        // (Rust `String` invariant) — assert content is sane too.
        assert!(snapshot.ends_with('中'));
    }

    #[test]
    fn truncate_to_tokens_does_not_split_multibyte_chars() {
        let mut buf = OutputBuffer::new(1 << 20);
        buf.push(&"中".repeat(50));
        buf.truncate_to_tokens(1);
        let snapshot = buf.snapshot();
        assert!(snapshot.contains("...[token limit]..."));
        assert!(snapshot.ends_with('中'));
    }

    #[test]
    fn utf8_chunk_decoder_reassembles_split_codepoint() {
        let bytes = "héllo 中文".as_bytes();
        for split in 1..bytes.len() {
            let mut decoder = Utf8ChunkDecoder::default();
            let mut out = decoder.decode(&bytes[..split]);
            out.push_str(&decoder.decode(&bytes[split..]));
            out.push_str(&decoder.finish());
            assert_eq!(out, "héllo 中文", "split at byte {split} corrupted output");
        }
    }

    #[test]
    fn utf8_chunk_decoder_flushes_invalid_bytes_instead_of_growing_forever() {
        let mut decoder = Utf8ChunkDecoder::default();
        let out = decoder.decode(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, b'x']);
        assert!(out.contains('x'));
        assert!(decoder.finish().is_empty());
    }
}
