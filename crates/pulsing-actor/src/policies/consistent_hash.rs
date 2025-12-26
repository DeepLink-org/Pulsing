//! Consistent hashing load balancing policy
//!
//! Routes requests to workers based on session ID or user ID using consistent hashing,
//! ensuring that requests from the same user/session consistently go to the same worker.
//! This is essential for scenarios requiring session affinity or cache locality.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders, Worker};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Number of virtual nodes per physical worker (for better load distribution)
const VIRTUAL_NODES_PER_WORKER: u32 = 160;

/// Consistent hashing policy
///
/// Routes requests based on session ID or user ID using consistent hashing,
/// ensuring that requests from the same user/session consistently go to the same worker.
/// Uses virtual nodes for better load distribution when workers are added/removed.
#[derive(Debug)]
pub struct ConsistentHashPolicy {
    /// Hash ring mapping hash values to worker URLs
    hash_ring: RwLock<BTreeMap<u64, String>>,
    /// Current set of workers (for detecting changes)
    current_workers: RwLock<Vec<String>>,
}

impl ConsistentHashPolicy {
    /// Create a new consistent hash policy
    pub fn new() -> Self {
        Self {
            hash_ring: RwLock::new(BTreeMap::new()),
            current_workers: RwLock::new(Vec::new()),
        }
    }

    /// MurmurHash64A implementation (from Facebook's mcrouter)
    fn murmur_hash_64a(key: &[u8], seed: u32) -> u64 {
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: i32 = 47;

        let mut h = (seed as u64) ^ ((key.len() as u64).wrapping_mul(M));

        // Process 8-byte chunks
        let chunks = key.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut k = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);

            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);

            h ^= k;
            h = h.wrapping_mul(M);
        }

        // Process remaining bytes
        match remainder.len() {
            7 => {
                h ^= (remainder[6] as u64) << 48;
                h ^= (remainder[5] as u64) << 40;
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            6 => {
                h ^= (remainder[5] as u64) << 40;
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            5 => {
                h ^= (remainder[4] as u64) << 32;
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            4 => {
                h ^= (remainder[3] as u64) << 24;
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            3 => {
                h ^= (remainder[2] as u64) << 16;
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            2 => {
                h ^= (remainder[1] as u64) << 8;
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            1 => {
                h ^= remainder[0] as u64;
                h = h.wrapping_mul(M);
            }
            _ => {}
        }

        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;

        h
    }

    /// Hash function for consistent hashing
    fn hash(key: &str) -> u64 {
        Self::murmur_hash_64a(key.as_bytes(), 4193360111)
    }

    /// Update the hash ring when workers change
    fn update_hash_ring(&self, workers: &[Arc<dyn Worker>]) {
        let worker_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        // Check if workers have changed
        {
            let current = self.current_workers.read().unwrap();
            if *current == worker_urls {
                return; // No change needed
            }
        }

        // Rebuild hash ring
        let mut new_ring = BTreeMap::new();

        for worker_url in &worker_urls {
            // Create virtual nodes for better distribution
            for i in 0..VIRTUAL_NODES_PER_WORKER {
                let virtual_key = format!("{}:{}", worker_url, i);
                let hash_value = Self::hash(&virtual_key);
                new_ring.insert(hash_value, worker_url.clone());
            }
        }

        // Update both the ring and current workers
        {
            let mut ring = self.hash_ring.write().unwrap();
            *ring = new_ring;
        }
        {
            let mut current = self.current_workers.write().unwrap();
            *current = worker_urls;
        }

        info!(
            "Updated consistent hash ring with {} workers and {} virtual nodes",
            workers.len(),
            workers.len() as u32 * VIRTUAL_NODES_PER_WORKER
        );
    }

    /// Find the worker for a given hash key
    fn find_worker_by_hash(&self, hash_key: &str) -> Option<String> {
        let hash_value = Self::hash(hash_key);

        let ring = self.hash_ring.read().unwrap();
        if ring.is_empty() {
            return None;
        }

        // Find the first worker with hash >= our hash value
        // If none found, wrap around to the first worker
        let selected_worker = ring
            .range(hash_value..)
            .next()
            .or_else(|| ring.iter().next())
            .map(|(_, worker_url)| worker_url.clone());

        if let Some(ref worker) = selected_worker {
            debug!(
                "Consistent hash: key='{}' hash={:016x} -> worker='{}'",
                hash_key, hash_value, worker
            );
        }

        selected_worker
    }

    /// HTTP header names to check for session ID (case-insensitive, checked in order)
    const SESSION_HEADER_NAMES: &'static [&'static str] = &[
        "x-session-id",
        "x-user-id",
        "x-tenant-id",
        "x-request-id",
        "x-correlation-id",
        "x-trace-id",
    ];

    /// Extract hash key from HTTP headers
    fn extract_hash_key_from_headers(&self, headers: &RequestHeaders) -> Option<String> {
        for header_name in Self::SESSION_HEADER_NAMES {
            if let Some(value) = headers.get(*header_name) {
                if !value.is_empty() {
                    debug!("Found session key in header '{}': {}", header_name, value);
                    return Some(format!("header:{}:{}", header_name, value));
                }
            }
        }
        None
    }

    /// Extract hash key from request body
    fn extract_hash_key_from_body(&self, request_text: Option<&str>) -> Option<String> {
        let text = request_text.unwrap_or("");
        if text.is_empty() {
            return None;
        }

        // Try to extract session_params.session_id
        if let Some(session_id) =
            self.extract_nested_field_value(text, "session_params", "session_id")
        {
            return Some(format!("session:{}", session_id));
        }

        // Try to extract user field (OpenAI format)
        if let Some(user) = self.extract_field_value(text, "user") {
            return Some(format!("user:{}", user));
        }

        // Fallback: try session_id field
        if let Some(session_id) = self.extract_field_value(text, "session_id") {
            return Some(format!("session:{}", session_id));
        }

        // Fallback: try user_id field
        if let Some(user_id) = self.extract_field_value(text, "user_id") {
            return Some(format!("user:{}", user_id));
        }

        None
    }

    /// Extract hash key with priority: headers > body > fallback
    fn extract_hash_key(
        &self,
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> String {
        // First priority: HTTP headers
        if let Some(hdrs) = headers {
            if let Some(key) = self.extract_hash_key_from_headers(hdrs) {
                return key;
            }
        }

        // Second priority: Body fields
        if let Some(key) = self.extract_hash_key_from_body(request_text) {
            return key;
        }

        // Final fallback: hash of request body
        let text = request_text.unwrap_or("");
        if text.len() > 100 {
            format!("request_hash:{:016x}", Self::hash(text))
        } else {
            format!("request:{}", text)
        }
    }

    /// Extract nested field value like session_params.session_id from JSON text
    fn extract_nested_field_value(
        &self,
        text: &str,
        parent_field: &str,
        child_field: &str,
    ) -> Option<String> {
        if let Some(parent_start) = self.find_field_start(text, parent_field) {
            if let Some(obj_start) = text[parent_start..].find('{') {
                let obj_start_pos = parent_start + obj_start;
                if let Some(obj_content) = self.extract_json_object(&text[obj_start_pos..]) {
                    return self.extract_field_value(&obj_content, child_field);
                }
            }
        }
        None
    }

    /// Find the start position of a field in JSON text
    fn find_field_start(&self, text: &str, field_name: &str) -> Option<usize> {
        let patterns = [format!("\"{}\"", field_name), format!("'{}'", field_name)];

        for pattern in &patterns {
            if let Some(field_pos) = text.find(pattern) {
                let after_field = &text[field_pos + pattern.len()..];
                for (i, ch) in after_field.char_indices() {
                    if ch == ':' {
                        return Some(field_pos + pattern.len() + i + 1);
                    } else if !ch.is_whitespace() {
                        break;
                    }
                }
            }
        }
        None
    }

    /// Extract JSON object content (simple brace matching)
    fn extract_json_object(&self, text: &str) -> Option<String> {
        if !text.starts_with('{') {
            return None;
        }

        let mut brace_count = 0;
        let mut end_pos = 0;

        for (i, ch) in text.char_indices() {
            match ch {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        end_pos = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if brace_count == 0 && end_pos > 0 {
            Some(text[0..end_pos].to_string())
        } else {
            None
        }
    }

    /// Extract field value from JSON-like text
    fn extract_field_value(&self, text: &str, field_name: &str) -> Option<String> {
        let patterns = [
            format!("\"{}\"", field_name),
            format!("'{}'", field_name),
            field_name.to_string(),
        ];

        for pattern in &patterns {
            if let Some(field_pos) = text.find(pattern) {
                let after_field = &text[field_pos + pattern.len()..];

                let mut colon_pos = None;
                for (i, ch) in after_field.char_indices() {
                    if ch == ':' {
                        colon_pos = Some(i);
                        break;
                    } else if !ch.is_whitespace() {
                        break;
                    }
                }

                if let Some(colon_idx) = colon_pos {
                    let after_colon = &after_field[colon_idx + 1..];
                    let trimmed = after_colon.trim_start();

                    // Extract quoted string
                    if trimmed.starts_with('"') {
                        if let Some(stripped) = trimmed.strip_prefix('"') {
                            if let Some(end_quote) = stripped.find('"') {
                                return Some(stripped[..end_quote].to_string());
                            }
                        }
                    } else if trimmed.starts_with('\'') {
                        if let Some(stripped) = trimmed.strip_prefix('\'') {
                            if let Some(end_quote) = stripped.find('\'') {
                                return Some(stripped[..end_quote].to_string());
                            }
                        }
                    } else {
                        // Unquoted value
                        let end_pos = trimmed
                            .find(&[',', ' ', '}', ']', '\n', '\r', '\t'][..])
                            .unwrap_or(trimmed.len());
                        if end_pos > 0 {
                            return Some(trimmed[..end_pos].to_string());
                        }
                    }
                }
            }
        }

        None
    }
}

impl LoadBalancingPolicy for ConsistentHashPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Update hash ring if needed
        self.update_hash_ring(workers);

        // Extract hash key
        let hash_key = self.extract_hash_key(request_text, headers);
        debug!("Consistent hash: extracted key '{}'", hash_key);

        // Find target worker using consistent hashing
        let target_worker_url = match self.find_worker_by_hash(&hash_key) {
            Some(url) => url,
            None => {
                // Fallback to first healthy worker
                return Some(healthy_indices[0]);
            }
        };

        // Find the worker index that matches our target
        let selected_idx = workers.iter().position(|w| w.url() == target_worker_url);

        match selected_idx {
            Some(idx) => {
                // Verify the worker is healthy
                if workers[idx].is_healthy() && workers[idx].circuit_breaker_allows() {
                    debug!(
                        "Consistent hash routing: key='{}' -> worker='{}' (index={})",
                        hash_key,
                        workers[idx].url(),
                        idx
                    );
                    Some(idx)
                } else {
                    // Target worker is unhealthy, fall back to first healthy worker
                    debug!(
                        "Target worker '{}' is unhealthy, falling back",
                        workers[idx].url()
                    );
                    Some(healthy_indices[0])
                }
            }
            None => {
                // Worker not found, fall back to first healthy worker
                debug!(
                    "Target worker '{}' not found, falling back",
                    target_worker_url
                );
                Some(healthy_indices[0])
            }
        }
    }

    fn name(&self) -> &'static str {
        "consistent_hash"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn reset(&self) {
        let mut ring = self.hash_ring.write().unwrap();
        ring.clear();
        let mut current = self.current_workers.write().unwrap();
        current.clear();
        info!("Consistent hash policy reset - hash ring cleared");
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // For PD mode, use consistent hashing for both prefill and decode
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;
        let decode_idx = self.select_worker_with_headers(decode_workers, request_text, headers)?;
        Some((prefill_idx, decode_idx))
    }
}

impl Default for ConsistentHashPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::BasicWorker;

    #[test]
    fn test_hash_consistency() {
        let key = "test_session_123";
        let hash1 = ConsistentHashPolicy::hash(key);
        let hash2 = ConsistentHashPolicy::hash(key);
        assert_eq!(hash1, hash2, "Hash function should be deterministic");
    }

    #[test]
    fn test_extract_session_id() {
        let policy = ConsistentHashPolicy::new();

        let request1 = r#"{"session_id": "abc123", "prompt": "hello"}"#;
        assert_eq!(
            policy.extract_field_value(request1, "session_id"),
            Some("abc123".to_string())
        );

        let request2 = r#"{'session_id': 'def456', 'prompt': 'world'}"#;
        assert_eq!(
            policy.extract_field_value(request2, "session_id"),
            Some("def456".to_string())
        );
    }

    #[test]
    fn test_consistent_hash_selection() {
        let policy = ConsistentHashPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://worker1:8000".to_string())),
            Arc::new(BasicWorker::new("http://worker2:8000".to_string())),
            Arc::new(BasicWorker::new("http://worker3:8000".to_string())),
        ];

        // Same session should always go to same worker
        let request = r#"{"session_id": "consistent_test"}"#;
        let idx1 = policy.select_worker(&workers, Some(request));
        let idx2 = policy.select_worker(&workers, Some(request));
        let idx3 = policy.select_worker(&workers, Some(request));

        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);
        assert!(idx1.is_some());
    }

    #[test]
    fn test_consistent_hash_with_headers() {
        let policy = ConsistentHashPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new("http://worker1:8000".to_string())),
            Arc::new(BasicWorker::new("http://worker2:8000".to_string())),
        ];

        let mut headers = RequestHeaders::new();
        headers.insert("x-session-id".to_string(), "test-session-123".to_string());

        let idx1 = policy.select_worker_with_headers(&workers, None, Some(&headers));
        let idx2 = policy.select_worker_with_headers(&workers, None, Some(&headers));

        assert_eq!(idx1, idx2);
        assert!(idx1.is_some());
    }
}
