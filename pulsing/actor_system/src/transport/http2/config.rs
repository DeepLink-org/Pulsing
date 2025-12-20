//! HTTP/2 Transport Configuration

use std::time::Duration;

/// HTTP/2 transport configuration
#[derive(Debug, Clone)]
pub struct Http2Config {
    // ========== Server Configuration ==========
    /// Maximum number of concurrent streams per connection (default: 100)
    pub max_concurrent_streams: u32,

    /// Initial window size for flow control (default: 65535 bytes)
    pub initial_window_size: u32,

    /// Connection-level window size (default: 1MB)
    pub initial_connection_window_size: u32,

    /// Maximum frame size (default: 16KB)
    pub max_frame_size: u32,

    /// Maximum header list size (default: 16KB)
    pub max_header_list_size: u32,

    // ========== Client Configuration ==========
    /// Connection timeout (default: 5s)
    pub connect_timeout: Duration,

    /// Request timeout for non-streaming requests (default: 30s)
    pub request_timeout: Duration,

    /// Timeout for streaming requests (default: 5min)
    pub stream_timeout: Duration,

    /// Maximum connections per host (default: 10)
    pub max_connections_per_host: usize,

    // ========== Common Configuration ==========
    /// Keep-alive ping interval (default: 30s, None to disable)
    pub keepalive_interval: Option<Duration>,

    /// Keep-alive timeout (default: 10s)
    pub keepalive_timeout: Duration,

    /// Enable HTTP/1.1 fallback for compatibility (default: true)
    pub enable_http1_fallback: bool,

    /// Enable HTTP/2 prior knowledge mode (default: true)
    /// When true, client sends HTTP/2 preface directly without upgrade
    pub http2_prior_knowledge: bool,
}

impl Default for Http2Config {
    fn default() -> Self {
        Self {
            // Server defaults
            max_concurrent_streams: 100,
            initial_window_size: 65535,
            initial_connection_window_size: 1024 * 1024, // 1MB
            max_frame_size: 16 * 1024,                   // 16KB
            max_header_list_size: 16 * 1024,             // 16KB

            // Client defaults
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            stream_timeout: Duration::from_secs(300), // 5 minutes
            max_connections_per_host: 10,

            // Common defaults
            keepalive_interval: Some(Duration::from_secs(30)),
            keepalive_timeout: Duration::from_secs(10),
            enable_http1_fallback: true,
            http2_prior_knowledge: true,
        }
    }
}

impl Http2Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum concurrent streams
    pub fn max_concurrent_streams(mut self, n: u32) -> Self {
        self.max_concurrent_streams = n;
        self
    }

    /// Set initial window size
    pub fn initial_window_size(mut self, size: u32) -> Self {
        self.initial_window_size = size;
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set request timeout
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set stream timeout
    pub fn stream_timeout(mut self, timeout: Duration) -> Self {
        self.stream_timeout = timeout;
        self
    }

    /// Set keep-alive interval
    pub fn keepalive_interval(mut self, interval: Option<Duration>) -> Self {
        self.keepalive_interval = interval;
        self
    }

    /// Disable HTTP/1.1 fallback
    pub fn disable_http1_fallback(mut self) -> Self {
        self.enable_http1_fallback = false;
        self
    }

    /// Disable HTTP/2 prior knowledge (use upgrade instead)
    pub fn disable_prior_knowledge(mut self) -> Self {
        self.http2_prior_knowledge = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Http2Config::default();
        assert_eq!(config.max_concurrent_streams, 100);
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert!(config.enable_http1_fallback);
        assert!(config.http2_prior_knowledge);
    }

    #[test]
    fn test_builder_pattern() {
        let config = Http2Config::new()
            .max_concurrent_streams(200)
            .connect_timeout(Duration::from_secs(10))
            .disable_http1_fallback();

        assert_eq!(config.max_concurrent_streams, 200);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert!(!config.enable_http1_fallback);
    }
}

