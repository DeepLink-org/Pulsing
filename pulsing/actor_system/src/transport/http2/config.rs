//! HTTP/2 Transport Configuration
//!
//! Provides comprehensive configuration options for HTTP/2 transport including:
//! - Server settings (concurrent streams, window sizes, etc.)
//! - Client settings (timeouts, connection pooling)
//! - Retry policies

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

    // ========== Retry Configuration ==========
    /// Maximum number of retry attempts (default: 3)
    pub max_retries: u32,

    /// Initial retry delay (default: 100ms)
    pub retry_initial_delay: Duration,

    /// Maximum retry delay (default: 10s)
    pub retry_max_delay: Duration,

    /// Whether to use jitter in retry delays (default: true)
    pub retry_use_jitter: bool,
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

            // Retry defaults
            max_retries: 3,
            retry_initial_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(10),
            retry_use_jitter: true,
        }
    }
}

impl Http2Config {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for low-latency workloads
    pub fn low_latency() -> Self {
        Self {
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(10),
            max_retries: 2,
            retry_initial_delay: Duration::from_millis(50),
            retry_max_delay: Duration::from_secs(1),
            keepalive_interval: Some(Duration::from_secs(15)),
            ..Default::default()
        }
    }

    /// Create a configuration optimized for high-throughput workloads
    pub fn high_throughput() -> Self {
        Self {
            max_concurrent_streams: 200,
            initial_window_size: 256 * 1024,                 // 256KB
            initial_connection_window_size: 4 * 1024 * 1024, // 4MB
            max_frame_size: 64 * 1024,                       // 64KB
            max_connections_per_host: 20,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for streaming workloads (e.g., LLM inference)
    pub fn streaming() -> Self {
        Self {
            stream_timeout: Duration::from_secs(600), // 10 minutes
            max_concurrent_streams: 50,
            initial_window_size: 128 * 1024, // 128KB
            keepalive_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        }
    }

    // ========== Builder Methods ==========

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

    /// Set connection-level window size
    pub fn initial_connection_window_size(mut self, size: u32) -> Self {
        self.initial_connection_window_size = size;
        self
    }

    /// Set maximum frame size
    pub fn max_frame_size(mut self, size: u32) -> Self {
        self.max_frame_size = size;
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

    /// Set maximum connections per host
    pub fn max_connections_per_host(mut self, n: usize) -> Self {
        self.max_connections_per_host = n;
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

    /// Set maximum retry attempts
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Disable retries
    pub fn no_retries(mut self) -> Self {
        self.max_retries = 0;
        self
    }

    /// Set retry initial delay
    pub fn retry_initial_delay(mut self, delay: Duration) -> Self {
        self.retry_initial_delay = delay;
        self
    }

    /// Set retry max delay
    pub fn retry_max_delay(mut self, delay: Duration) -> Self {
        self.retry_max_delay = delay;
        self
    }

    /// Disable retry jitter
    pub fn disable_retry_jitter(mut self) -> Self {
        self.retry_use_jitter = false;
        self
    }

    /// Convert to retry config
    pub fn to_retry_config(&self) -> super::retry::RetryConfig {
        super::retry::RetryConfig {
            max_retries: self.max_retries,
            initial_delay: self.retry_initial_delay,
            max_delay: self.retry_max_delay,
            backoff_multiplier: 2.0,
            use_jitter: self.retry_use_jitter,
            idempotent_only: true,
        }
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
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_builder_pattern() {
        let config = Http2Config::new()
            .max_concurrent_streams(200)
            .connect_timeout(Duration::from_secs(10))
            .disable_http1_fallback()
            .max_retries(5);

        assert_eq!(config.max_concurrent_streams, 200);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert!(!config.enable_http1_fallback);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_low_latency_preset() {
        let config = Http2Config::low_latency();
        assert_eq!(config.connect_timeout, Duration::from_secs(2));
        assert_eq!(config.request_timeout, Duration::from_secs(10));
        assert_eq!(config.max_retries, 2);
    }

    #[test]
    fn test_high_throughput_preset() {
        let config = Http2Config::high_throughput();
        assert_eq!(config.max_concurrent_streams, 200);
        assert_eq!(config.max_connections_per_host, 20);
    }

    #[test]
    fn test_streaming_preset() {
        let config = Http2Config::streaming();
        assert_eq!(config.stream_timeout, Duration::from_secs(600));
        assert_eq!(config.max_concurrent_streams, 50);
    }

    #[test]
    fn test_to_retry_config() {
        let http2_config = Http2Config::default()
            .max_retries(5)
            .retry_initial_delay(Duration::from_millis(200));

        let retry_config = http2_config.to_retry_config();
        assert_eq!(retry_config.max_retries, 5);
        assert_eq!(retry_config.initial_delay, Duration::from_millis(200));
    }
}
