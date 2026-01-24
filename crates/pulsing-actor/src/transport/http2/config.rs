//! HTTP/2 transport configuration.

use std::time::Duration;

#[cfg(feature = "tls")]
use super::tls::TlsConfig;

/// HTTP/2 transport configuration.
#[derive(Debug, Clone)]
pub struct Http2Config {
    pub max_concurrent_streams: u32,
    pub initial_window_size: u32,
    pub initial_connection_window_size: u32,
    pub max_frame_size: u32,
    pub max_header_list_size: u32,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub stream_timeout: Duration,
    pub max_connections_per_host: usize,
    pub keepalive_interval: Option<Duration>,
    pub keepalive_timeout: Duration,
    pub http2_prior_knowledge: bool,
    pub max_retries: u32,
    pub retry_initial_delay: Duration,
    pub retry_max_delay: Duration,
    pub retry_use_jitter: bool,
    #[cfg(feature = "tls")]
    pub tls: Option<TlsConfig>,
}

impl Default for Http2Config {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 100,
            initial_window_size: 65535,
            initial_connection_window_size: 1024 * 1024,
            max_frame_size: 16 * 1024,
            max_header_list_size: 16 * 1024,

            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            stream_timeout: Duration::from_secs(300),
            max_connections_per_host: 10,
            keepalive_interval: Some(Duration::from_secs(30)),
            keepalive_timeout: Duration::from_secs(10),
            http2_prior_knowledge: true,
            max_retries: 3,
            retry_initial_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(10),
            retry_use_jitter: true,
            #[cfg(feature = "tls")]
            tls: None,
        }
    }
}

impl Http2Config {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn high_throughput() -> Self {
        Self {
            max_concurrent_streams: 200,
            initial_window_size: 256 * 1024,
            initial_connection_window_size: 4 * 1024 * 1024,
            max_frame_size: 64 * 1024,
            max_connections_per_host: 20,
            ..Default::default()
        }
    }

    pub fn streaming() -> Self {
        Self {
            stream_timeout: Duration::from_secs(600),
            max_concurrent_streams: 50,
            initial_window_size: 128 * 1024,
            keepalive_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        }
    }

    pub fn max_concurrent_streams(mut self, n: u32) -> Self {
        self.max_concurrent_streams = n;
        self
    }

    pub fn initial_window_size(mut self, size: u32) -> Self {
        self.initial_window_size = size;
        self
    }

    pub fn initial_connection_window_size(mut self, size: u32) -> Self {
        self.initial_connection_window_size = size;
        self
    }

    pub fn max_frame_size(mut self, size: u32) -> Self {
        self.max_frame_size = size;
        self
    }

    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn stream_timeout(mut self, timeout: Duration) -> Self {
        self.stream_timeout = timeout;
        self
    }

    pub fn max_connections_per_host(mut self, n: usize) -> Self {
        self.max_connections_per_host = n;
        self
    }

    /// Set keep-alive interval
    pub fn keepalive_interval(mut self, interval: Option<Duration>) -> Self {
        self.keepalive_interval = interval;
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

    /// Enable TLS with passphrase-derived certificates
    ///
    /// All nodes using the same passphrase will be able to communicate securely.
    /// The passphrase is used to derive a shared CA certificate, enabling
    /// automatic mutual TLS authentication.
    #[cfg(feature = "tls")]
    pub fn with_tls(mut self, passphrase: &str) -> anyhow::Result<Self> {
        self.tls = Some(TlsConfig::from_passphrase(passphrase)?);
        Ok(self)
    }

    /// Set TLS configuration directly
    #[cfg(feature = "tls")]
    pub fn tls_config(mut self, tls: TlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Check if TLS is enabled
    #[cfg(feature = "tls")]
    pub fn is_tls_enabled(&self) -> bool {
        self.tls.is_some()
    }

    /// Check if TLS is enabled (always false when `tls` feature is not enabled)
    #[cfg(not(feature = "tls"))]
    pub fn is_tls_enabled(&self) -> bool {
        false
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
        assert!(config.http2_prior_knowledge);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_builder_pattern() {
        let config = Http2Config::new()
            .max_concurrent_streams(200)
            .connect_timeout(Duration::from_secs(10))
            .max_retries(5);

        assert_eq!(config.max_concurrent_streams, 200);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
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
