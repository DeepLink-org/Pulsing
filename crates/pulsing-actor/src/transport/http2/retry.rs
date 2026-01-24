//! Retry and timeout strategies for HTTP/2 transport.

use std::time::Duration;

/// Retry configuration.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub use_jitter: bool,
    pub idempotent_only: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            use_jitter: true,
            idempotent_only: true,
        }
    }
}

impl RetryConfig {
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    pub fn with_max_retries(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    pub fn backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    pub fn use_jitter(mut self, enable: bool) -> Self {
        self.use_jitter = enable;
        self
    }

    pub fn allow_non_idempotent(mut self) -> Self {
        self.idempotent_only = false;
        self
    }

    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let base_delay = self.initial_delay.as_millis() as f64
            * self
                .backoff_multiplier
                .powi(attempt.saturating_sub(1) as i32);

        let capped_delay = base_delay.min(self.max_delay.as_millis() as f64);

        let final_delay = if self.use_jitter {
            let jitter = rand_jitter();
            capped_delay * (0.5 + jitter)
        } else {
            capped_delay
        };

        Duration::from_millis(final_delay as u64)
    }
}

fn rand_jitter() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

/// Error classification for retry decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryableError {
    Connection,
    Timeout,
    ServerOverloaded,
    ServerError,
    ClientError,
    Unknown,
}

impl RetryableError {
    pub fn classify(error: &anyhow::Error) -> Self {
        let msg = error.to_string().to_lowercase();

        if msg.contains("backing off") {
            return Self::Unknown;
        }

        if msg.contains("connection")
            || msg.contains("connect")
            || msg.contains("refused")
            || msg.contains("reset")
        {
            return Self::Connection;
        }

        if msg.contains("timeout") || msg.contains("timed out") {
            return Self::Timeout;
        }

        if msg.contains("503") || msg.contains("service unavailable") {
            return Self::ServerOverloaded;
        }

        if msg.contains("500")
            || msg.contains("502")
            || msg.contains("504")
            || msg.contains("internal server error")
            || msg.contains("bad gateway")
            || msg.contains("gateway timeout")
        {
            return Self::ServerError;
        }

        if msg.contains("400")
            || msg.contains("401")
            || msg.contains("403")
            || msg.contains("404")
            || msg.contains("bad request")
            || msg.contains("unauthorized")
            || msg.contains("forbidden")
            || msg.contains("not found")
        {
            return Self::ClientError;
        }

        Self::Unknown
    }

    pub fn is_retryable(&self, idempotent_only: bool, is_idempotent: bool) -> bool {
        match self {
            // Connection errors are always retryable
            Self::Connection => true,
            // Timeout is retryable for idempotent ops
            Self::Timeout => !idempotent_only || is_idempotent,
            // Server overloaded is retryable
            Self::ServerOverloaded => true,
            // Server errors might be retryable for idempotent ops
            Self::ServerError => !idempotent_only || is_idempotent,
            // Client errors are never retryable
            Self::ClientError => false,
            // Unknown errors are not retryable
            Self::Unknown => false,
        }
    }
}

/// Retry executor
pub struct RetryExecutor {
    config: RetryConfig,
}

impl RetryExecutor {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Execute a function with retry logic
    pub async fn execute<F, Fut, T>(&self, is_idempotent: bool, mut f: F) -> anyhow::Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<T>>,
    {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            // Wait before retry (except for first attempt)
            if attempt > 0 {
                let delay = self.config.delay_for_attempt(attempt);
                tracing::debug!(
                    attempt = attempt,
                    delay_ms = delay.as_millis(),
                    "Retrying after delay"
                );
                tokio::time::sleep(delay).await;
            }

            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    let error_type = RetryableError::classify(&e);
                    let should_retry = attempt < self.config.max_retries
                        && error_type.is_retryable(self.config.idempotent_only, is_idempotent);

                    if should_retry {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max = self.config.max_retries,
                            error = %e,
                            error_type = ?error_type,
                            "Request failed, will retry"
                        );
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Max retries exceeded")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert!(config.use_jitter);
        assert!(config.idempotent_only);
    }

    #[test]
    fn test_delay_calculation() {
        let config = RetryConfig::default().use_jitter(false);

        // First attempt has no delay
        assert_eq!(config.delay_for_attempt(0), Duration::ZERO);

        // Second attempt uses initial delay
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(100));

        // Third attempt doubles (2.0 multiplier)
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(200));

        // Fourth attempt doubles again
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(400));
    }

    #[test]
    fn test_max_delay_cap() {
        let config = RetryConfig::default()
            .use_jitter(false)
            .initial_delay(Duration::from_secs(1))
            .max_delay(Duration::from_secs(5))
            .backoff_multiplier(10.0);

        // First retry: 1s
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(1));

        // Second retry: 10s, but capped at 5s
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(5));
    }

    #[test]
    fn test_error_classification() {
        let conn_err = anyhow::anyhow!("Connection refused");
        assert_eq!(
            RetryableError::classify(&conn_err),
            RetryableError::Connection
        );

        let timeout_err = anyhow::anyhow!("Request timeout");
        assert_eq!(
            RetryableError::classify(&timeout_err),
            RetryableError::Timeout
        );

        let server_err = anyhow::anyhow!("500 Internal Server Error");
        assert_eq!(
            RetryableError::classify(&server_err),
            RetryableError::ServerError
        );

        let client_err = anyhow::anyhow!("404 Not Found");
        assert_eq!(
            RetryableError::classify(&client_err),
            RetryableError::ClientError
        );
    }

    #[test]
    fn test_retryable_checks() {
        // Connection errors are always retryable
        assert!(RetryableError::Connection.is_retryable(true, false));
        assert!(RetryableError::Connection.is_retryable(true, true));

        // Timeout is only retryable for idempotent ops when idempotent_only=true
        assert!(RetryableError::Timeout.is_retryable(true, true));
        assert!(!RetryableError::Timeout.is_retryable(true, false));
        assert!(RetryableError::Timeout.is_retryable(false, false));

        // Client errors are never retryable
        assert!(!RetryableError::ClientError.is_retryable(false, true));
    }

    #[tokio::test]
    async fn test_retry_executor_success() {
        let executor = RetryExecutor::new(RetryConfig::with_max_retries(3));
        let result = executor
            .execute(true, || async { Ok::<_, anyhow::Error>(42) })
            .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_executor_retry_then_success() {
        let executor = RetryExecutor::new(
            RetryConfig::with_max_retries(3).initial_delay(Duration::from_millis(1)),
        );
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

        let counter_clone = counter.clone();
        let result = executor
            .execute(true, || {
                let count = counter_clone.clone();
                async move {
                    let n = count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if n < 2 {
                        Err(anyhow::anyhow!("Connection refused"))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
