//! Advanced connection pool management for HTTP/2 transport
//!
//! Features:
//! - Connection health checking
//! - Connection expiration/eviction
//! - Connection reuse optimization
//! - Pool statistics

use super::config::Http2Config;
use bytes::Bytes;
use http_body_util::Full;
use hyper::client::conn::http2;
use hyper_util::rt::{TokioExecutor, TokioIo};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock, Semaphore};

/// Connection pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Total connections created
    pub connections_created: AtomicU64,
    /// Total connections closed
    pub connections_closed: AtomicU64,
    /// Total connections reused
    pub connections_reused: AtomicU64,
    /// Total connection errors
    pub connection_errors: AtomicU64,
    /// Current active connections
    pub active_connections: AtomicUsize,
    /// Current idle connections
    pub idle_connections: AtomicUsize,
}

impl PoolStats {
    /// Get stats as JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "connections_created": self.connections_created.load(Ordering::Relaxed),
            "connections_closed": self.connections_closed.load(Ordering::Relaxed),
            "connections_reused": self.connections_reused.load(Ordering::Relaxed),
            "connection_errors": self.connection_errors.load(Ordering::Relaxed),
            "active_connections": self.active_connections.load(Ordering::Relaxed),
            "idle_connections": self.idle_connections.load(Ordering::Relaxed),
        })
    }
}

/// Connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is available
    Idle,
    /// Connection is in use
    Active,
    /// Connection is unhealthy
    Unhealthy,
    /// Connection has expired
    Expired,
}

/// A pooled HTTP/2 connection
pub struct PooledConnection {
    /// The HTTP/2 sender
    pub sender: http2::SendRequest<Full<Bytes>>,
    /// When the connection was created
    pub created_at: Instant,
    /// When the connection was last used
    pub last_used: Instant,
    /// Number of requests made on this connection
    pub request_count: u64,
    /// Current state
    pub state: ConnectionState,
}

impl PooledConnection {
    fn new(sender: http2::SendRequest<Full<Bytes>>) -> Self {
        let now = Instant::now();
        Self {
            sender,
            created_at: now,
            last_used: now,
            request_count: 0,
            state: ConnectionState::Idle,
        }
    }

    /// Check if the connection is healthy
    pub fn is_healthy(&self, config: &PoolConfig) -> bool {
        // Check if connection is ready
        if !self.sender.is_ready() {
            return false;
        }

        // Check max age
        if let Some(max_age) = config.max_connection_age {
            if self.created_at.elapsed() > max_age {
                return false;
            }
        }

        // Check max idle time
        if let Some(max_idle) = config.max_idle_time {
            if self.last_used.elapsed() > max_idle {
                return false;
            }
        }

        // Check max requests per connection
        if let Some(max_requests) = config.max_requests_per_connection {
            if self.request_count >= max_requests {
                return false;
            }
        }

        true
    }

    /// Mark the connection as used
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.request_count += 1;
        self.state = ConnectionState::Active;
    }

    /// Mark the connection as idle
    pub fn mark_idle(&mut self) {
        self.state = ConnectionState::Idle;
    }
}

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum connections per host
    pub max_connections_per_host: usize,

    /// Minimum idle connections per host
    pub min_idle_per_host: usize,

    /// Maximum total connections
    pub max_total_connections: usize,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Maximum connection age (None = no limit)
    pub max_connection_age: Option<Duration>,

    /// Maximum idle time (None = no limit)
    pub max_idle_time: Option<Duration>,

    /// Maximum requests per connection (None = no limit)
    pub max_requests_per_connection: Option<u64>,

    /// How often to run cleanup
    pub cleanup_interval: Duration,

    /// Whether to enable connection warming
    pub enable_warming: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 10,
            min_idle_per_host: 1,
            max_total_connections: 100,
            connect_timeout: Duration::from_secs(5),
            max_connection_age: Some(Duration::from_secs(300)), // 5 minutes
            max_idle_time: Some(Duration::from_secs(60)),       // 1 minute
            max_requests_per_connection: Some(1000),
            cleanup_interval: Duration::from_secs(30),
            enable_warming: false,
        }
    }
}

impl From<&Http2Config> for PoolConfig {
    fn from(config: &Http2Config) -> Self {
        Self {
            max_connections_per_host: config.max_connections_per_host,
            connect_timeout: config.connect_timeout,
            ..Default::default()
        }
    }
}

/// Host-specific connection pool
struct HostPool {
    /// Available connections
    connections: Vec<Arc<Mutex<PooledConnection>>>,
    /// Semaphore for limiting concurrent connections
    semaphore: Arc<Semaphore>,
    /// Last connection error time
    last_error: Option<Instant>,
    /// Consecutive error count
    error_count: u32,
}

impl HostPool {
    fn new(max_connections: usize) -> Self {
        Self {
            connections: Vec::with_capacity(max_connections),
            semaphore: Arc::new(Semaphore::new(max_connections)),
            last_error: None,
            error_count: 0,
        }
    }

    fn record_error(&mut self) {
        self.last_error = Some(Instant::now());
        self.error_count = self.error_count.saturating_add(1);
    }

    fn record_success(&mut self) {
        self.error_count = 0;
    }

    /// Check if we should back off from connecting to this host
    fn should_backoff(&self) -> bool {
        if let Some(last_error) = self.last_error {
            // Exponential backoff based on error count
            let backoff_ms = (100 * 2_u32.saturating_pow(self.error_count.min(6))) as u64;
            last_error.elapsed() < Duration::from_millis(backoff_ms)
        } else {
            false
        }
    }
}

/// Advanced connection pool
pub struct ConnectionPool {
    /// Per-host connection pools
    pools: RwLock<HashMap<SocketAddr, HostPool>>,
    /// Pool configuration
    config: PoolConfig,
    /// HTTP/2 configuration
    http2_config: Http2Config,
    /// Statistics
    stats: Arc<PoolStats>,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(http2_config: Http2Config) -> Self {
        let config = PoolConfig::from(&http2_config);
        Self {
            pools: RwLock::new(HashMap::new()),
            config,
            http2_config,
            stats: Arc::new(PoolStats::default()),
        }
    }

    /// Create with custom pool config
    pub fn with_config(http2_config: Http2Config, pool_config: PoolConfig) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            config: pool_config,
            http2_config,
            stats: Arc::new(PoolStats::default()),
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> &Arc<PoolStats> {
        &self.stats
    }

    /// Get or create a connection to the given address
    pub async fn get_connection(&self, addr: SocketAddr) -> anyhow::Result<ConnectionGuard> {
        // Try to get an existing healthy connection first
        if let Some(conn) = self.try_get_existing(addr).await {
            self.stats
                .connections_reused
                .fetch_add(1, Ordering::Relaxed);
            return Ok(conn);
        }

        // Create a new connection
        self.create_new_connection(addr).await
    }

    /// Try to get an existing healthy connection
    async fn try_get_existing(&self, addr: SocketAddr) -> Option<ConnectionGuard> {
        let pools = self.pools.read().await;

        if let Some(host_pool) = pools.get(&addr) {
            for conn in &host_pool.connections {
                // Try to lock without blocking
                if let Ok(mut guard) = conn.try_lock() {
                    if guard.state == ConnectionState::Idle && guard.is_healthy(&self.config) {
                        guard.mark_used();
                        self.stats
                            .active_connections
                            .fetch_add(1, Ordering::Relaxed);
                        self.stats.idle_connections.fetch_sub(1, Ordering::Relaxed);
                        return Some(ConnectionGuard {
                            conn: conn.clone(),
                            stats: self.stats.clone(),
                        });
                    }
                }
            }
        }

        None
    }

    /// Create a new connection
    async fn create_new_connection(&self, addr: SocketAddr) -> anyhow::Result<ConnectionGuard> {
        // Check if we should back off
        {
            let pools = self.pools.read().await;
            if let Some(host_pool) = pools.get(&addr) {
                if host_pool.should_backoff() {
                    return Err(anyhow::anyhow!(
                        "Connection to {} is backing off due to recent errors",
                        addr
                    ));
                }
            }
        }

        // Ensure host pool exists and get semaphore
        let semaphore = {
            let mut pools = self.pools.write().await;
            let host_pool = pools
                .entry(addr)
                .or_insert_with(|| HostPool::new(self.config.max_connections_per_host));
            host_pool.semaphore.clone()
        };

        // Acquire permit (limits concurrent connections per host)
        let _permit = semaphore
            .try_acquire()
            .map_err(|_| anyhow::anyhow!("Connection pool exhausted for {}", addr))?;

        // Create the connection
        let result = self.create_connection_inner(addr).await;

        // Update host pool state
        {
            let mut pools = self.pools.write().await;
            if let Some(host_pool) = pools.get_mut(&addr) {
                match &result {
                    Ok(_) => host_pool.record_success(),
                    Err(_) => {
                        host_pool.record_error();
                        self.stats.connection_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        let conn = result?;
        let conn = Arc::new(Mutex::new(conn));

        // Add to pool
        {
            let mut pools = self.pools.write().await;
            if let Some(host_pool) = pools.get_mut(&addr) {
                if host_pool.connections.len() < self.config.max_connections_per_host {
                    host_pool.connections.push(conn.clone());
                }
            }
        }

        self.stats
            .connections_created
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .active_connections
            .fetch_add(1, Ordering::Relaxed);

        Ok(ConnectionGuard {
            conn,
            stats: self.stats.clone(),
        })
    }

    /// Create the actual TCP + HTTP/2 connection
    async fn create_connection_inner(&self, addr: SocketAddr) -> anyhow::Result<PooledConnection> {
        let stream =
            tokio::time::timeout(self.http2_config.connect_timeout, TcpStream::connect(addr))
                .await
                .map_err(|_| anyhow::anyhow!("Connection timeout to {}", addr))?
                .map_err(|e| anyhow::anyhow!("Connection failed to {}: {}", addr, e))?;

        // Set TCP options
        stream.set_nodelay(true)?;

        let io = TokioIo::new(stream);

        // Create HTTP/2 connection with prior knowledge (h2c)
        let (sender, conn) = http2::handshake(TokioExecutor::new(), io)
            .await
            .map_err(|e| anyhow::anyhow!("HTTP/2 handshake failed with {}: {}", addr, e))?;

        // Spawn connection driver
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                tracing::debug!(error = %e, "HTTP/2 connection closed");
            }
        });

        let mut pooled = PooledConnection::new(sender);
        pooled.mark_used();

        Ok(pooled)
    }

    /// Run cleanup to remove unhealthy connections
    pub async fn cleanup(&self) {
        let mut pools = self.pools.write().await;
        let mut total_removed = 0;

        for (addr, host_pool) in pools.iter_mut() {
            let before_len = host_pool.connections.len();

            // Remove unhealthy connections
            host_pool.connections.retain(|conn| {
                if let Ok(guard) = conn.try_lock() {
                    guard.is_healthy(&self.config)
                } else {
                    true // Keep connections that are in use
                }
            });

            let removed = before_len - host_pool.connections.len();
            if removed > 0 {
                tracing::debug!(
                    addr = %addr,
                    removed = removed,
                    remaining = host_pool.connections.len(),
                    "Cleaned up connections"
                );
                total_removed += removed;
            }
        }

        if total_removed > 0 {
            self.stats
                .connections_closed
                .fetch_add(total_removed as u64, Ordering::Relaxed);
        }
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(self: &Arc<Self>, cancel: tokio_util::sync::CancellationToken) {
        let pool = Arc::clone(self);
        let interval = self.config.cleanup_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        pool.cleanup().await;
                    }
                    _ = cancel.cancelled() => {
                        tracing::debug!("Connection pool cleanup task stopped");
                        break;
                    }
                }
            }
        });
    }

    /// Get pool info for diagnostics
    pub async fn pool_info(&self) -> serde_json::Value {
        let pools = self.pools.read().await;
        let hosts: Vec<_> = pools
            .iter()
            .map(|(addr, pool)| {
                let healthy = pool
                    .connections
                    .iter()
                    .filter(|c| {
                        c.try_lock()
                            .map(|g| g.is_healthy(&self.config))
                            .unwrap_or(false)
                    })
                    .count();

                serde_json::json!({
                    "address": addr.to_string(),
                    "total_connections": pool.connections.len(),
                    "healthy_connections": healthy,
                    "error_count": pool.error_count,
                })
            })
            .collect();

        serde_json::json!({
            "hosts": hosts,
            "stats": self.stats.to_json(),
        })
    }
}

/// RAII guard for a pooled connection
pub struct ConnectionGuard {
    conn: Arc<Mutex<PooledConnection>>,
    stats: Arc<PoolStats>,
}

impl ConnectionGuard {
    /// Get mutable access to the connection
    pub async fn get(&self) -> tokio::sync::MutexGuard<'_, PooledConnection> {
        self.conn.lock().await
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        // Mark connection as idle when guard is dropped
        if let Ok(mut guard) = self.conn.try_lock() {
            guard.mark_idle();
        }
        self.stats
            .active_connections
            .fetch_sub(1, Ordering::Relaxed);
        self.stats.idle_connections.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections_per_host, 10);
        assert_eq!(config.min_idle_per_host, 1);
        assert!(config.max_connection_age.is_some());
    }

    #[test]
    fn test_pool_stats() {
        let stats = PoolStats::default();
        stats.connections_created.fetch_add(1, Ordering::Relaxed);
        stats.connections_reused.fetch_add(5, Ordering::Relaxed);

        let json = stats.to_json();
        assert_eq!(json["connections_created"], 1);
        assert_eq!(json["connections_reused"], 5);
    }

    #[test]
    fn test_host_pool_backoff() {
        let mut pool = HostPool::new(10);

        // No backoff initially
        assert!(!pool.should_backoff());

        // Record errors
        pool.record_error();
        assert!(pool.should_backoff());

        // Success resets
        pool.record_success();
        assert_eq!(pool.error_count, 0);
    }

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let pool = ConnectionPool::new(Http2Config::default());
        let stats = pool.stats();
        assert_eq!(stats.connections_created.load(Ordering::Relaxed), 0);
    }
}
