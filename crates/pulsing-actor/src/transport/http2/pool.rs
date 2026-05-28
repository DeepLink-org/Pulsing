//! Connection pool management for HTTP/2 transport.

use super::config::Http2Config;
use crate::error::{PulsingError, Result, RuntimeError};
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
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock, Semaphore};

/// Connection pool statistics.
#[derive(Debug, Default)]
pub struct PoolStats {
    pub connections_created: AtomicU64,
    pub connections_closed: AtomicU64,
    pub connections_reused: AtomicU64,
    pub connection_errors: AtomicU64,
    pub pool_expansions: AtomicU64,
    pub active_connections: AtomicUsize,
    pub idle_connections: AtomicUsize,
}

impl PoolStats {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "connections_created": self.connections_created.load(Ordering::Relaxed),
            "connections_closed": self.connections_closed.load(Ordering::Relaxed),
            "connections_reused": self.connections_reused.load(Ordering::Relaxed),
            "connection_errors": self.connection_errors.load(Ordering::Relaxed),
            "pool_expansions": self.pool_expansions.load(Ordering::Relaxed),
            "active_connections": self.active_connections.load(Ordering::Relaxed),
            "idle_connections": self.idle_connections.load(Ordering::Relaxed),
        })
    }
}

/// Connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Idle,
    Active,
    Unhealthy,
    Expired,
}

/// A pooled HTTP/2 connection.
pub struct PooledConnection {
    pub sender: http2::SendRequest<Full<Bytes>>,
    pub created_at: Instant,
    pub last_used: Instant,
    pub request_count: u64,
    pub state: ConnectionState,
    /// Released on Drop; idle-reuse via `try_get_existing` does NOT consume new permits.
    _permit: OwnedSemaphorePermit,
}

impl PooledConnection {
    fn new(sender: http2::SendRequest<Full<Bytes>>, permit: OwnedSemaphorePermit) -> Self {
        let now = Instant::now();
        Self {
            sender,
            created_at: now,
            last_used: now,
            request_count: 0,
            state: ConnectionState::Idle,
            _permit: permit,
        }
    }

    pub fn is_healthy(&self, config: &PoolConfig) -> bool {
        if !self.sender.is_ready() {
            return false;
        }

        if let Some(max_age) = config.max_connection_age {
            if self.created_at.elapsed() > max_age {
                return false;
            }
        }

        if let Some(max_idle) = config.max_idle_time {
            if self.last_used.elapsed() > max_idle {
                return false;
            }
        }

        if let Some(max_requests) = config.max_requests_per_connection {
            if self.request_count >= max_requests {
                return false;
            }
        }

        true
    }

    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.request_count += 1;
        self.state = ConnectionState::Active;
    }

    pub fn mark_idle(&mut self) {
        self.state = ConnectionState::Idle;
    }
}

/// Pool configuration.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_connections_per_host: usize,

    pub min_idle_per_host: usize,

    pub max_total_connections: usize,

    pub connect_timeout: Duration,

    pub max_connection_age: Option<Duration>,

    pub max_idle_time: Option<Duration>,

    pub max_requests_per_connection: Option<u64>,

    pub cleanup_interval: Duration,

    pub enable_warming: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 8,
            min_idle_per_host: 1,
            max_total_connections: 100,
            connect_timeout: Duration::from_secs(5),
            max_connection_age: Some(Duration::from_secs(300)),
            max_idle_time: Some(Duration::from_secs(60)),
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
    /// Current per-host live connection cap
    current_cap: usize,
    /// Last connection error time
    last_error: Option<Instant>,
    /// Consecutive error count
    error_count: u32,
}

impl HostPool {
    fn new(initial_cap: usize) -> Self {
        let cap = initial_cap.clamp(1, Semaphore::MAX_PERMITS);

        Self {
            connections: Vec::with_capacity(cap),
            semaphore: Arc::new(Semaphore::new(cap)),
            current_cap: cap,
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

    /// Double the host pool capacity up to Tokio's semaphore limit.
    fn expand(&mut self) -> Option<(usize, usize)> {
        let old = self.current_cap;
        let target = old.saturating_mul(2).min(Semaphore::MAX_PERMITS);
        let added = target.saturating_sub(old);

        if added == 0 {
            return None;
        }

        self.semaphore.add_permits(added);
        self.current_cap = target;

        Some((old, target))
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
    pub async fn get_connection(&self, addr: SocketAddr) -> Result<ConnectionGuard> {
        // Try to get an existing healthy connection first
        if let Some(conn) = self.try_get_existing(addr).await {
            self.stats
                .connections_reused
                .fetch_add(1, Ordering::Relaxed);
            return Ok(conn);
        }

        // Create a new connection
        self.create_new_connection(addr).await.map_err(|e| {
            PulsingError::from(RuntimeError::connection_failed(
                addr.to_string(),
                e.to_string(),
            ))
        })
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

        // Acquire permit (limits live connections per host)
        let permit = match semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => self.expand_and_acquire(addr, &semaphore).await?,
        };

        // Create the connection
        let result = self.create_connection_inner(addr, permit).await;

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
                // invariant: permit was acquired above, so cap always has room
                if host_pool.connections.len() < host_pool.current_cap {
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

    async fn expand_and_acquire(
        &self,
        addr: SocketAddr,
        semaphore: &Arc<Semaphore>,
    ) -> anyhow::Result<OwnedSemaphorePermit> {
        // `did_expand_here` is true only when *this* task performed the expand;
        // a concurrent task may have expanded already, in which case we just
        // skip and wait on the existing permits.
        let did_expand_here = {
            let mut pools = self.pools.write().await;
            let host_pool = pools
                .get_mut(&addr)
                .ok_or_else(|| anyhow::anyhow!("Host pool for {} disappeared", addr))?;

            if host_pool.semaphore.available_permits() == 0 {
                match host_pool.expand() {
                    Some((old, new)) => {
                        tracing::warn!(
                            addr = %addr,
                            old_cap = old,
                            new_cap = new,
                            "HTTP/2 connection pool exhausted, expanding capacity 2x"
                        );
                        true
                    }
                    None => {
                        return Err(anyhow::anyhow!(
                            "Connection pool for {} reached Semaphore::MAX_PERMITS",
                            addr
                        ));
                    }
                }
            } else {
                false
            }
        };

        if did_expand_here {
            self.stats.pool_expansions.fetch_add(1, Ordering::Relaxed);
        }

        tokio::time::timeout(
            self.http2_config.connect_timeout,
            semaphore.clone().acquire_owned(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Timed out waiting for connection pool permit on {}", addr))?
        .map_err(|_| anyhow::anyhow!("Semaphore closed for {}", addr))
    }

    /// Create the actual TCP + HTTP/2 connection
    async fn create_connection_inner(
        &self,
        addr: SocketAddr,
        permit: OwnedSemaphorePermit,
    ) -> anyhow::Result<PooledConnection> {
        let stream =
            tokio::time::timeout(self.http2_config.connect_timeout, TcpStream::connect(addr))
                .await
                .map_err(|_| anyhow::anyhow!("Connection timeout to {}", addr))?
                .map_err(|e| anyhow::anyhow!("Connection failed to {}: {}", addr, e))?;

        // Set TCP options
        stream.set_nodelay(true)?;

        // Create HTTP/2 connection - with or without TLS
        #[cfg(feature = "tls")]
        if let Some(ref tls_config) = self.http2_config.tls {
            // TLS mode: wrap TCP stream with TLS
            let server_name = addr.ip().to_string();
            let tls_stream = tls_config
                .connect(stream, &server_name)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let io = TokioIo::new(tls_stream);
            let (sender, conn) = http2::handshake(TokioExecutor::new(), io)
                .await
                .map_err(|e| anyhow::anyhow!("HTTP/2 TLS handshake failed with {}: {}", addr, e))?;

            // Spawn connection driver for TLS connection
            tokio::spawn(async move {
                if let Err(e) = conn.await {
                    tracing::debug!(error = %e, "HTTP/2 TLS connection closed");
                }
            });

            let mut pooled = PooledConnection::new(sender, permit);
            pooled.mark_used();
            return Ok(pooled);
        }

        // Plain h2c mode (no TLS or TLS feature disabled)
        let io = TokioIo::new(stream);
        let (sender, conn) = http2::handshake(TokioExecutor::new(), io)
            .await
            .map_err(|e| anyhow::anyhow!("HTTP/2 handshake failed with {}: {}", addr, e))?;

        // Spawn connection driver
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                tracing::debug!(error = %e, "HTTP/2 connection closed");
            }
        });

        let mut pooled = PooledConnection::new(sender, permit);
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
    use hyper::service::service_fn;
    use hyper::{Request, Response};
    use std::convert::Infallible;
    use std::future::Future;
    use std::task::{Context, Poll};
    use tokio::net::TcpListener;

    async fn spawn_h2c_server() -> SocketAddr {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                let (stream, peer_addr) = match listener.accept().await {
                    Ok(accepted) => accepted,
                    Err(_) => break,
                };

                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(|_req: Request<hyper::body::Incoming>| async {
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::new())))
                    });
                    let builder = hyper::server::conn::http2::Builder::new(TokioExecutor::new());
                    let conn = builder.serve_connection(io, service);

                    if let Err(e) = conn.await {
                        tracing::debug!(
                            peer = %peer_addr,
                            error = %e,
                            "Test HTTP/2 connection closed"
                        );
                    }
                });
            }
        });

        addr
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections_per_host, 8);
        assert_eq!(config.min_idle_per_host, 1);
        assert!(config.max_connection_age.is_some());
    }

    #[test]
    fn test_pool_stats() {
        let stats = PoolStats::default();
        stats.connections_created.fetch_add(1, Ordering::Relaxed);
        stats.connections_reused.fetch_add(5, Ordering::Relaxed);
        stats.pool_expansions.fetch_add(2, Ordering::Relaxed);

        let json = stats.to_json();
        assert_eq!(json["connections_created"], 1);
        assert_eq!(json["connections_reused"], 5);
        assert_eq!(json["pool_expansions"], 2);
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

    #[test]
    fn test_host_pool_expand_doubles_capacity() {
        let mut pool = HostPool::new(2);
        let sem = pool.semaphore.clone();

        let _p1 = sem.clone().try_acquire_owned().unwrap();
        let _p2 = sem.clone().try_acquire_owned().unwrap();
        assert!(sem.clone().try_acquire_owned().is_err());

        let (old, new) = pool.expand().unwrap();
        assert_eq!((old, new), (2, 4));
        assert_eq!(pool.current_cap, 4);
        assert_eq!(sem.available_permits(), 2);

        let _p3 = sem.clone().try_acquire_owned().unwrap();
        let _p4 = sem.clone().try_acquire_owned().unwrap();
        assert!(sem.clone().try_acquire_owned().is_err());
    }

    #[test]
    fn test_host_pool_expand_respects_max_permits() {
        let mut pool = HostPool::new(1);
        pool.current_cap = Semaphore::MAX_PERMITS;

        assert!(pool.expand().is_none());
    }

    #[tokio::test]
    async fn test_expand_and_acquire_times_out_when_expanded_permit_is_taken() {
        let addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        let http2_config = Http2Config::default().connect_timeout(Duration::from_millis(10));
        let pool = ConnectionPool::new(http2_config);

        let semaphore = {
            let mut pools = pool.pools.write().await;
            let host_pool = pools.entry(addr).or_insert_with(|| HostPool::new(1));
            host_pool.semaphore.clone()
        };

        let _held = semaphore.clone().try_acquire_owned().unwrap();
        let mut queued = Box::pin(semaphore.clone().acquire_owned());
        let waker = futures::task::noop_waker_ref();
        let mut cx = Context::from_waker(waker);
        assert!(matches!(queued.as_mut().poll(&mut cx), Poll::Pending));

        let err = pool.expand_and_acquire(addr, &semaphore).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Timed out waiting for connection pool permit"),
            "unexpected error: {err}"
        );
        assert_eq!(pool.stats.pool_expansions.load(Ordering::Relaxed), 1);

        let queued_permit = queued.await.unwrap();
        drop(queued_permit);
    }

    #[tokio::test]
    async fn test_expand_and_acquire_uses_existing_permit_without_expanding() {
        let addr: SocketAddr = "127.0.0.1:12346".parse().unwrap();
        let pool = ConnectionPool::new(Http2Config::default());

        let semaphore = {
            let mut pools = pool.pools.write().await;
            let host_pool = pools.entry(addr).or_insert_with(|| HostPool::new(1));
            host_pool.semaphore.clone()
        };

        let permit = pool.expand_and_acquire(addr, &semaphore).await.unwrap();
        assert_eq!(pool.stats.pool_expansions.load(Ordering::Relaxed), 0);
        assert_eq!(semaphore.available_permits(), 0);

        drop(permit);
        assert_eq!(semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn test_expand_and_acquire_errors_when_host_pool_disappears() {
        let addr: SocketAddr = "127.0.0.1:12347".parse().unwrap();
        let pool = ConnectionPool::new(Http2Config::default());
        let semaphore = Arc::new(Semaphore::new(1));

        let err = pool.expand_and_acquire(addr, &semaphore).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("Host pool for 127.0.0.1:12347 disappeared"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_expand_and_acquire_errors_at_semaphore_max_permits() {
        let addr: SocketAddr = "127.0.0.1:12348".parse().unwrap();
        let pool = ConnectionPool::new(Http2Config::default());

        let (semaphore, _held) = {
            let mut pools = pool.pools.write().await;
            let host_pool = pools.entry(addr).or_insert_with(|| HostPool::new(1));
            let semaphore = host_pool.semaphore.clone();
            let held = semaphore.clone().try_acquire_owned().unwrap();
            host_pool.current_cap = Semaphore::MAX_PERMITS;
            (semaphore, held)
        };

        let err = pool.expand_and_acquire(addr, &semaphore).await.unwrap_err();
        assert!(
            err.to_string().contains("reached Semaphore::MAX_PERMITS"),
            "unexpected error: {err}"
        );
        assert_eq!(pool.stats.pool_expansions.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_expand_and_acquire_errors_when_semaphore_is_closed() {
        let addr: SocketAddr = "127.0.0.1:12349".parse().unwrap();
        let pool = ConnectionPool::new(Http2Config::default());

        let semaphore = {
            let mut pools = pool.pools.write().await;
            let host_pool = pools.entry(addr).or_insert_with(|| HostPool::new(1));
            host_pool.semaphore.clone()
        };
        semaphore.close();

        let err = pool.expand_and_acquire(addr, &semaphore).await.unwrap_err();
        assert!(
            err.to_string().contains("Semaphore closed"),
            "unexpected error: {err}"
        );
        assert_eq!(pool.stats.pool_expansions.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let pool = ConnectionPool::new(Http2Config::default());
        let stats = pool.stats();
        assert_eq!(stats.connections_created.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_connection_pool_expands_and_releases_permits() {
        let addr = spawn_h2c_server().await;
        let http2_config = Http2Config::default().connect_timeout(Duration::from_secs(2));
        let pool_config = PoolConfig {
            max_connections_per_host: 2,
            max_connection_age: None,
            max_idle_time: Some(Duration::from_millis(10)),
            max_requests_per_connection: None,
            ..Default::default()
        };
        let pool = ConnectionPool::with_config(http2_config, pool_config);

        let (g1, g2, g3, g4) = tokio::join!(
            pool.get_connection(addr),
            pool.get_connection(addr),
            pool.get_connection(addr),
            pool.get_connection(addr),
        );
        let guards = vec![g1.unwrap(), g2.unwrap(), g3.unwrap(), g4.unwrap()];

        let info = pool.pool_info().await;
        let hosts = info["hosts"].as_array().unwrap();
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0]["total_connections"].as_u64().unwrap(), 4);
        assert!(
            info["stats"]["pool_expansions"].as_u64().unwrap() >= 1,
            "pool_info did not report an expansion: {info}"
        );

        drop(guards);
        // Wait past max_idle_time so cleanup() will treat all conns as stale.
        tokio::time::sleep(Duration::from_millis(20)).await;
        pool.cleanup().await;

        let (total_connections, current_cap, semaphore) = {
            let pools = pool.pools.read().await;
            let host_pool = pools.get(&addr).unwrap();
            (
                host_pool.connections.len(),
                host_pool.current_cap,
                host_pool.semaphore.clone(),
            )
        };

        assert_eq!(total_connections, 0);
        assert_eq!(semaphore.available_permits(), current_cap);
        let _permit = semaphore.try_acquire_owned().unwrap();
    }
}
