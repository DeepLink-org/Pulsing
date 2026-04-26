//! In-memory ring buffer of recent system metric snapshots.
//!
//! **Deprecated**: prefer `SELECT * FROM pulsing.metrics` via probing's DataFusion engine.

use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Default max snapshots retained per node.
pub const DEFAULT_PERFORMANCE_HISTORY_CAPACITY: usize = 4096;

/// One point-in-time row aligned with `GetMetrics` / dashboard metrics.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PerformanceSnapshot {
    /// Unix time in microseconds (UTC).
    pub ts_unix_micros: u64,
    pub node_id: String,
    pub actors_count: u64,
    pub messages_total: u64,
    pub actors_created: u64,
    pub actors_stopped: u64,
    pub uptime_secs: u64,
}

/// Bounded FIFO of [`PerformanceSnapshot`] (thread-safe).
pub struct PerformanceStore {
    max: usize,
    inner: Mutex<VecDeque<PerformanceSnapshot>>,
}

impl PerformanceStore {
    pub fn new(capacity: usize) -> Self {
        let max = capacity.max(1);
        Self {
            max,
            inner: Mutex::new(VecDeque::new()),
        }
    }

    pub fn capacity(&self) -> usize {
        self.max
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().map(|g| g.is_empty()).unwrap_or(true)
    }

    pub fn record(&self, snapshot: PerformanceSnapshot) {
        if let Ok(mut g) = self.inner.lock() {
            if g.len() >= self.max {
                g.pop_front();
            }
            g.push_back(snapshot);
        }
    }

    /// Newest-first, at most `limit` items.
    pub fn recent(&self, limit: usize) -> Vec<PerformanceSnapshot> {
        let Ok(g) = self.inner.lock() else {
            return Vec::new();
        };
        let n = limit.min(g.len());
        g.iter().rev().take(n).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_drops_oldest() {
        let s = PerformanceStore::new(2);
        for i in 0..3 {
            s.record(PerformanceSnapshot {
                ts_unix_micros: i,
                node_id: "n".into(),
                actors_count: i,
                messages_total: 0,
                actors_created: 0,
                actors_stopped: 0,
                uptime_secs: 0,
            });
        }
        assert_eq!(s.len(), 2);
        let r = s.recent(10);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].ts_unix_micros, 2);
        assert_eq!(r[1].ts_unix_micros, 1);
    }
}
