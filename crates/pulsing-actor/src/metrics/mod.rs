//! Metrics module for Pulsing Actor System
//!
//! Provides Prometheus-compatible metrics collection and exposition.
//!
//! ## Features
//!
//! - System-level metrics (actor count, uptime, cluster members)
//! - Actor-level metrics (mailbox depth, message count, latency)
//! - Transport-level metrics (HTTP requests, connections)
//! - Prometheus text format output
//!
//! ## Usage
//!
//! Metrics are automatically collected and exposed via `/metrics` endpoint.
//! No application code changes required.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Global metrics registry
pub struct MetricsRegistry {
    start_time: Instant,
    counters: dashmap::DashMap<String, AtomicU64>,
    gauges: dashmap::DashMap<String, AtomicU64>,
    /// Histogram data: metric_name -> (sum, count, buckets)
    histograms: dashmap::DashMap<String, HistogramData>,
}

/// Histogram bucket configuration
const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Histogram data storage
struct HistogramData {
    sum: AtomicU64,
    count: AtomicU64,
    buckets: Vec<AtomicU64>,
    bucket_bounds: Vec<f64>,
}

/// Histogram snapshot for export (avoids borrow issues)
struct HistogramSnapshot {
    sum: u64,
    count: u64,
    buckets: Vec<u64>,
    bucket_bounds: Vec<f64>,
}

impl HistogramData {
    fn new(bounds: &[f64]) -> Self {
        Self {
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
            buckets: bounds.iter().map(|_| AtomicU64::new(0)).collect(),
            bucket_bounds: bounds.to_vec(),
        }
    }

    fn observe(&self, value: f64) {
        // Update sum (store as micros for precision)
        let micros = (value * 1_000_000.0) as u64;
        self.sum.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Update buckets
        for (i, bound) in self.bucket_bounds.iter().enumerate() {
            if value <= *bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            counters: dashmap::DashMap::new(),
            gauges: dashmap::DashMap::new(),
            histograms: dashmap::DashMap::new(),
        }
    }

    /// Increment a counter
    pub fn counter_inc(&self, name: &str, labels: &[(&str, &str)]) {
        let key = format_key(name, labels);
        self.counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Add value to a counter
    pub fn counter_add(&self, name: &str, labels: &[(&str, &str)], value: u64) {
        let key = format_key(name, labels);
        self.counters
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(value, Ordering::Relaxed);
    }

    /// Set a gauge value
    pub fn gauge_set(&self, name: &str, labels: &[(&str, &str)], value: u64) {
        let key = format_key(name, labels);
        self.gauges
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .store(value, Ordering::Relaxed);
    }

    /// Increment a gauge
    pub fn gauge_inc(&self, name: &str, labels: &[(&str, &str)]) {
        let key = format_key(name, labels);
        self.gauges
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement a gauge
    pub fn gauge_dec(&self, name: &str, labels: &[(&str, &str)]) {
        let key = format_key(name, labels);
        self.gauges
            .entry(key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_sub(1, Ordering::Relaxed);
    }

    /// Observe a histogram value (in seconds)
    pub fn histogram_observe(&self, name: &str, labels: &[(&str, &str)], value: f64) {
        let key = format_key(name, labels);
        self.histograms
            .entry(key)
            .or_insert_with(|| HistogramData::new(LATENCY_BUCKETS))
            .observe(value);
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus(&self, extra_metrics: &SystemMetrics) -> String {
        let mut output = String::new();

        // System metrics from SystemActor
        output.push_str("# HELP pulsing_actors_total Current number of actors\n");
        output.push_str("# TYPE pulsing_actors_total gauge\n");
        output.push_str(&format!(
            "pulsing_actors_total{{node_id=\"{}\"}} {}\n",
            extra_metrics.node_id, extra_metrics.actors_count
        ));

        output.push_str("# HELP pulsing_uptime_seconds Node uptime in seconds\n");
        output.push_str("# TYPE pulsing_uptime_seconds counter\n");
        output.push_str(&format!(
            "pulsing_uptime_seconds{{node_id=\"{}\"}} {}\n",
            extra_metrics.node_id,
            self.uptime_secs()
        ));

        output.push_str("# HELP pulsing_messages_total Total messages processed\n");
        output.push_str("# TYPE pulsing_messages_total counter\n");
        output.push_str(&format!(
            "pulsing_messages_total{{node_id=\"{}\"}} {}\n",
            extra_metrics.node_id, extra_metrics.messages_total
        ));

        output.push_str("# HELP pulsing_actors_created_total Total actors created\n");
        output.push_str("# TYPE pulsing_actors_created_total counter\n");
        output.push_str(&format!(
            "pulsing_actors_created_total{{node_id=\"{}\"}} {}\n",
            extra_metrics.node_id, extra_metrics.actors_created
        ));

        output.push_str("# HELP pulsing_actors_stopped_total Total actors stopped\n");
        output.push_str("# TYPE pulsing_actors_stopped_total counter\n");
        output.push_str(&format!(
            "pulsing_actors_stopped_total{{node_id=\"{}\"}} {}\n",
            extra_metrics.node_id, extra_metrics.actors_stopped
        ));

        output.push_str("# HELP pulsing_cluster_members Number of cluster members\n");
        output.push_str("# TYPE pulsing_cluster_members gauge\n");
        for (status, count) in &extra_metrics.cluster_members {
            output.push_str(&format!(
                "pulsing_cluster_members{{node_id=\"{}\",status=\"{}\"}} {}\n",
                extra_metrics.node_id, status, count
            ));
        }

        // Export collected counters
        if !self.counters.is_empty() {
            let mut counter_groups: HashMap<String, Vec<(String, u64)>> = HashMap::new();
            for entry in self.counters.iter() {
                let (name, labels) = parse_key(entry.key());
                let value = entry.value().load(Ordering::Relaxed);
                counter_groups
                    .entry(name)
                    .or_default()
                    .push((labels, value));
            }

            for (name, values) in counter_groups {
                output.push_str(&format!("# TYPE {} counter\n", name));
                for (labels, value) in values {
                    if labels.is_empty() {
                        output.push_str(&format!("{} {}\n", name, value));
                    } else {
                        output.push_str(&format!("{}{{{}}} {}\n", name, labels, value));
                    }
                }
            }
        }

        // Export collected gauges
        if !self.gauges.is_empty() {
            let mut gauge_groups: HashMap<String, Vec<(String, u64)>> = HashMap::new();
            for entry in self.gauges.iter() {
                let (name, labels) = parse_key(entry.key());
                let value = entry.value().load(Ordering::Relaxed);
                gauge_groups.entry(name).or_default().push((labels, value));
            }

            for (name, values) in gauge_groups {
                output.push_str(&format!("# TYPE {} gauge\n", name));
                for (labels, value) in values {
                    if labels.is_empty() {
                        output.push_str(&format!("{} {}\n", name, value));
                    } else {
                        output.push_str(&format!("{}{{{}}} {}\n", name, labels, value));
                    }
                }
            }
        }

        // Export histograms - collect snapshot data first to avoid borrow issues
        if !self.histograms.is_empty() {
            // Collect histogram data snapshots
            let histogram_snapshots: Vec<(String, String, HistogramSnapshot)> = self
                .histograms
                .iter()
                .map(|entry| {
                    let (name, labels) = parse_key(entry.key());
                    let data = entry.value();
                    let snapshot = HistogramSnapshot {
                        sum: data.sum.load(Ordering::Relaxed),
                        count: data.count.load(Ordering::Relaxed),
                        buckets: data
                            .buckets
                            .iter()
                            .map(|b| b.load(Ordering::Relaxed))
                            .collect(),
                        bucket_bounds: data.bucket_bounds.clone(),
                    };
                    (name, labels, snapshot)
                })
                .collect();

            // Group by metric name
            let mut histogram_groups: HashMap<String, Vec<(String, HistogramSnapshot)>> =
                HashMap::new();
            for (name, labels, snapshot) in histogram_snapshots {
                histogram_groups
                    .entry(name)
                    .or_default()
                    .push((labels, snapshot));
            }

            for (name, values) in histogram_groups {
                output.push_str(&format!("# TYPE {} histogram\n", name));
                for (labels, data) in values {
                    let label_prefix = if labels.is_empty() {
                        String::new()
                    } else {
                        format!("{},", labels)
                    };

                    // Buckets
                    let mut cumulative = 0u64;
                    for (i, bound) in data.bucket_bounds.iter().enumerate() {
                        cumulative += data.buckets[i];
                        output.push_str(&format!(
                            "{}_bucket{{{}le=\"{}\"}} {}\n",
                            name, label_prefix, bound, cumulative
                        ));
                    }
                    output.push_str(&format!(
                        "{}_bucket{{{}le=\"+Inf\"}} {}\n",
                        name, label_prefix, data.count
                    ));

                    // Sum (convert from micros back to seconds)
                    let sum_secs = data.sum as f64 / 1_000_000.0;
                    if labels.is_empty() {
                        output.push_str(&format!("{}_sum {:.6}\n", name, sum_secs));
                        output.push_str(&format!("{}_count {}\n", name, data.count));
                    } else {
                        output.push_str(&format!("{}_sum{{{}}} {:.6}\n", name, labels, sum_secs));
                        output.push_str(&format!("{}_count{{{}}} {}\n", name, labels, data.count));
                    }
                }
            }
        }

        output
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// System-level metrics collected from SystemActor
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub node_id: u128,
    pub actors_count: usize,
    pub messages_total: u64,
    pub actors_created: u64,
    pub actors_stopped: u64,
    pub cluster_members: HashMap<String, usize>,
}

/// Format metric key with labels
fn format_key(name: &str, labels: &[(&str, &str)]) -> String {
    if labels.is_empty() {
        name.to_string()
    } else {
        let label_str: Vec<String> = labels
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();
        format!("{}:{}", name, label_str.join(","))
    }
}

/// Parse key back to name and labels
fn parse_key(key: &str) -> (String, String) {
    if let Some(pos) = key.find(':') {
        (key[..pos].to_string(), key[pos + 1..].to_string())
    } else {
        (key.to_string(), String::new())
    }
}

/// Global metrics registry instance
static METRICS: std::sync::OnceLock<Arc<MetricsRegistry>> = std::sync::OnceLock::new();

/// Get or initialize global metrics registry
pub fn metrics() -> &'static Arc<MetricsRegistry> {
    METRICS.get_or_init(|| Arc::new(MetricsRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_inc() {
        let registry = MetricsRegistry::new();
        registry.counter_inc("test_counter", &[]);
        registry.counter_inc("test_counter", &[]);

        let key = format_key("test_counter", &[]);
        assert_eq!(
            registry.counters.get(&key).unwrap().load(Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn test_counter_with_labels() {
        let registry = MetricsRegistry::new();
        registry.counter_inc("http_requests", &[("method", "GET"), ("path", "/api")]);
        registry.counter_inc("http_requests", &[("method", "POST"), ("path", "/api")]);

        assert_eq!(registry.counters.len(), 2);
    }

    #[test]
    fn test_gauge() {
        let registry = MetricsRegistry::new();
        registry.gauge_set("active_connections", &[], 10);
        registry.gauge_inc("active_connections", &[]);
        registry.gauge_dec("active_connections", &[]);

        let key = format_key("active_connections", &[]);
        assert_eq!(
            registry.gauges.get(&key).unwrap().load(Ordering::Relaxed),
            10
        );
    }

    #[test]
    fn test_histogram() {
        let registry = MetricsRegistry::new();
        registry.histogram_observe("request_duration", &[], 0.05);
        registry.histogram_observe("request_duration", &[], 0.15);
        registry.histogram_observe("request_duration", &[], 0.5);

        let key = format_key("request_duration", &[]);
        let data = registry.histograms.get(&key).unwrap();
        assert_eq!(data.count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_prometheus_export() {
        let registry = MetricsRegistry::new();
        registry.counter_inc("test_counter", &[("label", "value")]);
        registry.gauge_set("test_gauge", &[], 42);

        let system_metrics = SystemMetrics {
            node_id: 12345,
            actors_count: 10,
            messages_total: 1000,
            actors_created: 15,
            actors_stopped: 5,
            cluster_members: [("Alive".to_string(), 3)].into_iter().collect(),
        };

        let output = registry.export_prometheus(&system_metrics);
        assert!(output.contains("pulsing_actors_total"));
        assert!(output.contains("pulsing_uptime_seconds"));
        assert!(output.contains("test_counter"));
        assert!(output.contains("test_gauge"));
    }
}
