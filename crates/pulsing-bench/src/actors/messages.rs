//! Message types for Actor-based benchmark
//!
//! Defines all messages exchanged between benchmark actors.

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ============================================================================
// Coordinator Messages
// ============================================================================

/// Start the benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartBenchmark;

/// Stop the benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopBenchmark;

/// Benchmark completed (report is returned directly, not through message)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComplete {
    pub success: bool,
    pub message: String,
}

/// Progress update from collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressUpdate {
    pub phase: String,
    pub progress: f64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub requests_throughput: f64,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tpot_ms: Option<f64>,
    pub input_throughput: Option<f64>,
    pub output_throughput: Option<f64>,
}

// ============================================================================
// Executor Messages
// ============================================================================

/// Start executor with configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartExecutor {
    pub id: String,
    pub executor_type: String, // "constant_vus" or "constant_arrival_rate"
    pub max_vus: u64,
    pub duration_secs: u64,
    pub rate: Option<f64>,
    pub collector_actor: String, // Actor name to send results to
}

/// Request completed result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestResult {
    pub request_id: String,
    pub success: bool,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
    pub time_to_first_token_ms: Option<u64>,
    pub inter_token_latencies_ms: Vec<u64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub error: Option<String>,
}

/// Executor finished
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorFinished {
    pub id: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
}

// ============================================================================
// Collector Messages
// ============================================================================

/// Initialize collector for a benchmark phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitCollector {
    pub phase_id: String,
    pub executor_type: String,
    pub max_vus: u64,
    pub duration_secs: u64,
    pub rate: Option<f64>,
}

/// Get current results from collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetResults;

/// Results response (simplified, no BenchmarkResults since it contains non-serializable types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsResponse {
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_tokens: u64,
    pub duration_secs: f64,
}

/// Reset collector for new phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResetCollector {
    pub phase_id: String,
}

/// Finalize phase and get results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizePhase {
    pub phase_id: String,
}

/// Phase finalized response (simplified stats)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseFinalized {
    pub phase_id: String,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub duration_secs: f64,
    pub throughput: f64,
}

// ============================================================================
// Backend Messages (for VU actors)
// ============================================================================

/// Execute a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: u64,
}

/// Request started
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestStarted {
    pub request_id: String,
    pub start_time_ms: u64,
}

/// First token received
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstTokenReceived {
    pub request_id: String,
    pub time_ms: u64,
}

/// Token received
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenReceived {
    pub request_id: String,
    pub token_idx: u64,
    pub time_ms: u64,
}

/// Request completed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCompleted {
    pub request_id: String,
    pub success: bool,
    pub end_time_ms: u64,
    pub total_tokens: u64,
    pub error: Option<String>,
}

// ============================================================================
// Helper functions
// ============================================================================

/// Get current timestamp in milliseconds
pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl StartExecutor {
    pub fn constant_vus(id: impl Into<String>, max_vus: u64, duration: Duration, collector: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            executor_type: "constant_vus".to_string(),
            max_vus,
            duration_secs: duration.as_secs(),
            rate: None,
            collector_actor: collector.into(),
        }
    }

    pub fn constant_arrival_rate(
        id: impl Into<String>,
        max_vus: u64,
        duration: Duration,
        rate: f64,
        collector: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            executor_type: "constant_arrival_rate".to_string(),
            max_vus,
            duration_secs: duration.as_secs(),
            rate: Some(rate),
            collector_actor: collector.into(),
        }
    }
}

