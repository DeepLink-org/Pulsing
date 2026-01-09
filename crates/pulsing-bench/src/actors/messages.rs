//! Message definitions for the benchmark actor system

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration Messages
// ============================================================================

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Target URL
    pub url: String,
    /// API key for authentication
    pub api_key: String,
    /// Model name
    pub model_name: String,
    /// Tokenizer name (HuggingFace model name)
    pub tokenizer_name: Option<String>,
    /// Maximum virtual users (concurrent requests)
    pub max_vus: u64,
    /// Test duration in seconds
    pub duration_secs: u64,
    /// Target request rate (requests per second), None for max throughput
    pub rate: Option<f64>,
    /// Warmup duration in seconds
    pub warmup_secs: u64,
    /// Benchmark kind: "throughput", "sweep", "rate", "csweep"
    pub benchmark_kind: String,
    /// Number of rate steps for sweep
    pub num_rates: u64,
    /// Specific rates for rate benchmark
    pub rates: Option<Vec<f64>>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8000".to_string(),
            api_key: String::new(),
            model_name: "gpt2".to_string(),
            tokenizer_name: None,
            max_vus: 128,
            duration_secs: 120,
            rate: None,
            warmup_secs: 30,
            benchmark_kind: "throughput".to_string(),
            num_rates: 10,
            rates: None,
        }
    }
}

/// Request template for generating requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTemplate {
    /// Request ID
    pub id: String,
    /// Prompt text
    pub prompt: String,
    /// Number of prompt tokens
    pub num_prompt_tokens: u64,
    /// Target decode tokens (optional)
    pub num_decode_tokens: Option<u64>,
}

// ============================================================================
// Coordinator Messages
// ============================================================================

/// Start a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartBenchmark {
    pub config: BenchmarkConfig,
    pub run_id: String,
}

/// Stop the current benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopBenchmark {
    pub run_id: String,
    pub reason: String,
}

/// Benchmark phase transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    pub run_id: String,
    pub phase: BenchmarkPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BenchmarkPhase {
    Initializing,
    Warmup,
    Running,
    Cooldown,
    Completed,
    Failed(String),
}

/// Coordinator status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorStatus {
    pub run_id: Option<String>,
    pub phase: Option<BenchmarkPhase>,
    pub workers: u32,
    pub active_requests: u64,
}

// ============================================================================
// Scheduler Messages
// ============================================================================

/// Configure the scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureScheduler {
    pub scheduler_type: SchedulerType,
    pub max_vus: u64,
    pub duration_secs: u64,
    pub rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum SchedulerType {
    /// Constant number of virtual users
    #[default]
    ConstantVUs,
    /// Constant arrival rate
    ConstantArrivalRate,
}

/// Start scheduling requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartScheduling {
    pub phase_id: String,
}

/// Pause scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseScheduling;

/// Resume scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeScheduling;

/// Scheduler progress update (sent to Coordinator)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerProgress {
    pub phase_id: String,
    pub progress_pct: f64,
    pub sent_requests: u64,
    pub active_vus: u64,
    pub elapsed_secs: f64,
}

// ============================================================================
// Worker Messages
// ============================================================================

/// Send a request (from Scheduler to Worker)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendRequest {
    pub request_id: String,
    pub template: RequestTemplate,
    pub target_url: String,
    pub api_key: String,
    pub model_name: String,
}

/// Request completed (from Worker to Collector)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestCompleted {
    pub request_id: String,
    pub success: bool,
    pub error: Option<String>,
    /// Time to first token in milliseconds
    pub ttft_ms: Option<f64>,
    /// Total latency in milliseconds
    pub latency_ms: f64,
    /// Number of generated tokens
    pub generated_tokens: u64,
    /// Number of prompt tokens
    pub prompt_tokens: u64,
    /// Time per output token in milliseconds
    pub tpot_ms: Option<f64>,
    /// Token timestamps for detailed analysis
    pub token_times_ms: Vec<f64>,
}

/// Worker ready notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerReady {
    pub worker_id: String,
}

/// Worker status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub worker_id: String,
    pub active_requests: u32,
    pub completed_requests: u64,
    pub failed_requests: u64,
}

// ============================================================================
// Collector Messages
// ============================================================================

/// Register a new run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRun {
    pub run_id: String,
    pub config: BenchmarkConfig,
}

/// Start a new phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartPhase {
    pub run_id: String,
    pub phase_id: String,
    pub phase_name: String,
    pub scheduler_type: SchedulerType,
    pub target_rate: Option<f64>,
}

/// End current phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndPhase {
    pub run_id: String,
    pub phase_id: String,
}

/// Get current metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetMetrics {
    pub run_id: String,
}

/// Get final report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetReport {
    pub run_id: String,
}

/// Real-time metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsSnapshot {
    pub phase_id: String,
    pub phase_name: String,
    pub progress_pct: f64,
    pub elapsed_secs: f64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub request_rate: f64,
    pub avg_ttft_ms: Option<f64>,
    pub avg_tpot_ms: Option<f64>,
    pub ttft_std_ms: Option<f64>,
    pub tpot_std_ms: Option<f64>,
    pub avg_latency_ms: Option<f64>,
    pub p50_latency_ms: Option<f64>,
    pub p95_latency_ms: Option<f64>,
    pub p99_latency_ms: Option<f64>,
    pub input_throughput: Option<f64>,
    pub output_throughput: Option<f64>,
    pub total_throughput: Option<f64>,
}

/// Final benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub run_id: String,
    pub config: BenchmarkConfig,
    pub phases: Vec<PhaseReport>,
    pub start_time: String,
    pub end_time: String,
    pub total_duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReport {
    pub phase_id: String,
    pub phase_name: String,
    pub duration_secs: f64,
    pub metrics: MetricsSnapshot,
}

// ============================================================================
// Common Response Messages
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AckMessage {
    pub success: bool,
    pub message: String,
}

impl AckMessage {
    pub fn ok() -> Self {
        Self {
            success: true,
            message: "OK".to_string(),
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            message: msg.into(),
        }
    }
}
