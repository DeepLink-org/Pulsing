//! Pulsing Benchmark - Actor-based LLM Inference Benchmarking Tool
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
//!
//! This crate provides a high-performance benchmark tool for LLM inference endpoints
//! using the Actor model for better separation of concerns and real-time metrics.
//!
//! ## Architecture
//!
//! ```text
//!                              ┌─────────────────┐
//!                              │   Coordinator   │
//!                              │  (Orchestrator) │
//!                              └────────┬────────┘
//!                                       │
//!         ┌─────────────────────────────┼─────────────────────────────┐
//!         │                             │                             │
//!         ▼                             ▼                             ▼
//! ┌────────────────┐           ┌─────────────────┐           ┌────────────────┐
//! │   Scheduler    │           │   Worker(s)     │           │    Metrics     │
//! │ (Rate Control) │           │ (HTTP Clients)  │           │  Aggregator    │
//! └────────────────┘           └───────┬─────────┘           └────────┬───────┘
//!                                      │                              │
//!                                      │ RequestCompleted             │ DisplayUpdate
//!                                      └──────────────────────────────┤
//!                                                                     ▼
//!                                                            ┌────────────────┐
//!                                                            │    Console     │
//!                                                            │   Renderer     │
//!                                                            └────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use pulsing_bench::{run_benchmark, BenchmarkArgs};
//!
//! #[tokio::main]
//! async fn main() {
//!     let args = BenchmarkArgs {
//!         url: "http://localhost:8000".to_string(),
//!         model_name: "Qwen/Qwen3-0.6B".to_string(),
//!         max_vus: 10,
//!         duration_secs: 60,
//!         ..Default::default()
//!     };
//!
//!     let report = run_benchmark(args).await.unwrap();
//!     println!("Completed {} requests", report.phases.len());
//! }
//! ```

use log::info;
use pulsing_actor::prelude::*;

// Actor-based benchmark implementation
pub mod actors;

// Tokenizer service for accurate token counting
pub mod tokenizer;

// Re-export main types
pub use actors::{
    BenchmarkConfig, BenchmarkPhase, BenchmarkReport, CoordinatorActor, MetricsSnapshot,
    PhaseReport, SchedulerType,
};
pub use tokenizer::{TokenCounter, TokenizerService};

/// Benchmark arguments
#[derive(Debug, Clone)]
pub struct BenchmarkArgs {
    /// Target URL for the LLM endpoint
    pub url: String,
    /// API key for authentication
    pub api_key: String,
    /// Model name to use in requests
    pub model_name: String,
    /// Tokenizer name (HuggingFace model name, defaults to model_name)
    pub tokenizer_name: Option<String>,
    /// HuggingFace token for private models
    pub hf_token: Option<String>,
    /// Maximum number of virtual users (concurrent requests)
    pub max_vus: u64,
    /// Duration of each benchmark phase in seconds
    pub duration_secs: u64,
    /// Warmup duration in seconds
    pub warmup_secs: u64,
    /// Benchmark kind: "throughput", "sweep", "csweep", "rate"
    pub benchmark_kind: String,
    /// Number of rate steps for sweep benchmarks
    pub num_rates: u64,
    /// Specific rates for rate benchmarks
    pub rates: Option<Vec<f64>>,
    /// Number of worker actors
    pub num_workers: u32,
}

impl Default for BenchmarkArgs {
    fn default() -> Self {
        Self {
            url: "http://localhost:8000".to_string(),
            api_key: String::new(),
            model_name: "gpt2".to_string(),
            tokenizer_name: None,
            hf_token: None,
            max_vus: 128,
            duration_secs: 120,
            warmup_secs: 30,
            benchmark_kind: "throughput".to_string(),
            num_rates: 10,
            rates: None,
            num_workers: 4,
        }
    }
}

/// Parse duration string (e.g., "120s", "2m") into Duration
pub fn parse_duration(s: &str) -> anyhow::Result<std::time::Duration> {
    humantime::parse_duration(s).map_err(|e| anyhow::anyhow!("Invalid duration: {}", e))
}

/// Run benchmark with the given arguments
///
/// This is the main entry point for running benchmarks. It creates an Actor system,
/// spawns the necessary actors, and orchestrates the benchmark phases.
///
/// # Arguments
///
/// * `args` - Benchmark configuration arguments
///
/// # Returns
///
/// * `Ok(BenchmarkReport)` - The benchmark results
/// * `Err` - If the benchmark fails
///
/// # Example
///
/// ```rust,ignore
/// let args = BenchmarkArgs {
///     url: "http://localhost:8000".to_string(),
///     model_name: "Qwen/Qwen3-0.6B".to_string(),
///     max_vus: 10,
///     duration_secs: 60,
///     ..Default::default()
/// };
///
/// let report = run_benchmark(args).await?;
/// ```
pub async fn run_benchmark(args: BenchmarkArgs) -> anyhow::Result<BenchmarkReport> {
    use actors::*;

    let git_sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
    println!(
        "Pulsing Benchmark {} ({})",
        env!("CARGO_PKG_VERSION"),
        git_sha
    );

    // Determine tokenizer name (use model_name if not specified)
    let tokenizer_name = args
        .tokenizer_name
        .clone()
        .unwrap_or_else(|| args.model_name.clone());

    // Load tokenizer
    println!("Loading tokenizer: {}", tokenizer_name);
    let token_counter = match TokenCounter::real(&tokenizer_name, args.hf_token.clone()) {
        Ok(tc) => {
            println!("Tokenizer loaded successfully");
            tc
        }
        Err(e) => {
            println!(
                "Warning: Failed to load tokenizer ({}), using estimation",
                e
            );
            TokenCounter::estimate()
        }
    };

    // Create actor system
    let system = ActorSystem::new(SystemConfig::standalone()).await?;

    // Create and spawn coordinator
    let coordinator = CoordinatorActor::new()
        .with_system(system.clone())
        .with_num_workers(args.num_workers)
        .with_token_counter(token_counter);

    let coordinator_ref = system.spawn_named("coordinator", coordinator).await?;

    // Build benchmark config
    let config = BenchmarkConfig {
        url: args.url,
        api_key: args.api_key,
        model_name: args.model_name,
        tokenizer_name: Some(tokenizer_name),
        max_vus: args.max_vus,
        duration_secs: args.duration_secs,
        rate: None,
        warmup_secs: args.warmup_secs,
        benchmark_kind: args.benchmark_kind,
        num_rates: args.num_rates,
        rates: args.rates,
    };

    let run_id = uuid::Uuid::new_v4().to_string()[..8].to_string();

    // Start benchmark
    let start_msg = StartBenchmark {
        config,
        run_id: run_id.clone(),
    };

    // Handle ctrl-c
    let coordinator_ref_clone = coordinator_ref.clone();
    let run_id_clone = run_id.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for ctrl-c");
        info!("Received ctrl-c, stopping benchmark");
        let stop_msg = StopBenchmark {
            run_id: run_id_clone,
            reason: "User interrupted".to_string(),
        };
        let _ = coordinator_ref_clone.tell(stop_msg).await;
    });

    // Run benchmark and wait for completion
    let result: AckMessage = coordinator_ref.ask(start_msg).await?;

    if !result.success {
        return Err(anyhow::anyhow!("Benchmark failed: {}", result.message));
    }

    // Get final report
    let get_report = GetReport { run_id };
    let report: BenchmarkReport = coordinator_ref.ask(get_report).await?;

    // Shutdown
    system.shutdown().await?;

    Ok(report)
}

/// Run a simple throughput test
///
/// Convenience function for running a basic throughput benchmark.
///
/// # Arguments
///
/// * `url` - Target URL
/// * `model_name` - Model name (also used as tokenizer name)
/// * `max_vus` - Maximum concurrent requests
/// * `duration_secs` - Test duration in seconds
pub async fn run_throughput_test(
    url: &str,
    model_name: &str,
    max_vus: u64,
    duration_secs: u64,
) -> anyhow::Result<BenchmarkReport> {
    let args = BenchmarkArgs {
        url: url.to_string(),
        model_name: model_name.to_string(),
        max_vus,
        duration_secs,
        warmup_secs: 10,
        benchmark_kind: "throughput".to_string(),
        ..Default::default()
    };
    run_benchmark(args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_args_default() {
        let args = BenchmarkArgs::default();
        assert_eq!(args.url, "http://localhost:8000");
        assert_eq!(args.model_name, "gpt2");
        assert_eq!(args.max_vus, 128);
        assert_eq!(args.duration_secs, 120);
        assert_eq!(args.warmup_secs, 30);
        assert_eq!(args.benchmark_kind, "throughput");
        assert!(args.api_key.is_empty());
        assert!(args.tokenizer_name.is_none());
        assert!(args.hf_token.is_none());
        assert!(args.rates.is_none());
        assert_eq!(args.num_rates, 10);
        assert_eq!(args.num_workers, 4);
    }

    #[test]
    fn test_benchmark_args_custom() {
        let args = BenchmarkArgs {
            url: "http://example.com:8080".to_string(),
            api_key: "secret".to_string(),
            model_name: "llama".to_string(),
            tokenizer_name: Some("meta-llama/Llama-2".to_string()),
            hf_token: Some("hf_token".to_string()),
            max_vus: 64,
            duration_secs: 60,
            warmup_secs: 15,
            benchmark_kind: "sweep".to_string(),
            num_rates: 5,
            rates: Some(vec![1.0, 2.0, 3.0]),
            num_workers: 8,
        };

        assert_eq!(args.url, "http://example.com:8080");
        assert_eq!(args.api_key, "secret");
        assert_eq!(args.model_name, "llama");
        assert_eq!(args.tokenizer_name, Some("meta-llama/Llama-2".to_string()));
        assert_eq!(args.max_vus, 64);
        assert_eq!(args.rates, Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_parse_duration_seconds() {
        let duration = parse_duration("120s").unwrap();
        assert_eq!(duration.as_secs(), 120);
    }

    #[test]
    fn test_parse_duration_minutes() {
        let duration = parse_duration("2m").unwrap();
        assert_eq!(duration.as_secs(), 120);
    }

    #[test]
    fn test_parse_duration_hours() {
        let duration = parse_duration("1h").unwrap();
        assert_eq!(duration.as_secs(), 3600);
    }

    #[test]
    fn test_parse_duration_milliseconds() {
        let duration = parse_duration("500ms").unwrap();
        assert_eq!(duration.as_millis(), 500);
    }

    #[test]
    fn test_parse_duration_combined() {
        let duration = parse_duration("1h30m").unwrap();
        assert_eq!(duration.as_secs(), 5400);
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration("invalid").is_err());
        assert!(parse_duration("abc123").is_err());
    }
}
