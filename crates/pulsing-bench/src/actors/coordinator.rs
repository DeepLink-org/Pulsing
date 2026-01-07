//! CoordinatorActor - Coordinates benchmark execution
//!
//! Responsible for:
//! - Managing benchmark lifecycle
//! - Spawning ExecutorActors and CollectorActor
//! - Running benchmark phases (warmup, throughput, sweep, etc.)
//! - Generating final report

use super::executor::ExecutorActor;
use super::messages::*;
use crate::benchmark::{BenchmarkConfig, BenchmarkKind};
use crate::requests::{TextGenerationBackend, TextRequestGenerator};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::ExecutorType;
use async_trait::async_trait;
use pulsing_actor::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

const THROUGHPUT_BUDGET: f64 = 1.2;

/// CoordinatorActor coordinates the benchmark flow
pub struct CoordinatorActor {
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Backend factory (creates new backends for executors)
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    /// Request generator
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    /// Benchmark report
    report: BenchmarkReport,
    /// Start time
    start_time: Option<Instant>,
    /// Collector actor ref (reserved for future distributed usage)
    #[allow(dead_code)]
    collector_ref: Option<ActorRef>,
    /// Executor actor ref (reserved for future distributed usage)
    #[allow(dead_code)]
    executor_ref: Option<ActorRef>,
}

impl CoordinatorActor {
    pub fn new(
        config: BenchmarkConfig,
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    ) -> Self {
        Self {
            config,
            backend,
            requests,
            report: BenchmarkReport::new(),
            start_time: None,
            collector_ref: None,
            executor_ref: None,
        }
    }

    /// Run the complete benchmark (direct execution, without actor system)
    pub async fn run_benchmark_direct(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.run_benchmark_internal().await
    }

    /// Run the complete benchmark (internal implementation)
    async fn run_benchmark_internal(&mut self) -> anyhow::Result<BenchmarkReport> {
        info!("Starting benchmark");
        self.start_time = Some(Instant::now());
        self.report.start();

        // Run warmup
        info!("Running warmup phase");
        self.run_phase("warmup", ExecutorType::ConstantVUs, 1, self.config.warmup_duration, None)
            .await?;
        info!("Warmup complete");

        // Run main benchmark based on kind
        match self.config.benchmark_kind {
            BenchmarkKind::Throughput => {
                self.run_throughput().await?;
            }
            BenchmarkKind::Sweep => {
                self.run_sweep().await?;
            }
            BenchmarkKind::ConcurrencySweep => {
                self.run_concurrency_sweep().await?;
            }
            BenchmarkKind::Rate => {
                self.run_rates().await?;
            }
        }

        self.report.end();
        
        let duration = self.start_time.map(|t| t.elapsed()).unwrap_or_default();
        info!("Benchmark complete in {:?}", duration);
        
        Ok(self.report.clone())
    }

    /// Run a single benchmark phase
    async fn run_phase(
        &mut self,
        id: &str,
        executor_type: ExecutorType,
        max_vus: u64,
        duration: Duration,
        rate: Option<f64>,
    ) -> anyhow::Result<BenchmarkResults> {
        debug!("Running phase: {} ({:?}, {} VUs, {:?})", id, executor_type, max_vus, duration);

        // Create executor actor inline (simpler approach)
        let mut executor = ExecutorActor::new(
            self.backend.clone(),
            self.requests.clone(),
        );

        // Build start message
        let start_msg = match executor_type {
            ExecutorType::ConstantVUs => StartExecutor::constant_vus(
                id,
                max_vus,
                duration,
                "collector",
            ),
            ExecutorType::ConstantArrivalRate => {
                let r = rate.ok_or_else(|| anyhow::anyhow!("Rate required for constant arrival rate"))?;
                StartExecutor::constant_arrival_rate(
                    id,
                    max_vus,
                    duration,
                    r,
                    "collector",
                )
            }
        };

        // Run executor directly (without actor system for simplicity)
        let result = match executor_type {
            ExecutorType::ConstantVUs => executor.run_constant_vus(start_msg).await?,
            ExecutorType::ConstantArrivalRate => executor.run_constant_arrival_rate(start_msg).await?,
        };

        info!(
            "Phase {} complete: {} successful, {} failed",
            id, result.successful_requests, result.failed_requests
        );

        // Get results and convert to BenchmarkResults
        let results_data = executor.results.read().await.clone();
        let mut benchmark_results = BenchmarkResults::new(
            id.to_string(),
            executor_type,
            crate::executors::ExecutorConfig {
                max_vus,
                duration,
                rate,
            },
        );

        // Convert request results to responses
        // Note: We create a minimal response since we're tracking the important metrics separately
        for result in results_data {
            // Create request for the response
            let request = Arc::new(crate::requests::TextGenerationRequest {
                id: None,
                prompt: String::new(),
                num_prompt_tokens: result.input_tokens,
                num_decode_tokens: Some(result.output_tokens),
            });
            
            // Create response using the constructor
            let mut response = crate::requests::TextGenerationAggregatedResponse::new(request);
            response.start_time = Some(tokio::time::Instant::now());
            response.end_time = Some(tokio::time::Instant::now());
            response.failed = !result.success;
            response.ended = true;
            response.num_generated_tokens = result.output_tokens;
            response.times_to_tokens = if let Some(ttft) = result.time_to_first_token_ms {
                let mut times = vec![Duration::from_millis(ttft)];
                let mut current = ttft;
                for &itl in &result.inter_token_latencies_ms {
                    current += itl;
                    times.push(Duration::from_millis(current));
                }
                times
            } else {
                Vec::new()
            };
            
            benchmark_results.add_response(response);
        }

        self.report.add_benchmark_result(benchmark_results.clone());
        Ok(benchmark_results)
    }

    /// Run throughput benchmark
    async fn run_throughput(&mut self) -> anyhow::Result<()> {
        info!("Running throughput benchmark");
        self.run_phase(
            "throughput",
            ExecutorType::ConstantVUs,
            self.config.max_vus,
            self.config.duration,
            None,
        ).await?;
        Ok(())
    }

    /// Run sweep benchmark
    async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // First run throughput to find max rate
        self.run_throughput().await?;

        // Get max throughput from results
        let throughput_results = &self.report.get_results()[1]; // Index 1 = after warmup
        let max_throughput = throughput_results.successful_request_rate()?;
        
        info!("Max throughput detected: {:.2} req/s", max_throughput);

        // Run sweep at different rates
        let num_rates = self.config.num_rates;
        for i in 1..=num_rates {
            let rate = i as f64 * max_throughput * THROUGHPUT_BUDGET / num_rates as f64;
            self.run_rate(rate).await?;
        }

        Ok(())
    }

    /// Run concurrency sweep benchmark
    async fn run_concurrency_sweep(&mut self) -> anyhow::Result<()> {
        info!("Running concurrency sweep benchmark");

        let max_concurrency = self.config.max_vus;
        let mut best_concurrency = 1;
        let mut best_throughput = 0.0;

        // Generate concurrency levels to test
        let mut levels = Vec::new();
        levels.push(1);
        let mut level = 2;
        while level <= max_concurrency {
            levels.push(level);
            if level < 10 {
                level += 1;
            } else if level < 50 {
                level += 5;
            } else if level < 100 {
                level += 10;
            } else {
                level += 20;
            }
        }
        if !levels.contains(&max_concurrency) {
            levels.push(max_concurrency);
        }
        levels.sort();
        levels.dedup();

        info!("Testing concurrency levels: {:?}", levels);

        for concurrency in levels {
            let results = self.run_phase(
                &format!("concurrency#{}vus", concurrency),
                ExecutorType::ConstantVUs,
                concurrency,
                self.config.duration,
                None,
            ).await?;

            let throughput = results.successful_request_rate().unwrap_or(0.0);
            info!("Concurrency {}: {:.2} req/s", concurrency, throughput);

            if throughput > best_throughput {
                best_throughput = throughput;
                best_concurrency = concurrency;
            }

            // Early stop if throughput declining
            if concurrency > 10 && throughput < best_throughput * 0.9 {
                warn!("Throughput declining, stopping early");
                break;
            }
        }

        info!(
            "Optimal concurrency: {} ({:.2} req/s)",
            best_concurrency, best_throughput
        );

        Ok(())
    }

    /// Run rate benchmark
    async fn run_rate(&mut self, rate: f64) -> anyhow::Result<()> {
        debug!("Running benchmark at rate: {} req/s", rate);
        self.run_phase(
            &format!("rate@{:.1}reqs", rate),
            ExecutorType::ConstantArrivalRate,
            self.config.max_vus,
            self.config.duration,
            Some(rate),
        ).await?;
        Ok(())
    }

    /// Run rates benchmark
    async fn run_rates(&mut self) -> anyhow::Result<()> {
        let rates = self.config.rates.clone().ok_or_else(|| {
            anyhow::anyhow!("Rates must be specified for rate benchmark")
        })?;

        for rate in rates {
            self.run_rate(rate).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl Actor for CoordinatorActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();

        if msg_type.ends_with("StartBenchmark") {
            match self.run_benchmark_internal().await {
                Ok(_) => {
                    return Message::pack(&BenchmarkComplete {
                        success: true,
                        message: "Benchmark completed successfully".to_string(),
                    });
                }
                Err(e) => {
                    return Message::pack(&BenchmarkComplete {
                        success: false,
                        message: format!("Benchmark failed: {}", e),
                    });
                }
            }
        }

        if msg_type.ends_with("StopBenchmark") {
            // Signal stop to executors
            self.report.end();
            return Message::pack(&BenchmarkComplete {
                success: true,
                message: "Benchmark stopped".to_string(),
            });
        }

        if msg_type.ends_with("ProgressUpdate") {
            // Return current progress
            let progress = ProgressUpdate {
                phase: "unknown".to_string(),
                progress: 0.0,
                successful_requests: 0,
                failed_requests: 0,
                requests_throughput: 0.0,
                avg_ttft_ms: None,
                avg_tpot_ms: None,
                input_throughput: None,
                output_throughput: None,
            };
            return Message::pack(&progress);
        }

        Err(anyhow::anyhow!("Unknown message type: {}", msg_type))
    }
}

