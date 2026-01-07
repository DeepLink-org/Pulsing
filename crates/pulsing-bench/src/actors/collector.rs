//! CollectorActor - Aggregates benchmark results
//!
//! Responsible for:
//! - Collecting results from ExecutorActors
//! - Computing statistics (TTFT, TPOT, throughput, etc.)
//! - Providing progress updates

use super::messages::*;
use crate::executors::ExecutorConfig;
use crate::scheduler::ExecutorType;
use async_trait::async_trait;
use pulsing_actor::prelude::*;
use std::time::{Duration, Instant};
use tracing::info;

/// CollectorActor aggregates benchmark results
pub struct CollectorActor {
    /// Current phase ID
    phase_id: String,
    /// Executor type for current phase
    executor_type: ExecutorType,
    /// Configuration for current phase
    config: ExecutorConfig,
    /// Start time of current phase
    start_time: Option<Instant>,
    /// Collected results
    results: Vec<RequestResult>,
    /// Cached statistics
    cached_stats: Option<BenchmarkStats>,
    /// Last stats update time
    last_stats_update: Option<Instant>,
}

/// Cached statistics for quick access
#[derive(Debug, Clone)]
struct BenchmarkStats {
    successful_requests: u64,
    failed_requests: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    ttft_sum_ms: u64,
    ttft_count: u64,
    tpot_sum_ms: u64,
    tpot_count: u64,
    duration_secs: f64,
}

impl CollectorActor {
    pub fn new() -> Self {
        Self {
            phase_id: String::new(),
            executor_type: ExecutorType::ConstantVUs,
            config: ExecutorConfig {
                max_vus: 1,
                duration: Duration::from_secs(1),
                rate: None,
            },
            start_time: None,
            results: Vec::new(),
            cached_stats: None,
            last_stats_update: None,
        }
    }

    /// Initialize for a new benchmark phase
    fn init_phase(&mut self, config: InitCollector) {
        self.phase_id = config.phase_id;
        self.executor_type = match config.executor_type.as_str() {
            "constant_arrival_rate" => ExecutorType::ConstantArrivalRate,
            _ => ExecutorType::ConstantVUs,
        };
        self.config = ExecutorConfig {
            max_vus: config.max_vus,
            duration: Duration::from_secs(config.duration_secs),
            rate: config.rate,
        };
        self.start_time = Some(Instant::now());
        self.results.clear();
        self.cached_stats = None;
        self.last_stats_update = None;
        
        info!("Collector initialized for phase: {}", self.phase_id);
    }

    /// Add a result
    fn add_result(&mut self, result: RequestResult) {
        self.results.push(result);
        // Invalidate cache
        self.cached_stats = None;
    }

    /// Compute current statistics
    fn compute_stats(&mut self) -> BenchmarkStats {
        // Check cache
        if let Some(ref stats) = self.cached_stats {
            if let Some(last_update) = self.last_stats_update {
                if last_update.elapsed() < Duration::from_millis(100) {
                    return stats.clone();
                }
            }
        }

        let mut stats = BenchmarkStats {
            successful_requests: 0,
            failed_requests: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            ttft_sum_ms: 0,
            ttft_count: 0,
            tpot_sum_ms: 0,
            tpot_count: 0,
            duration_secs: self.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(1.0),
        };

        for result in &self.results {
            if result.success {
                stats.successful_requests += 1;
                stats.total_input_tokens += result.input_tokens;
                stats.total_output_tokens += result.output_tokens;
                
                if let Some(ttft) = result.time_to_first_token_ms {
                    stats.ttft_sum_ms += ttft;
                    stats.ttft_count += 1;
                }
                
                for &itl in &result.inter_token_latencies_ms {
                    stats.tpot_sum_ms += itl;
                    stats.tpot_count += 1;
                }
            } else {
                stats.failed_requests += 1;
            }
        }

        self.cached_stats = Some(stats.clone());
        self.last_stats_update = Some(Instant::now());
        stats
    }

    /// Get progress update
    fn get_progress(&mut self) -> ProgressUpdate {
        let stats = self.compute_stats();
        let duration_secs = stats.duration_secs.max(0.001);
        let expected_duration = self.config.duration.as_secs_f64();
        
        ProgressUpdate {
            phase: self.phase_id.clone(),
            progress: (stats.duration_secs / expected_duration * 100.0).min(100.0),
            successful_requests: stats.successful_requests,
            failed_requests: stats.failed_requests,
            requests_throughput: stats.successful_requests as f64 / duration_secs,
            avg_ttft_ms: if stats.ttft_count > 0 {
                Some(stats.ttft_sum_ms as f64 / stats.ttft_count as f64)
            } else {
                None
            },
            avg_tpot_ms: if stats.tpot_count > 0 {
                Some(stats.tpot_sum_ms as f64 / stats.tpot_count as f64)
            } else {
                None
            },
            input_throughput: Some(stats.total_input_tokens as f64 / duration_secs),
            output_throughput: Some(stats.total_output_tokens as f64 / duration_secs),
        }
    }

}

impl Default for CollectorActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for CollectorActor {
    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();
        
        if msg_type.ends_with("InitCollector") {
            let config: InitCollector = msg.unpack()?;
            self.init_phase(config);
            return Message::pack(&"Initialized");
        }
        
        if msg_type.ends_with("RequestResult") {
            let result: RequestResult = msg.unpack()?;
            self.add_result(result);
            return Message::pack(&"Added");
        }
        
        if msg_type.ends_with("GetResults") || msg_type.ends_with("ProgressUpdate") {
            let progress = self.get_progress();
            return Message::pack(&progress);
        }
        
        if msg_type.ends_with("FinalizePhase") {
            let _: FinalizePhase = msg.unpack()?;
            let stats = self.compute_stats();
            return Message::pack(&PhaseFinalized {
                phase_id: self.phase_id.clone(),
                successful_requests: stats.successful_requests,
                failed_requests: stats.failed_requests,
                duration_secs: stats.duration_secs,
                throughput: stats.successful_requests as f64 / stats.duration_secs.max(0.001),
            });
        }
        
        if msg_type.ends_with("ResetCollector") {
            let reset: ResetCollector = msg.unpack()?;
            self.phase_id = reset.phase_id;
            self.results.clear();
            self.cached_stats = None;
            self.start_time = Some(Instant::now());
            return Message::pack(&"Reset");
        }
        
        Err(anyhow::anyhow!("Unknown message type: {}", msg_type))
    }
}

