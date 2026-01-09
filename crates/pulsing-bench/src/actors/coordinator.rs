//! Coordinator Actor - orchestrates the benchmark lifecycle
//!
//! This actor is responsible for:
//! - Managing actor lifecycle (spawn/stop workers, scheduler, metrics, renderer)
//! - Orchestrating benchmark phases
//! - Handling start/stop commands
//!
//! Refactored to use unified phase execution logic.

use super::messages::*;
use super::scheduler::{RequestGenerator, SimpleRequestGenerator, TokenizedRequestGenerator};
use super::{ConsoleRendererActor, MetricsAggregatorActor, SchedulerActor, WorkerActor};
use crate::tokenizer::TokenCounter;
use async_trait::async_trait;
use pulsing_actor::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

/// Coordinator state
#[derive(Debug, Clone, PartialEq)]
enum CoordinatorState {
    Idle,
    Running(String), // run_id
    Stopping,
}

/// Phase definition - describes how to run a benchmark phase
#[derive(Debug, Clone)]
struct PhaseDefinition {
    phase_id: String,
    phase_name: String,
    scheduler_type: SchedulerType,
    max_vus: u64,
    duration_secs: u64,
    rate: Option<f64>,
}

/// Coordinator Actor - manages the entire benchmark lifecycle
pub struct CoordinatorActor {
    /// Actor system reference
    system: Option<Arc<ActorSystem>>,
    /// Current state
    state: CoordinatorState,
    /// Current phase
    phase: Option<BenchmarkPhase>,
    /// Current configuration
    config: Option<BenchmarkConfig>,
    /// Worker references
    workers: Vec<ActorRef>,
    /// Scheduler reference
    scheduler_ref: Option<ActorRef>,
    /// Metrics aggregator reference
    metrics_ref: Option<ActorRef>,
    /// Console renderer reference
    renderer_ref: Option<ActorRef>,
    /// Number of workers to spawn
    num_workers: u32,
    /// Request generator
    request_gen: Arc<dyn RequestGenerator>,
    /// Token counter for accurate token counting
    token_counter: Option<TokenCounter>,
    /// Enable console display
    display_enabled: bool,
    /// Cached final report (saved before cleanup)
    final_report: Option<BenchmarkReport>,
}

impl CoordinatorActor {
    pub fn new() -> Self {
        Self {
            system: None,
            state: CoordinatorState::Idle,
            phase: None,
            config: None,
            workers: Vec::new(),
            scheduler_ref: None,
            metrics_ref: None,
            renderer_ref: None,
            num_workers: 4,
            request_gen: Arc::new(SimpleRequestGenerator::default_prompts()),
            token_counter: None,
            display_enabled: true,
            final_report: None,
        }
    }

    pub fn with_system(mut self, system: Arc<ActorSystem>) -> Self {
        self.system = Some(system);
        self
    }

    pub fn with_num_workers(mut self, num: u32) -> Self {
        self.num_workers = num;
        self
    }

    pub fn with_request_generator(mut self, gen: Arc<dyn RequestGenerator>) -> Self {
        self.request_gen = gen;
        self
    }

    pub fn with_token_counter(mut self, tc: TokenCounter) -> Self {
        self.token_counter = Some(tc);
        self
    }

    pub fn with_display(mut self, enabled: bool) -> Self {
        self.display_enabled = enabled;
        self
    }

    /// Start a new benchmark run
    async fn start_benchmark(&mut self, start: StartBenchmark) -> anyhow::Result<AckMessage> {
        if self.state != CoordinatorState::Idle {
            return Ok(AckMessage::error("Benchmark already running"));
        }

        let system = self
            .system
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ActorSystem not configured"))?;

        info!("Starting benchmark run: {}", start.run_id);
        self.config = Some(start.config.clone());
        self.state = CoordinatorState::Running(start.run_id.clone());
        self.phase = Some(BenchmarkPhase::Initializing);

        // Spawn Console Renderer first (if enabled)
        if self.display_enabled {
            let renderer = ConsoleRendererActor::new();
            let renderer_ref = system
                .spawn(format!("renderer-{}", start.run_id), renderer)
                .await?;
            self.renderer_ref = Some(renderer_ref);
        }

        // Spawn Metrics Aggregator
        let mut metrics = MetricsAggregatorActor::new();
        if let Some(ref renderer) = self.renderer_ref {
            metrics = metrics.with_renderer(renderer.clone());
        }
        let metrics_ref = system
            .spawn(format!("metrics-{}", start.run_id), metrics)
            .await?;
        self.metrics_ref = Some(metrics_ref.clone());

        // Register run with metrics
        let register = RegisterRun {
            run_id: start.run_id.clone(),
            config: start.config.clone(),
        };
        metrics_ref.tell(register).await?;

        // Spawn Workers
        self.workers.clear();
        for i in 0..self.num_workers {
            let worker =
                WorkerActor::new(format!("worker-{}", i)).with_metrics(metrics_ref.clone());
            let worker_ref = system
                .spawn(format!("worker-{}-{}", start.run_id, i), worker)
                .await?;
            self.workers.push(worker_ref);
        }

        // Spawn Scheduler with appropriate request generator
        let request_gen: Arc<dyn RequestGenerator> = if let Some(ref tc) = self.token_counter {
            Arc::new(TokenizedRequestGenerator::new(tc.clone()))
        } else {
            self.request_gen.clone()
        };

        let scheduler = SchedulerActor::new()
            .with_workers(self.workers.clone())
            .with_metrics(metrics_ref.clone())
            .with_request_generator(request_gen)
            .with_target(
                start.config.url.clone(),
                start.config.api_key.clone(),
                start.config.model_name.clone(),
            );
        let scheduler_ref = system
            .spawn(format!("scheduler-{}", start.run_id), scheduler)
            .await?;
        self.scheduler_ref = Some(scheduler_ref);

        info!(
            "Initialized {} workers, scheduler, metrics for run {}",
            self.num_workers, start.run_id
        );

        // Run benchmark phases
        let result = self.run_benchmark_phases(&start).await;

        // Get and cache final report before cleanup
        if let Some(ref metrics) = self.metrics_ref {
            let get_report = GetReport {
                run_id: start.run_id.clone(),
            };
            if let Ok(report) = metrics.ask::<_, BenchmarkReport>(get_report).await {
                self.final_report = Some(report);
            }
        }

        // Cleanup
        self.cleanup(&start.run_id).await;

        match result {
            Ok(()) => {
                self.phase = Some(BenchmarkPhase::Completed);
                Ok(AckMessage::ok())
            }
            Err(e) => {
                self.phase = Some(BenchmarkPhase::Failed(e.to_string()));
                Ok(AckMessage::error(e.to_string()))
            }
        }
    }

    async fn run_benchmark_phases(&mut self, start: &StartBenchmark) -> anyhow::Result<()> {
        let config = &start.config;

        // Phase 1: Warmup
        self.phase = Some(BenchmarkPhase::Warmup);
        info!("Starting warmup phase");

        let warmup = PhaseDefinition {
            phase_id: "warmup".to_string(),
            phase_name: "Warmup".to_string(),
            scheduler_type: SchedulerType::ConstantVUs,
            max_vus: 1,
            duration_secs: config.warmup_secs,
            rate: None,
        };
        self.run_phase(&start.run_id, &warmup).await?;

        // Phase 2: Main benchmark
        self.phase = Some(BenchmarkPhase::Running);
        info!("Starting main benchmark phase");

        let phases = self.generate_phases(config);
        for phase_def in phases {
            self.run_phase(&start.run_id, &phase_def).await?;
        }

        // Phase 3: Cooldown
        self.phase = Some(BenchmarkPhase::Cooldown);
        info!("Cooldown phase");
        tokio::time::sleep(Duration::from_secs(2)).await;

        Ok(())
    }

    /// Generate phase definitions based on benchmark kind
    fn generate_phases(&self, config: &BenchmarkConfig) -> Vec<PhaseDefinition> {
        match config.benchmark_kind.as_str() {
            "throughput" => vec![PhaseDefinition {
                phase_id: "throughput".to_string(),
                phase_name: "Max Throughput".to_string(),
                scheduler_type: SchedulerType::ConstantVUs,
                max_vus: config.max_vus,
                duration_secs: config.duration_secs,
                rate: None,
            }],

            "sweep" => {
                // Sweep needs to first discover max rate, then run rate tests
                // For simplicity, use num_rates evenly distributed
                let mut phases = vec![
                    // First: throughput discovery
                    PhaseDefinition {
                        phase_id: "throughput-discovery".to_string(),
                        phase_name: "Throughput Discovery".to_string(),
                        scheduler_type: SchedulerType::ConstantVUs,
                        max_vus: config.max_vus,
                        duration_secs: config.duration_secs,
                        rate: None,
                    },
                ];

                // Rate sweep phases (will be dynamically adjusted)
                for i in 1..=config.num_rates {
                    let rate_ratio = i as f64 / config.num_rates as f64;
                    phases.push(PhaseDefinition {
                        phase_id: format!("rate-{}", i),
                        phase_name: format!("Rate Sweep {}/{}", i, config.num_rates),
                        scheduler_type: SchedulerType::ConstantArrivalRate,
                        max_vus: config.max_vus,
                        duration_secs: config.duration_secs,
                        rate: Some(rate_ratio * 100.0), // Placeholder, will be adjusted
                    });
                }
                phases
            }

            "rate" => config
                .rates
                .clone()
                .unwrap_or_default()
                .into_iter()
                .map(|rate| PhaseDefinition {
                    phase_id: format!("rate@{:.1}reqs", rate),
                    phase_name: format!("Rate {:.1} req/s", rate),
                    scheduler_type: SchedulerType::ConstantArrivalRate,
                    max_vus: config.max_vus,
                    duration_secs: config.duration_secs,
                    rate: Some(rate),
                })
                .collect(),

            "csweep" => {
                let levels = self.generate_concurrency_levels(config.max_vus);
                levels
                    .into_iter()
                    .map(|vus| PhaseDefinition {
                        phase_id: format!("concurrency#{}vus", vus),
                        phase_name: format!("Concurrency {} VUs", vus),
                        scheduler_type: SchedulerType::ConstantVUs,
                        max_vus: vus,
                        duration_secs: config.duration_secs,
                        rate: None,
                    })
                    .collect()
            }

            _ => vec![PhaseDefinition {
                phase_id: "unknown".to_string(),
                phase_name: "Unknown".to_string(),
                scheduler_type: SchedulerType::ConstantVUs,
                max_vus: config.max_vus,
                duration_secs: config.duration_secs,
                rate: None,
            }],
        }
    }

    /// Generate concurrency levels for sweep
    fn generate_concurrency_levels(&self, max_vus: u64) -> Vec<u64> {
        let mut levels = vec![1u64];
        let mut level = 2;
        while level <= max_vus {
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
        if !levels.contains(&max_vus) {
            levels.push(max_vus);
        }
        levels
    }

    /// Unified phase execution - reduces code duplication
    async fn run_phase(&self, run_id: &str, phase: &PhaseDefinition) -> anyhow::Result<()> {
        let scheduler_ref = self
            .scheduler_ref
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Scheduler not initialized"))?;
        let metrics_ref = self
            .metrics_ref
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Metrics not initialized"))?;

        info!("Running phase: {}", phase.phase_name);

        // Step 1: Configure scheduler
        let config_msg = ConfigureScheduler {
            scheduler_type: phase.scheduler_type.clone(),
            max_vus: phase.max_vus,
            duration_secs: phase.duration_secs,
            rate: phase.rate,
        };
        scheduler_ref.tell(config_msg).await?;

        // Step 2: Notify metrics of phase start
        let start_phase = StartPhase {
            run_id: run_id.to_string(),
            phase_id: phase.phase_id.clone(),
            phase_name: phase.phase_name.clone(),
            scheduler_type: phase.scheduler_type.clone(),
            target_rate: phase.rate,
        };
        metrics_ref.tell(start_phase).await?;

        // Step 3: Start scheduling and wait for completion
        let start_msg = StartScheduling {
            phase_id: phase.phase_id.clone(),
        };
        scheduler_ref.ask::<_, AckMessage>(start_msg).await?;

        // Step 4: Notify metrics of phase end
        let end_phase = EndPhase {
            run_id: run_id.to_string(),
            phase_id: phase.phase_id.clone(),
        };
        metrics_ref.tell(end_phase).await?;

        info!("Phase {} completed", phase.phase_name);
        Ok(())
    }

    async fn cleanup(&mut self, run_id: &str) {
        let system = match &self.system {
            Some(s) => s,
            None => return,
        };

        info!("Cleaning up actors for run {}", run_id);

        // Stop scheduler
        if self.scheduler_ref.is_some() {
            let _ = system.stop(format!("scheduler-{}", run_id)).await;
        }

        // Stop workers
        for i in 0..self.num_workers {
            let _ = system.stop(format!("worker-{}-{}", run_id, i)).await;
        }

        // Stop metrics (this will trigger final report to renderer)
        if self.metrics_ref.is_some() {
            let _ = system.stop(format!("metrics-{}", run_id)).await;
        }

        // Stop renderer
        if self.renderer_ref.is_some() {
            // Give renderer time to process final report
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = system.stop(format!("renderer-{}", run_id)).await;
        }

        self.scheduler_ref = None;
        self.metrics_ref = None;
        self.renderer_ref = None;
        self.workers.clear();
        self.state = CoordinatorState::Idle;
    }

    async fn stop_benchmark(&mut self, stop: StopBenchmark) -> AckMessage {
        if let CoordinatorState::Running(ref run_id) = self.state {
            if *run_id != stop.run_id {
                return AckMessage::error("Run ID mismatch");
            }

            info!("Stopping benchmark: {}", stop.reason);
            self.state = CoordinatorState::Stopping;

            // Signal scheduler to stop
            if let Some(ref scheduler) = self.scheduler_ref {
                let _ = scheduler.tell(PauseScheduling).await;
            }

            self.cleanup(&stop.run_id).await;
            AckMessage::ok()
        } else {
            AckMessage::error("No benchmark running")
        }
    }

    fn get_status(&self) -> CoordinatorStatus {
        let (run_id, phase) = match &self.state {
            CoordinatorState::Running(id) => (Some(id.clone()), self.phase.clone()),
            _ => (None, None),
        };

        CoordinatorStatus {
            run_id,
            phase,
            workers: self.workers.len() as u32,
            active_requests: 0, // Could aggregate from metrics
        }
    }
}

impl Default for CoordinatorActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for CoordinatorActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        info!("Coordinator started with actor_id {:?}", ctx.id());
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        info!("Coordinator stopped");
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();

        if msg_type.ends_with("StartBenchmark") {
            let start: StartBenchmark = msg.unpack()?;
            let result = self.start_benchmark(start).await?;
            return Message::pack(&result);
        }

        if msg_type.ends_with("StopBenchmark") {
            let stop: StopBenchmark = msg.unpack()?;
            let result = self.stop_benchmark(stop).await;
            return Message::pack(&result);
        }

        if msg_type.ends_with("CoordinatorStatus") || msg_type.ends_with("GetStatus") {
            return Message::pack(&self.get_status());
        }

        if msg_type.ends_with("GetReport") {
            // Try to get from metrics actor first
            if let Some(ref metrics) = self.metrics_ref {
                let report: GetReport = msg.unpack()?;
                return metrics.send(Message::pack(&report)?).await;
            }
            // Fall back to cached report
            if let Some(ref report) = self.final_report {
                return Message::pack(report);
            }
            return Err(anyhow::anyhow!("No report available"));
        }

        Err(anyhow::anyhow!("Unknown message type: {}", msg_type))
    }
}
