//! Metrics Aggregator Actor - collects and computes statistics
//!
//! This actor is responsible for:
//! - Receiving RequestCompleted events from workers
//! - Maintaining per-phase statistics
//! - Computing metrics (percentiles, averages, throughput)
//! - Providing metrics snapshots to other actors

use super::messages::*;
use async_trait::async_trait;
use pulsing_actor::prelude::*;
use std::time::{Duration, Instant};
use tracing::info;

/// Phase statistics - internal data structure for metrics computation
#[derive(Debug, Clone, Default)]
pub struct PhaseStats {
    pub phase_id: String,
    pub phase_name: String,
    pub scheduler_type: SchedulerType,
    pub target_rate: Option<f64>,
    pub start_time: Option<Instant>,
    pub end_time: Option<Instant>,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    ttft_values: Vec<f64>,
    tpot_values: Vec<f64>,
    latency_values: Vec<f64>,
}

impl PhaseStats {
    pub fn new(
        phase_id: String,
        phase_name: String,
        scheduler_type: SchedulerType,
        target_rate: Option<f64>,
    ) -> Self {
        Self {
            phase_id,
            phase_name,
            scheduler_type,
            target_rate,
            start_time: Some(Instant::now()),
            ..Default::default()
        }
    }

    pub fn add_result(&mut self, result: &RequestCompleted) {
        if result.success {
            self.successful_requests += 1;
            self.total_prompt_tokens += result.prompt_tokens;
            self.total_generated_tokens += result.generated_tokens;
            self.latency_values.push(result.latency_ms);

            if let Some(ttft) = result.ttft_ms {
                self.ttft_values.push(ttft);
            }
            if let Some(tpot) = result.tpot_ms {
                self.tpot_values.push(tpot);
            }
        } else {
            self.failed_requests += 1;
        }
    }

    pub fn duration(&self) -> Duration {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end.duration_since(start),
            (Some(start), None) => start.elapsed(),
            _ => Duration::ZERO,
        }
    }

    pub fn progress(&self, expected_duration: Duration) -> f64 {
        let elapsed = self.duration().as_secs_f64();
        let expected = expected_duration.as_secs_f64();
        if expected > 0.0 {
            (elapsed / expected * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    pub fn request_rate(&self) -> f64 {
        let duration = self.duration().as_secs_f64();
        if duration > 0.0 {
            self.successful_requests as f64 / duration
        } else {
            0.0
        }
    }

    pub fn error_rate(&self) -> f64 {
        let total = self.successful_requests + self.failed_requests;
        if total > 0 {
            self.failed_requests as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    }

    pub fn avg_ttft_ms(&self) -> Option<f64> {
        if self.ttft_values.is_empty() {
            None
        } else {
            Some(self.ttft_values.iter().sum::<f64>() / self.ttft_values.len() as f64)
        }
    }

    pub fn avg_tpot_ms(&self) -> Option<f64> {
        if self.tpot_values.is_empty() {
            None
        } else {
            Some(self.tpot_values.iter().sum::<f64>() / self.tpot_values.len() as f64)
        }
    }

    fn std_dev(values: &[f64]) -> Option<f64> {
        if values.len() < 2 {
            return None;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        Some(variance.sqrt())
    }

    pub fn ttft_std_ms(&self) -> Option<f64> {
        Self::std_dev(&self.ttft_values)
    }

    pub fn tpot_std_ms(&self) -> Option<f64> {
        Self::std_dev(&self.tpot_values)
    }

    fn percentile(values: &[f64], p: f64) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p * (sorted.len() - 1) as f64) / 100.0).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    pub fn avg_latency_ms(&self) -> Option<f64> {
        if self.latency_values.is_empty() {
            None
        } else {
            Some(self.latency_values.iter().sum::<f64>() / self.latency_values.len() as f64)
        }
    }

    pub fn p50_latency_ms(&self) -> Option<f64> {
        Self::percentile(&self.latency_values, 50.0)
    }

    pub fn p95_latency_ms(&self) -> Option<f64> {
        Self::percentile(&self.latency_values, 95.0)
    }

    pub fn p99_latency_ms(&self) -> Option<f64> {
        Self::percentile(&self.latency_values, 99.0)
    }

    pub fn input_throughput(&self) -> Option<f64> {
        let duration = self.duration().as_secs_f64();
        if duration > 0.0 && self.total_prompt_tokens > 0 {
            Some(self.total_prompt_tokens as f64 / duration)
        } else {
            None
        }
    }

    pub fn output_throughput(&self) -> Option<f64> {
        let duration = self.duration().as_secs_f64();
        if duration > 0.0 && self.total_generated_tokens > 0 {
            Some(self.total_generated_tokens as f64 / duration)
        } else {
            None
        }
    }

    pub fn total_throughput(&self) -> Option<f64> {
        let input = self.input_throughput().unwrap_or(0.0);
        let output = self.output_throughput().unwrap_or(0.0);
        if input > 0.0 || output > 0.0 {
            Some(input + output)
        } else {
            None
        }
    }

    /// Convert to a serializable metrics snapshot
    pub fn to_metrics(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            phase_id: self.phase_id.clone(),
            phase_name: self.phase_name.clone(),
            progress_pct: 100.0,
            elapsed_secs: self.duration().as_secs_f64(),
            successful_requests: self.successful_requests,
            failed_requests: self.failed_requests,
            request_rate: self.request_rate(),
            avg_ttft_ms: self.avg_ttft_ms(),
            avg_tpot_ms: self.avg_tpot_ms(),
            ttft_std_ms: self.ttft_std_ms(),
            tpot_std_ms: self.tpot_std_ms(),
            avg_latency_ms: self.avg_latency_ms(),
            p50_latency_ms: self.p50_latency_ms(),
            p95_latency_ms: self.p95_latency_ms(),
            p99_latency_ms: self.p99_latency_ms(),
            input_throughput: self.input_throughput(),
            output_throughput: self.output_throughput(),
            total_throughput: self.total_throughput(),
        }
    }

    /// Create a display-friendly snapshot with progress
    pub fn to_display_snapshot(&self, expected_duration: Duration) -> PhaseDisplayData {
        let is_completed = self.end_time.is_some();
        // If phase is completed, progress is always 100%
        let progress_pct = if is_completed {
            100.0
        } else {
            self.progress(expected_duration)
        };

        PhaseDisplayData {
            phase_id: self.phase_id.clone(),
            phase_name: self.phase_name.clone(),
            is_completed,
            progress_pct,
            error_rate: self.error_rate(),
            request_rate: self.request_rate(),
            avg_ttft_ms: self.avg_ttft_ms(),
            ttft_std_ms: self.ttft_std_ms(),
            avg_tpot_ms: self.avg_tpot_ms(),
            tpot_std_ms: self.tpot_std_ms(),
            input_throughput: self.input_throughput(),
            output_throughput: self.output_throughput(),
            total_throughput: self.total_throughput(),
        }
    }
}

/// Data sent to console renderer for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDisplayData {
    pub phase_id: String,
    pub phase_name: String,
    pub is_completed: bool,
    pub progress_pct: f64,
    pub error_rate: f64,
    pub request_rate: f64,
    pub avg_ttft_ms: Option<f64>,
    pub ttft_std_ms: Option<f64>,
    pub avg_tpot_ms: Option<f64>,
    pub tpot_std_ms: Option<f64>,
    pub input_throughput: Option<f64>,
    pub output_throughput: Option<f64>,
    pub total_throughput: Option<f64>,
}

/// Display update message sent to ConsoleRenderer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayUpdate {
    pub config: Option<BenchmarkConfig>,
    pub completed_phases: Vec<PhaseDisplayData>,
    pub current_phase: Option<PhaseDisplayData>,
    pub messages: Vec<DisplayMessage>,
}

/// Log message for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayMessage {
    pub message: String,
    pub timestamp: String,
    pub level: String,
}

/// Metrics Aggregator Actor - collects and computes statistics
pub struct MetricsAggregatorActor {
    /// Run ID
    run_id: Option<String>,
    /// Configuration
    config: Option<BenchmarkConfig>,
    /// Start time
    start_time: Option<Instant>,
    /// Current phase statistics
    current_phase: Option<PhaseStats>,
    /// Completed phases
    completed_phases: Vec<PhaseStats>,
    /// Console renderer reference
    renderer_ref: Option<ActorRef>,
    /// Display refresh interval
    refresh_interval: Duration,
    /// Last display update
    last_display: Instant,
    /// Recent log messages
    messages: Vec<DisplayMessage>,
    /// Expected duration for current phase
    expected_duration: Duration,
}

impl MetricsAggregatorActor {
    pub fn new() -> Self {
        Self {
            run_id: None,
            config: None,
            start_time: None,
            current_phase: None,
            completed_phases: Vec::new(),
            renderer_ref: None,
            refresh_interval: Duration::from_millis(500),
            last_display: Instant::now(),
            messages: Vec::new(),
            expected_duration: Duration::from_secs(60),
        }
    }

    pub fn with_renderer(mut self, renderer: ActorRef) -> Self {
        self.renderer_ref = Some(renderer);
        self
    }

    fn add_message(&mut self, message: String) {
        self.messages.push(DisplayMessage {
            message,
            timestamp: chrono::Utc::now().format("%H:%M:%S").to_string(),
            level: "INFO".to_string(),
        });
        // Keep only last 50 messages
        if self.messages.len() > 50 {
            self.messages.remove(0);
        }
    }

    fn register_run(&mut self, register: RegisterRun) {
        self.run_id = Some(register.run_id);
        self.config = Some(register.config.clone());
        self.start_time = Some(Instant::now());
        self.completed_phases.clear();
        self.current_phase = None;
        self.messages.clear();
        self.expected_duration = Duration::from_secs(register.config.duration_secs);

        // Notify renderer of config
        self.send_display_update();
    }

    fn start_phase(&mut self, phase: StartPhase) {
        // End current phase if any
        if let Some(mut current) = self.current_phase.take() {
            current.end_time = Some(Instant::now());
            self.completed_phases.push(current);
        }

        let stats = PhaseStats::new(
            phase.phase_id.clone(),
            phase.phase_name.clone(),
            phase.scheduler_type,
            phase.target_rate,
        );
        self.current_phase = Some(stats);

        self.add_message(format!("Starting phase: {}", phase.phase_name));
        self.send_display_update();
    }

    fn end_phase(&mut self, _end: EndPhase) {
        if let Some(mut current) = self.current_phase.take() {
            current.end_time = Some(Instant::now());

            self.add_message(format!(
                "Phase {} completed: {} requests at {:.2} req/s",
                current.phase_name,
                current.successful_requests,
                current.request_rate()
            ));

            self.completed_phases.push(current);
        }

        self.send_display_update();
    }

    fn add_result(&mut self, result: RequestCompleted) {
        let should_update = if let Some(ref mut phase) = self.current_phase {
            phase.add_result(&result);
            self.last_display.elapsed() >= self.refresh_interval
        } else {
            false
        };

        if should_update {
            self.send_display_update();
            self.last_display = Instant::now();
        }
    }

    fn send_display_update(&self) {
        if let Some(ref renderer) = self.renderer_ref {
            let update = self.build_display_update();
            // Fire and forget - don't block on display
            let renderer = renderer.clone();
            tokio::spawn(async move {
                let _ = renderer.tell(update).await;
            });
        }
    }

    fn build_display_update(&self) -> DisplayUpdate {
        let completed: Vec<PhaseDisplayData> = self
            .completed_phases
            .iter()
            .map(|p| p.to_display_snapshot(self.expected_duration))
            .collect();

        let current = self
            .current_phase
            .as_ref()
            .map(|p| p.to_display_snapshot(self.expected_duration));

        DisplayUpdate {
            config: self.config.clone(),
            completed_phases: completed,
            current_phase: current,
            messages: self.messages.clone(),
        }
    }

    fn get_metrics(&self) -> MetricsSnapshot {
        self.current_phase
            .as_ref()
            .map(|p| p.to_metrics())
            .unwrap_or_else(|| {
                self.completed_phases
                    .last()
                    .map(|p| p.to_metrics())
                    .unwrap_or_default()
            })
    }

    fn get_report(&self) -> BenchmarkReport {
        let phases: Vec<PhaseReport> = self
            .completed_phases
            .iter()
            .map(|p| PhaseReport {
                phase_id: p.phase_id.clone(),
                phase_name: p.phase_name.clone(),
                duration_secs: p.duration().as_secs_f64(),
                metrics: p.to_metrics(),
            })
            .collect();

        let total_duration = self
            .start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        BenchmarkReport {
            run_id: self.run_id.clone().unwrap_or_default(),
            config: self.config.clone().unwrap_or_default(),
            phases,
            start_time: chrono::Utc::now().to_rfc3339(),
            end_time: chrono::Utc::now().to_rfc3339(),
            total_duration_secs: total_duration,
        }
    }

    /// Send final update to renderer
    fn finalize(&self) {
        if let Some(ref renderer) = self.renderer_ref {
            // Send final display update
            let update = self.build_display_update();
            let renderer = renderer.clone();

            // Also send a FinalReport message
            let report = self.get_report();
            tokio::spawn(async move {
                let _ = renderer.tell(update).await;
                let _ = renderer.tell(FinalReport { report }).await;
            });
        }
    }
}

impl Default for MetricsAggregatorActor {
    fn default() -> Self {
        Self::new()
    }
}

/// Message to signal final report is ready
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalReport {
    pub report: BenchmarkReport,
}

#[async_trait]
impl Actor for MetricsAggregatorActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> anyhow::Result<()> {
        info!("MetricsAggregator started with actor_id {:?}", ctx.id());
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> anyhow::Result<()> {
        self.finalize();
        info!("MetricsAggregator stopped");
        Ok(())
    }

    async fn receive(&mut self, msg: Message, _ctx: &mut ActorContext) -> anyhow::Result<Message> {
        let msg_type = msg.msg_type();

        if msg_type.ends_with("RegisterRun") {
            let register: RegisterRun = msg.unpack()?;
            self.register_run(register);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("StartPhase") {
            let phase: StartPhase = msg.unpack()?;
            self.start_phase(phase);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("EndPhase") {
            let end: EndPhase = msg.unpack()?;
            self.end_phase(end);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("RequestCompleted") {
            let result: RequestCompleted = msg.unpack()?;
            self.add_result(result);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("GetMetrics") {
            return Message::pack(&self.get_metrics());
        }

        if msg_type.ends_with("GetReport") {
            return Message::pack(&self.get_report());
        }

        Err(anyhow::anyhow!("Unknown message type: {}", msg_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_stats() {
        let mut stats = PhaseStats::new(
            "test".to_string(),
            "Test Phase".to_string(),
            SchedulerType::ConstantVUs,
            None,
        );

        let result = RequestCompleted {
            request_id: "1".to_string(),
            success: true,
            error: None,
            ttft_ms: Some(100.0),
            latency_ms: 500.0,
            generated_tokens: 50,
            prompt_tokens: 10,
            tpot_ms: Some(8.0),
            token_times_ms: vec![],
        };

        stats.add_result(&result);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.avg_ttft_ms(), Some(100.0));
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // For 10 elements, 50th percentile rounds to index 5 (6.0)
        assert_eq!(PhaseStats::percentile(&values, 50.0), Some(6.0));
        // For 10 elements, 90th percentile rounds to index 8 (9.0)
        assert_eq!(PhaseStats::percentile(&values, 90.0), Some(9.0));
    }
}
