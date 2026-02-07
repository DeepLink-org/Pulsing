//! Console Renderer Actor - handles terminal display
//!
//! This actor is responsible for:
//! - Receiving display updates from MetricsAggregator
//! - Rendering benchmark progress to the terminal
//! - Displaying tables, progress bars, and log messages
//!
//! Separation from metrics collection allows:
//! - Easy replacement with other output formats (JSON, Web UI, etc.)
//! - Testing metrics logic without terminal dependencies
//! - Clean separation of concerns

use super::messages::*;
use super::metrics_aggregator::{DisplayUpdate, FinalReport, PhaseDisplayData};
use async_trait::async_trait;
use colored::Colorize;
use pulsing_actor::prelude::*;
use std::io::{self, Write};
use tracing::info;

/// Console Renderer Actor - pure display logic
pub struct ConsoleRendererActor {
    /// Enable display
    enabled: bool,
    /// Last received config
    config: Option<BenchmarkConfig>,
    /// Has printed header
    header_printed: bool,
}

impl ConsoleRendererActor {
    pub fn new() -> Self {
        Self {
            enabled: true,
            config: None,
            header_printed: false,
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    fn handle_display_update(&mut self, update: DisplayUpdate) {
        if !self.enabled {
            return;
        }

        // Store config if provided
        if update.config.is_some() {
            self.config = update.config.clone();
        }

        // Print header on first update with config
        if !self.header_printed {
            if let Some(ref config) = self.config {
                self.print_header(config);
                self.header_printed = true;
            }
        }

        // Render the display
        self.render_progress(&update);
    }

    fn print_header(&self, config: &BenchmarkConfig) {
        println!("{}", "=".repeat(80).blue());
        println!("{}", "🚀 PULSING BENCHMARK (Actor Edition)".blue().bold());
        println!("{}", "=".repeat(80).blue());

        println!(
            "{}",
            format!(
                "Target: {} | Model: {} | Max VUs: {} | Duration: {}s | Warmup: {}s | Kind: {}",
                config.url,
                config.model_name,
                config.max_vus,
                config.duration_secs,
                config.warmup_secs,
                config.benchmark_kind
            )
            .cyan()
            .bold()
        );

        println!("{}", "=".repeat(80).blue());
        println!();
    }

    fn render_progress(&self, update: &DisplayUpdate) {
        // Clear screen and move cursor to top
        print!("\x1B[2J\x1B[1;1H");

        // Re-print header if we have config
        if let Some(ref config) = self.config {
            self.print_header(config);
        }

        // Print benchmark progress header
        println!("{}", "BENCHMARK PROGRESS".blue().bold());
        println!("{}", "-".repeat(120).blue());

        // Print table header
        println!(
            "{:<20} {:<10} {:>6} {:>6} {:>12} {:>18} {:>18} {:>10} {:>10} {:>12}",
            "Bench".blue().bold(),
            "Status".blue().bold(),
            "%".blue().bold(),
            "Err%".blue().bold(),
            "Throughput".blue().bold(),
            "TTFT".blue().bold(),
            "TPOT".blue().bold(),
            "In tok/s".blue().bold(),
            "Out tok/s".blue().bold(),
            "Total tok/s".blue().bold()
        );
        println!("{}", "-".repeat(120).blue());

        // Print completed phases
        for phase in &update.completed_phases {
            self.print_phase_row(phase);
        }

        // Print current phase
        if let Some(ref phase) = update.current_phase {
            self.print_phase_row(phase);

            // Progress bar for running phase
            if phase.progress_pct < 100.0 {
                println!(
                    "  {} {}",
                    "Progress:".dimmed(),
                    create_progress_bar(phase.progress_pct)
                );
            }
        }

        println!();

        // Print recent messages
        if !update.messages.is_empty() {
            println!("{}", "RECENT MESSAGES".blue().bold());
            println!("{}", "-".repeat(80).blue());

            for message in update.messages.iter().rev().take(5) {
                let level_str = match message.level.as_str() {
                    "INFO" => "INFO".green(),
                    "WARN" | "WARNING" => "WARN".yellow(),
                    "ERROR" => "ERROR".red(),
                    _ => message.level.normal(),
                };
                println!(
                    "{} {} {}",
                    message.timestamp.bright_black(),
                    level_str,
                    message.message.white()
                );
            }
            println!();
        }

        io::stdout().flush().unwrap();
    }

    fn print_phase_row(&self, phase: &PhaseDisplayData) {
        let status = if phase.is_completed {
            "Completed".green().bold()
        } else {
            "Running".yellow().bold()
        };

        let error_str = if phase.error_rate > 0.0 {
            format!("{:.1}%", phase.error_rate).red().bold()
        } else {
            "0%".green().normal()
        };

        let ttft_display = match (phase.avg_ttft_ms, phase.ttft_std_ms) {
            (Some(mean), Some(std)) => format!("{:.1}±{:.1}ms", mean, std),
            (Some(mean), None) => format!("{:.1}ms", mean),
            _ => "N/A".to_string(),
        };

        let tpot_display = match (phase.avg_tpot_ms, phase.tpot_std_ms) {
            (Some(mean), Some(std)) => format!("{:.2}±{:.2}ms", mean, std),
            (Some(mean), None) => format!("{:.2}ms", mean),
            _ => "N/A".to_string(),
        };

        let input_tput = phase
            .input_throughput
            .map_or("N/A".to_string(), |t| format!("{:.1}", t));
        let output_tput = phase
            .output_throughput
            .map_or("N/A".to_string(), |t| format!("{:.1}", t));
        let total_tput = phase
            .total_throughput
            .map_or("N/A".to_string(), |t| format!("{:.1}", t));

        println!(
            "{:<20} {:<10} {:>5.0}% {:>6} {:>10.2} r/s {:>18} {:>18} {:>10} {:>10} {:>12}",
            truncate(&phase.phase_name, 20).white(),
            status,
            phase.progress_pct,
            error_str,
            phase.request_rate,
            ttft_display.cyan(),
            tpot_display.magenta(),
            input_tput.blue(),
            output_tput.yellow(),
            total_tput.green()
        );
    }

    fn handle_final_report(&self, final_report: FinalReport) {
        if !self.enabled {
            return;
        }

        let report = &final_report.report;

        println!();
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .bright_green()
                .bold()
        );
        println!(
            "{}",
            "║                           BENCHMARK COMPLETE                                 ║"
                .bright_green()
                .bold()
        );
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .bright_green()
                .bold()
        );
        println!();

        // Print results table
        println!(
            "{:<20} {:>12} {:>14} {:>14} {:>14} {:>12} {:>10}",
            "Phase".bold(),
            "QPS".bold(),
            "E2E Latency".bold(),
            "TTFT (avg)".bold(),
            "TPOT (avg)".bold(),
            "Throughput".bold(),
            "Error %".bold()
        );
        println!("{}", "─".repeat(100));

        for phase in &report.phases {
            let metrics = &phase.metrics;
            let e2e = metrics
                .avg_latency_ms
                .map_or("N/A".to_string(), |v| format!("{:.0}ms", v));
            let ttft = metrics
                .avg_ttft_ms
                .map_or("N/A".to_string(), |v| format!("{:.1}ms", v));
            let tpot = metrics
                .avg_tpot_ms
                .map_or("N/A".to_string(), |v| format!("{:.2}ms", v));
            let throughput = metrics
                .total_throughput
                .map_or("N/A".to_string(), |v| format!("{:.0} t/s", v));

            let total = metrics.successful_requests + metrics.failed_requests;
            let error_pct = if total > 0 {
                metrics.failed_requests as f64 / total as f64 * 100.0
            } else {
                0.0
            };

            println!(
                "{:<20} {:>10.2}/s {:>14} {:>14} {:>14} {:>12} {:>9.1}%",
                truncate(&phase.phase_name, 20),
                metrics.request_rate,
                e2e,
                ttft,
                tpot,
                throughput,
                error_pct
            );
        }

        println!("{}", "─".repeat(100));

        let total_requests: u64 = report
            .phases
            .iter()
            .map(|p| p.metrics.successful_requests)
            .sum();
        println!(
            "{}",
            format!(
                "Total duration: {:.1}s | Total requests: {}",
                report.total_duration_secs, total_requests
            )
            .dimmed()
        );
        println!();

        io::stdout().flush().unwrap();
    }
}

impl Default for ConsoleRendererActor {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to truncate strings
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

/// Helper function to create a progress bar
fn create_progress_bar(progress: f64) -> String {
    let width = 30;
    let filled = (progress / 100.0 * width as f64) as usize;
    let empty = width - filled;

    let filled_bar = "█".repeat(filled);
    let empty_bar = "░".repeat(empty);

    format!(
        "[{}{}] {:>5.1}%",
        filled_bar.green(),
        empty_bar.dimmed(),
        progress
    )
}

#[async_trait]
impl Actor for ConsoleRendererActor {
    async fn on_start(&mut self, ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        info!("ConsoleRenderer started with actor_id {:?}", ctx.id());
        Ok(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext) -> pulsing_actor::error::Result<()> {
        info!("ConsoleRenderer stopped");
        Ok(())
    }

    async fn receive(
        &mut self,
        msg: Message,
        _ctx: &mut ActorContext,
    ) -> pulsing_actor::error::Result<Message> {
        let msg_type = msg.msg_type();

        if msg_type.ends_with("DisplayUpdate") {
            let update: DisplayUpdate = msg.unpack()?;
            self.handle_display_update(update);
            return Message::pack(&AckMessage::ok());
        }

        if msg_type.ends_with("FinalReport") {
            let report: FinalReport = msg.unpack()?;
            self.handle_final_report(report);
            return Message::pack(&AckMessage::ok());
        }

        // Ignore unknown messages silently - renderer is fire-and-forget
        Message::pack(&AckMessage::ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actors::metrics_aggregator::DisplayMessage;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_progress_bar() {
        let bar = create_progress_bar(50.0);
        assert!(bar.contains("50.0%"));
    }

    #[test]
    fn test_renderer_creation() {
        let renderer = ConsoleRendererActor::new();
        assert!(renderer.enabled);
        assert!(!renderer.header_printed);
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 10), "");
    }

    #[test]
    fn test_truncate_short_max() {
        // When max_len is very short, we still try to truncate
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_progress_bar_zero() {
        let bar = create_progress_bar(0.0);
        assert!(bar.contains("0.0%"));
        // Should have all empty blocks
        assert!(bar.contains("░"));
    }

    #[test]
    fn test_progress_bar_full() {
        let bar = create_progress_bar(100.0);
        assert!(bar.contains("100.0%"));
        // Should have all filled blocks
        assert!(bar.contains("█"));
    }

    #[test]
    fn test_progress_bar_partial() {
        let bar = create_progress_bar(75.0);
        assert!(bar.contains("75.0%"));
        // Should have both filled and empty blocks
        assert!(bar.contains("█"));
        assert!(bar.contains("░"));
    }

    #[test]
    fn test_renderer_default() {
        let renderer = ConsoleRendererActor::default();
        assert!(renderer.enabled);
        assert!(!renderer.header_printed);
        assert!(renderer.config.is_none());
    }

    #[test]
    fn test_renderer_with_enabled() {
        let renderer = ConsoleRendererActor::new().with_enabled(false);
        assert!(!renderer.enabled);
    }

    #[test]
    fn test_renderer_disabled_display_update() {
        let mut renderer = ConsoleRendererActor::new().with_enabled(false);
        let update = DisplayUpdate {
            config: Some(BenchmarkConfig::default()),
            completed_phases: vec![],
            current_phase: None,
            messages: vec![],
        };

        // Should not panic even when disabled
        renderer.handle_display_update(update);
        // Config should not be stored when disabled
        assert!(renderer.config.is_none());
    }

    #[test]
    fn test_renderer_disabled_final_report() {
        let renderer = ConsoleRendererActor::new().with_enabled(false);
        let report = BenchmarkReport {
            run_id: "test".to_string(),
            config: BenchmarkConfig::default(),
            phases: vec![],
            start_time: "".to_string(),
            end_time: "".to_string(),
            total_duration_secs: 60.0,
        };

        // Should not panic when disabled
        renderer.handle_final_report(FinalReport { report });
    }

    #[test]
    fn test_phase_display_data_formatting() {
        let phase = PhaseDisplayData {
            phase_id: "phase-1".to_string(),
            phase_name: "Test Phase".to_string(),
            is_completed: true,
            progress_pct: 100.0,
            error_rate: 5.0,
            request_rate: 50.0,
            avg_ttft_ms: Some(100.0),
            ttft_std_ms: Some(10.0),
            avg_tpot_ms: Some(8.0),
            tpot_std_ms: Some(1.5),
            input_throughput: Some(500.0),
            output_throughput: Some(1000.0),
            total_throughput: Some(1500.0),
        };

        assert!(phase.is_completed);
        assert_eq!(phase.progress_pct, 100.0);
        assert_eq!(phase.error_rate, 5.0);
    }

    #[test]
    fn test_phase_display_data_none_values() {
        let phase = PhaseDisplayData {
            phase_id: "phase-1".to_string(),
            phase_name: "Test Phase".to_string(),
            is_completed: false,
            progress_pct: 50.0,
            error_rate: 0.0,
            request_rate: 25.0,
            avg_ttft_ms: None,
            ttft_std_ms: None,
            avg_tpot_ms: None,
            tpot_std_ms: None,
            input_throughput: None,
            output_throughput: None,
            total_throughput: None,
        };

        assert!(!phase.is_completed);
        assert!(phase.avg_ttft_ms.is_none());
        assert!(phase.total_throughput.is_none());
    }

    #[test]
    fn test_display_update_with_config() {
        let mut renderer = ConsoleRendererActor::new();
        let config = BenchmarkConfig::default();
        let update = DisplayUpdate {
            config: Some(config.clone()),
            completed_phases: vec![],
            current_phase: None,
            messages: vec![],
        };

        renderer.handle_display_update(update);

        // Config should be stored
        assert!(renderer.config.is_some());
        assert_eq!(renderer.config.as_ref().unwrap().url, config.url);
    }

    #[test]
    fn test_display_update_preserves_config() {
        let mut renderer = ConsoleRendererActor::new();

        // First update with config
        let config = BenchmarkConfig {
            url: "http://test.com".to_string(),
            ..Default::default()
        };
        renderer.handle_display_update(DisplayUpdate {
            config: Some(config),
            completed_phases: vec![],
            current_phase: None,
            messages: vec![],
        });

        // Second update without config
        renderer.handle_display_update(DisplayUpdate {
            config: None,
            completed_phases: vec![],
            current_phase: None,
            messages: vec![],
        });

        // Config should still be preserved
        assert!(renderer.config.is_some());
        assert_eq!(renderer.config.as_ref().unwrap().url, "http://test.com");
    }

    #[test]
    fn test_display_message_levels() {
        let info_msg = DisplayMessage {
            message: "Info message".to_string(),
            timestamp: "12:00:00".to_string(),
            level: "INFO".to_string(),
        };
        assert_eq!(info_msg.level, "INFO");

        let warn_msg = DisplayMessage {
            message: "Warning message".to_string(),
            timestamp: "12:00:01".to_string(),
            level: "WARN".to_string(),
        };
        assert_eq!(warn_msg.level, "WARN");

        let error_msg = DisplayMessage {
            message: "Error message".to_string(),
            timestamp: "12:00:02".to_string(),
            level: "ERROR".to_string(),
        };
        assert_eq!(error_msg.level, "ERROR");
    }
}
