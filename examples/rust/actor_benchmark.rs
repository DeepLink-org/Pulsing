//! Benchmark Example
//!
//! This example demonstrates how to use the Actor-based benchmark architecture.
//!
//! Architecture:
//! - Worker Actor: Sends HTTP requests to the target endpoint
//! - Scheduler Actor: Controls request timing (constant VUs or constant rate)
//! - Coordinator Actor: Orchestrates the benchmark lifecycle
//! - MetricsAggregator Actor: Collects metrics and calculates statistics
//! - ConsoleRenderer Actor: Displays real-time progress
//!
//! Run with:
//! ```bash
//! cargo run --example actor_benchmark -- --url http://localhost:8000 --model gpt2
//! ```

use pulsing_bench::{run_benchmark, BenchmarkArgs};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse command line arguments (simplified)
    let args: Vec<String> = std::env::args().collect();

    let mut benchmark_args = BenchmarkArgs::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--url" if i + 1 < args.len() => {
                benchmark_args.url = args[i + 1].clone();
                i += 2;
            }
            "--model" if i + 1 < args.len() => {
                benchmark_args.model_name = args[i + 1].clone();
                i += 2;
            }
            "--max-vus" if i + 1 < args.len() => {
                benchmark_args.max_vus = args[i + 1].parse()?;
                i += 2;
            }
            "--duration" if i + 1 < args.len() => {
                benchmark_args.duration_secs = args[i + 1].parse()?;
                i += 2;
            }
            "--warmup" if i + 1 < args.len() => {
                benchmark_args.warmup_secs = args[i + 1].parse()?;
                i += 2;
            }
            "--workers" if i + 1 < args.len() => {
                benchmark_args.num_workers = args[i + 1].parse()?;
                i += 2;
            }
            "--kind" if i + 1 < args.len() => {
                benchmark_args.benchmark_kind = args[i + 1].clone();
                i += 2;
            }
            "--api-key" if i + 1 < args.len() => {
                benchmark_args.api_key = args[i + 1].clone();
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("Starting benchmark...");
    println!("  URL: {}", benchmark_args.url);
    println!("  Model: {}", benchmark_args.model_name);
    println!("  Max VUs: {}", benchmark_args.max_vus);
    println!("  Duration: {}s", benchmark_args.duration_secs);
    println!("  Workers: {}", benchmark_args.num_workers);
    println!("  Kind: {}", benchmark_args.benchmark_kind);
    println!();

    // Run the benchmark
    let report = run_benchmark(benchmark_args).await?;

    // Print summary
    println!("\n=== Benchmark Report ===");
    println!("Run ID: {}", report.run_id);
    println!("Total Duration: {:.2}s", report.total_duration_secs);
    println!("Phases completed: {}", report.phases.len());

    for phase in &report.phases {
        println!("\n--- {} ---", phase.phase_name);
        println!("  Successful requests: {}", phase.metrics.successful_requests);
        println!("  Failed requests: {}", phase.metrics.failed_requests);
        println!("  Request rate: {:.2} req/s", phase.metrics.request_rate);
        if let Some(ttft) = phase.metrics.avg_ttft_ms {
            println!("  Avg TTFT: {:.1}ms", ttft);
        }
        if let Some(tpot) = phase.metrics.avg_tpot_ms {
            println!("  Avg TPOT: {:.2}ms", tpot);
        }
    }

    Ok(())
}

fn print_help() {
    println!(
        r#"
Pulsing Benchmark Example

USAGE:
    actor_benchmark [OPTIONS]

OPTIONS:
    --url <URL>           Target URL (default: http://localhost:8000)
    --model <MODEL>       Model name (default: gpt2)
    --max-vus <N>         Maximum virtual users (default: 128)
    --duration <SECS>     Test duration in seconds (default: 120)
    --warmup <SECS>       Warmup duration in seconds (default: 30)
    --workers <N>         Number of worker actors (default: 4)
    --kind <KIND>         Benchmark kind: throughput, sweep, rate, csweep (default: throughput)
    --api-key <KEY>       API key for authentication
    -h, --help            Print this help message

BENCHMARK KINDS:
    throughput  - Run at maximum throughput (constant VUs)
    sweep       - Find max throughput, then test at various rates
    rate        - Test at specific rates (use --rates)
    csweep      - Concurrency sweep to find optimal VU count

EXAMPLES:
    # Simple throughput test
    actor_benchmark --url http://localhost:8000 --model gpt2 --duration 60

    # Concurrency sweep
    actor_benchmark --url http://localhost:8000 --model gpt2 --kind csweep --max-vus 100

    # With authentication
    actor_benchmark --url http://api.example.com --api-key sk-xxx --model gpt-4
"#
    );
}
