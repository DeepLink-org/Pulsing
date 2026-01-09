"""Pulsing CLI - Benchmark commands"""

import asyncio
import os


def run_benchmark(
    model_name: str,
    url: str = "http://localhost:8000",
    api_key: str = "",
    tokenizer_name: str | None = None,
    hf_token: str | None = None,
    max_vus: int = 128,
    duration: str = "120s",
    warmup: str = "30s",
    benchmark_kind: str = "throughput",
    num_rates: int = 10,
    rates: list | None = None,
    num_workers: int = 4,
):
    """Run inference benchmarks

    Args:
        model_name: The name of the model to benchmark (required)
        url: Backend URL (default: "http://localhost:8000")
        api_key: API key for authentication (default: "")
        tokenizer_name: HuggingFace tokenizer name (default: same as model_name)
        hf_token: HuggingFace token for private models (default: from HF_TOKEN env)
        max_vus: Maximum number of virtual users (default: 128)
        duration: Duration of each benchmark step (default: "120s")
        warmup: Warmup duration (default: "30s")
        benchmark_kind: Kind of benchmark - throughput, sweep, csweep, rate (default: "throughput")
        num_rates: Number of rates to sweep (default: 10)
        rates: List of rates for rate benchmark
        num_workers: Number of worker actors (default: 4)
    """
    try:
        from pulsing._bench import benchmark_main
    except ImportError:
        print(
            "Error: 'pulsing._bench' module not found.\n"
            "The benchmark tools are optional and need to be installed separately.\n\n"
            "To install them, please run:\n"
            "  maturin develop --manifest-path crates/pulsing-bench-py/Cargo.toml"
        )
        return

    # Get HF token from environment if not provided
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    config = {
        "model_name": model_name,
        "url": url,
        "api_key": api_key,
        "max_vus": max_vus,
        "duration": duration,
        "warmup": warmup,
        "benchmark_kind": benchmark_kind,
        "num_rates": num_rates,
        "num_workers": num_workers,
    }

    if tokenizer_name is not None:
        config["tokenizer_name"] = tokenizer_name

    if hf_token is not None:
        config["hf_token"] = hf_token

    if rates is not None:
        config["rates"] = rates

    # Run the async benchmark
    async def _run():
        return await benchmark_main(config)

    # Try to use uvloop for better performance
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

    result = asyncio.run(_run())

    # Result is a JSON string with the benchmark report
    if result:
        import json

        print("\n" + "=" * 80)
        print("BENCHMARK REPORT (JSON)")
        print("=" * 80)
        try:
            report = json.loads(result)
            print(json.dumps(report, indent=2))
        except json.JSONDecodeError:
            print(result)

    return result
