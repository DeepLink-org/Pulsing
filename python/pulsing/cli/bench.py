"""Pulsing CLI - Benchmark commands"""

import asyncio


def run_benchmark(
    model_name: str,
    url: str = "http://localhost:8000",
    api_key: str = "",
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
        max_vus: Maximum number of virtual users (default: 128)
        duration: Duration of each benchmark step (default: "120s")
        warmup: Warmup duration (default: "30s")
        benchmark_kind: Kind of benchmark - throughput, sweep, csweep, rate (default: "throughput")
        num_rates: Number of rates to sweep (default: 10)
        rates: List of rates for rate benchmark
        num_workers: Number of worker actors (default: 4)
    """
    from pulsing._core import benchmark_main

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
