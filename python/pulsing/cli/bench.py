"""Pulsing CLI - Benchmark commands"""

import uvloop


def run_benchmark(
    tokenizer_name: str | None = None,
    model_name: str | None = None,
    max_vus: int = 128,
    duration: str = "120s",
    rates: list | None = None,
    num_rates: int = 10,
    profile: str | None = None,
    benchmark_kind: str = "sweep",
    warmup: str = "30s",
    url: str = "http://localhost:8000",
    api_key: str = "",
    prompt_options: str | None = None,
    decode_options: str | None = None,
    dataset: str = "hlarcher/inference-benchmarker",
    dataset_file: str = "share_gpt_filtered_small.json",
    extra_meta: str | None = None,
    run_id: str | None = None,
    engine: str = "classic",
    num_workers: int = 4,
):
    """Run inference benchmarks

    Args:
        tokenizer_name: The name of the tokenizer to use (required for classic engine)
        model_name: The name of the model to use. If not provided, same as tokenizer_name
        max_vus: Maximum number of virtual users (default: 128)
        duration: Duration of each benchmark step (default: "120s")
        rates: List of rates for ConstantArrivalRate benchmark
        num_rates: Number of rates to sweep (default: 10)
        profile: Benchmark profile to use
        benchmark_kind: Kind of benchmark - throughput, sweep, csweep, rate (default: "sweep")
        warmup: Warmup duration (default: "30s")
        url: Backend URL (default: "http://localhost:8000")
        api_key: API key for authentication (default: "")
        prompt_options: Prompt tokenizer options as "key=value,key2=value2"
        decode_options: Decode tokenizer options as "key=value,key2=value2"
        dataset: Hugging Face dataset name (default: "hlarcher/inference-benchmarker")
        dataset_file: Dataset file name (default: "share_gpt_filtered_small.json")
        extra_meta: Extra metadata as "key1=value1,key2=value2"
        run_id: Run identifier for results file
        engine: Benchmark engine - "classic" (original) or "actor" (new) (default: "classic")
        num_workers: Number of worker actors (only for actor engine, default: 4)
    """
    from pulsing._core import benchmark_main

    config = {
        "max_vus": max_vus,
        "duration": duration,
        "num_rates": num_rates,
        "benchmark_kind": benchmark_kind,
        "warmup": warmup,
        "url": url,
        "api_key": api_key,
        "dataset": dataset,
        "dataset_file": dataset_file,
        "engine": engine,
        "num_workers": num_workers,
    }

    # tokenizer_name is required for classic engine
    if engine == "classic":
        if tokenizer_name is None:
            raise ValueError("tokenizer_name is required for classic engine")
        config["tokenizer_name"] = tokenizer_name
    else:
        # For actor engine, model_name is required
        if model_name is None and tokenizer_name is None:
            raise ValueError("model_name (or tokenizer_name) is required for actor engine")
        if model_name is not None:
            config["model_name"] = model_name
        elif tokenizer_name is not None:
            config["model_name"] = tokenizer_name

    if model_name is not None:
        config["model_name"] = model_name
    if rates is not None:
        config["rates"] = rates
    if profile is not None:
        config["profile"] = profile
    if prompt_options is not None:
        config["prompt_options"] = prompt_options
    if decode_options is not None:
        config["decode_options"] = decode_options
    if extra_meta is not None:
        config["extra_meta"] = extra_meta
    if run_id is not None:
        config["run_id"] = run_id

    result = uvloop.run(benchmark_main(config))
    
    # For actor engine, result is a JSON string
    if engine == "actor" and result:
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


