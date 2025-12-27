"""Pulsing CLI - Main entry point"""

import hyperparameter as hp


@hp.param("actor")
def actor(
    actor_type: str,  # 位置参数，支持: router, transformers, vllm
    namespace: str = "pulsing",
    addr: str | None = None,
    seeds: str | None = None,
    model: str | None = None,
    model_name: str = "pulsing-model",
    device: str = "cuda",
    max_new_tokens: int = 512,
    role: str = "aggregated",
    preload_model: bool = False,
    http_host: str = "0.0.0.0",
    http_port: int = 8080,
    scheduler: str = "stream_load",
    # macOS Metal/MLX 支持参数
    mlx_device: str | None = None,  # 'gpu' 或 'cpu'，默认 'gpu'
    metal_memory_fraction: float | None = None,  # 0.0-1.0，默认 0.8
):
    """
    Start an Actor-based service.

    This command starts actors based on the Pulsing Actor System.
    Supported actor types:
    - router: Load balancing router (with OpenAI-compatible HTTP API)
    - transformers: Transformers-based inference worker
    - vllm: vLLM-based high-performance inference worker

    Args:
        actor_type: Actor type (positional argument). Options: 'router', 'transformers', 'vllm'
        namespace: Service namespace. Default: 'pulsing'
        addr: Actor System bind address (e.g., '0.0.0.0:8000')
        seeds: Comma-separated list of seed nodes (e.g., '192.168.1.1:8000,192.168.1.2:8000')
        model: Model path (required for 'transformers' and 'vllm' type)
        model_name: Model name for OpenAI API. Default: 'pulsing-model'
        device: Device for inference ('cuda', 'cpu', 'mps'). Default: 'cuda'
        max_new_tokens: Max tokens to generate. Default: 512
        role: Worker role for vLLM ('aggregated', 'prefill', 'decode'). Default: 'aggregated'
        scheduler: Scheduler algorithm for router. Options: 'round_robin', 'random', 'least_connection', 'stream_load'. Default: 'stream_load'
        preload_model: Preload model on startup. Default: False
        http_host: HTTP server host (for router). Default: '0.0.0.0'
        http_port: HTTP server port (for router). Default: 8080
        mlx_device: MLX device type for macOS ('gpu' or 'cpu'). Default: 'gpu'
        metal_memory_fraction: Metal memory fraction for macOS (0.0-1.0). Default: 0.8

    Examples:
        # Start a router with OpenAI-compatible API on port 8080
        pulsing actor router --http_port 8080 --model_name my-llm

        # Test with curl
        curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"my-llm","messages":[{"role":"user","content":"Hello"}]}'

        # Start a vLLM worker
        pulsing actor vllm --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000

        # Start a vLLM worker on macOS with Metal support
        pulsing actor vllm --model Qwen/Qwen3-0.6B --mlx_device gpu --metal_memory_fraction 0.8 --addr 0.0.0.0:8001

        # Start a transformers worker
        pulsing actor transformers --model Qwen/Qwen3-0.6B --addr 0.0.0.0:8001 --seeds 192.168.1.100:8000

        # Start worker on CPU
        pulsing actor transformers --model gpt2 --device cpu
    """
    from .actors import start_router, start_transformers, start_vllm

    # Parse seeds
    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]

    if actor_type == "router":
        start_router(
            namespace, addr, seed_list, http_host, http_port, model_name, scheduler
        )
    elif actor_type == "transformers":
        if not model:
            raise ValueError("--model is required for 'transformers' actor type")
        start_transformers(
            model=model,
            namespace=namespace,
            addr=addr,
            seeds=seed_list,
            device=device,
            max_new_tokens=max_new_tokens,
            preload_model=preload_model,
        )
    elif actor_type == "vllm":
        if not model:
            raise ValueError("--model is required for 'vllm' actor type")
        start_vllm(
            model=model,
            namespace=namespace,
            addr=addr,
            seeds=seed_list,
            max_new_tokens=max_new_tokens,
            role=role,
            mlx_device=mlx_device,
            metal_memory_fraction=metal_memory_fraction,
        )
    else:
        raise ValueError(
            f"Unknown actor type: {actor_type}. Supported types: router, transformers, vllm"
        )


@hp.param("inspect")
def inspect(seeds: str | None = None):
    """
    Inspect the actor system state.

    Args:
        seeds: Comma-separated list of seed nodes to join the cluster.

    Examples:
        pulsing inspect --seeds 127.0.0.1:8000
    """
    from .inspect import inspect_system

    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
    inspect_system(seed_list)


@hp.param("bench")
def bench(
    tokenizer_name: str,
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
):
    """
    Run inference benchmarks.

    This command runs the pulsing benchmark tool with the specified parameters.

    Args:
        tokenizer_name: The name of the tokenizer to use (required)
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

    Examples:
        pulsing bench --tokenizer_name gpt2 --url http://localhost:8080
    """
    from .bench import run_benchmark

    run_benchmark(
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        max_vus=max_vus,
        duration=duration,
        rates=rates,
        num_rates=num_rates,
        profile=profile,
        benchmark_kind=benchmark_kind,
        warmup=warmup,
        url=url,
        api_key=api_key,
        prompt_options=prompt_options,
        decode_options=decode_options,
        dataset=dataset,
        dataset_file=dataset_file,
        extra_meta=extra_meta,
        run_id=run_id,
    )


def main():
    hp.launch()


if __name__ == "__main__":
    main()
