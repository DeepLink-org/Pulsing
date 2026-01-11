"""Pulsing CLI - Main entry point"""

import hyperparameter as hp


@hp.param("actor")
def actor(
    actor_type: str,  # 位置参数，支持: router, transformers, vllm, list
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
    # Actor list parameters
    endpoint: str | None = None,  # Single actor system endpoint (list only)
    all_actors: bool = False,  # Show all actors including internal ones (list only)
    json: bool = False,  # Output as JSON (list only)
):
    r"""
    Start an Actor-based service or list actors.

    This command starts actors based on the Pulsing Actor System or lists existing actors.
    Supported actor types:
    - router: Load balancing router (with OpenAI-compatible HTTP API)
    - transformers: Transformers-based inference worker
    - vllm: vLLM-based high-performance inference worker
    - list: List actors in the current system

    Args:
        actor_type: Actor type (positional argument). Options: 'router', 'transformers', 'vllm', 'list'
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
        endpoint: (list only) Single actor system endpoint (e.g., '127.0.0.1:8000')
        all_actors: (list only) Show all actors including internal system actors
        json: (list only) Output in JSON format

    Examples:
        # Start a router with OpenAI-compatible API on port 8080
        pulsing actor router --http_port 8080 --model_name my-llm

        # Start a vLLM worker
        pulsing actor vllm --model Qwen/Qwen2 --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000

        # Start a transformers worker
        pulsing actor transformers --model gpt2 --addr 0.0.0.0:8001 --seeds 192.168.1.100:8000

        # Start worker on CPU
        pulsing actor transformers --model gpt2 --device cpu

        # List actors from single endpoint
        pulsing actor list --endpoint 127.0.0.1:8000

        # List actors from cluster
        pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001

        # List all actors including internal ones
        pulsing actor list --endpoint 127.0.0.1:8000 --all_actors True

        # Output as JSON
        pulsing actor list --endpoint 127.0.0.1:8000 --json True
    """
    from .actors import start_router, start_transformers, start_vllm

    # Handle 'list' subcommand
    if actor_type == "list":
        from .actor_list import list_actors_command

        list_actors_command(
            endpoint=endpoint,
            seeds=seeds,
            all_actors=all_actors,
            json_output=json,
        )
        return

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
            f"Unknown actor type: {actor_type}. Supported types: router, transformers, vllm, list"
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
    model: str,  # Positional argument
    url: str = "http://localhost:8000",
    api_key: str = "",
    tokenizer: str | None = None,
    hf_token: str | None = None,
    max_vus: int = 128,
    duration: str = "120s",
    warmup: str = "30s",
    benchmark_kind: str = "throughput",
    num_rates: int = 10,
    rates: list | None = None,
    num_workers: int = 4,
):
    """
    Run inference benchmarks.

    This command runs the pulsing benchmark tool to test LLM inference endpoints.
    It uses an Actor-based architecture for real-time metrics and high performance.

    Args:
        model: The name of the model to benchmark (positional, required)
        url: Backend URL (default: "http://localhost:8000")
        api_key: API key for authentication (default: "")
        tokenizer: HuggingFace tokenizer name for accurate token counting (default: same as model)
        hf_token: HuggingFace token for private models (default: from HF_TOKEN env)
        max_vus: Maximum number of virtual users / concurrent requests (default: 128)
        duration: Duration of each benchmark phase (default: "120s")
        warmup: Warmup duration (default: "30s")
        benchmark_kind: Kind of benchmark (default: "throughput")
            - throughput: Max throughput test with constant VUs
            - sweep: Auto-discover max rate then sweep rates
            - csweep: Concurrency sweep (vary VUs)
            - rate: Test specific rates
        num_rates: Number of rate steps for sweep (default: 10)
        rates: Specific rates for rate benchmark (comma-separated)
        num_workers: Number of worker actors (default: 4)

    Examples:
        # Basic throughput test (uses model name for tokenizer)
        pulsing bench gpt2 --url http://localhost:8080

        # Use different tokenizer
        pulsing bench gpt-4 --tokenizer gpt2 --url http://api.openai.com

        # Concurrency sweep test
        pulsing bench gpt2 --benchmark_kind csweep --max_vus 64

        # Rate test with specific rates
        pulsing bench gpt2 --benchmark_kind rate --rates 1,5,10,20

        # Long duration test with more workers
        pulsing bench gpt2 --duration 300s --num_workers 8
    """
    from .bench import run_benchmark

    run_benchmark(
        model_name=model,
        url=url,
        api_key=api_key,
        tokenizer_name=tokenizer,
        hf_token=hf_token,
        max_vus=max_vus,
        duration=duration,
        warmup=warmup,
        benchmark_kind=benchmark_kind,
        num_rates=num_rates,
        rates=rates,
        num_workers=num_workers,
    )


def main():
    hp.launch()


if __name__ == "__main__":
    main()
