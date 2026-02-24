"""Pulsing CLI - Main entry point"""

import hyperparameter as hp


@hp.param("actor")
def actor(
    actor_type: str,  # Positional argument: full class path (e.g., 'pulsing.serving.worker.TransformersWorker')
    addr: str | None = None,
    seeds: str | None = None,
    name: str = "worker",  # Actor name (default: "worker")
    extra_kwargs: dict
    | None = None,  # Additional arguments for Actor constructor (--key value from CLI)
):
    r"""
    Start an Actor-based service.

    This command starts actors based on the Pulsing Actor System.

    Actor type must be a full class path:
    - Format: 'module.path.ClassName'
    - Example: 'pulsing.serving.Router'
    - Example: 'pulsing.serving.TransformersWorker'
    - Example: 'pulsing.serving.VllmWorker'
    - Example: 'my_module.my_actor.MyCustomActor'

    Parameter separation (avoids name collision):
    - Actor-level (process/cluster): --addr, --seeds, --name. Pass before \"--\".
    - Actor constructor args: pass after \"--\" so they never collide with --addr/--seeds/--name.

    Note: To list actors, use 'pulsing inspect actors' instead.

    Args:
        actor_type: Full class path (positional), e.g. pulsing.serving.Router
        addr: Bind address (e.g. 0.0.0.0:8000)
        seeds: Comma-separated seed nodes (e.g. 192.168.1.1:8000,192.168.1.2:8000)
        name: Actor name (default: worker). Use different names for multiple workers.
        extra_kwargs: Constructor arguments. Pass after \"--\" or via -D actor.extra_kwargs='{...}'.

    Examples:
        # Actor-level before \"--\", constructor args after (no collision)
        pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 --name my-llm -- --model_name my-llm --http_port 8080

        # Start a Transformers worker
        pulsing actor pulsing.serving.TransformersWorker --addr 0.0.0.0:8000 -- --model_name gpt2 --device cpu

        # Start a Router (constructor args after --)
        pulsing actor pulsing.serving.Router --addr 0.0.0.0:8000 --name my-llm -- --http_host 0.0.0.0 --http_port 8080 --model_name my-llm --worker_name worker

        # Multiple workers
        pulsing actor pulsing.serving.TransformersWorker --name worker-1 --seeds 127.0.0.1:8000 -- --model_name gpt2
        pulsing actor pulsing.serving.TransformersWorker --name worker-2 --seeds 127.0.0.1:8000 -- --model_name gpt2
    """
    from .actors import start_generic_actor

    # Check for deprecated 'list' subcommand
    if actor_type == "list":
        print("Error: 'pulsing actor list' has been removed.")
        print("Use 'pulsing inspect actors' instead:")
        print("  pulsing inspect actors --endpoint 127.0.0.1:8000")
        print("  pulsing inspect actors --seeds 127.0.0.1:8000")
        return

    # Check if actor_type is a valid class path (must contain dots)
    if "." not in actor_type:
        raise ValueError(
            f"Error: Actor type must be a full class path (e.g., 'pulsing.serving.worker.TransformersWorker').\n"
            f"Received: '{actor_type}'\n"
            f"Example: pulsing actor pulsing.serving.worker.TransformersWorker --model_name gpt2"
        )

    # Parse seeds
    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]

    # extra_kwargs may be str when passed via -D actor.extra_kwargs='{...}'
    import json

    kwargs = extra_kwargs or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs) if kwargs.strip() else {}
    # Start generic Actor class
    start_generic_actor(
        actor_type=actor_type,
        addr=addr,
        seeds=seed_list,
        name=name,
        extra_kwargs=kwargs,
    )


@hp.param("inspect")
def inspect(
    subcommand: str,  # Positional argument: cluster, actors, metrics, watch
    seeds: str | None = None,
    # Common options
    timeout: float = 10.0,
    best_effort: bool = False,
    # cluster subcommand options (no specific options yet)
    # actors subcommand options
    top: int | None = None,
    filter: str | None = None,
    all_actors: bool = False,
    endpoint: str | None = None,  # Single node endpoint (alternative to seeds)
    json: bool = False,  # JSON output format
    detailed: bool = False,  # Show detailed info (class, module) per node
    # metrics subcommand options
    raw: bool = True,
    # watch subcommand options
    interval: float = 1.0,
    kind: str = "all",  # cluster, actors, metrics, all
    max_rounds: int | None = None,
):
    """
    Inspect the actor system state (observer mode via HTTP API).

    This command uses lightweight HTTP observer mode - does NOT join the gossip cluster.

    Args:
        subcommand: Subcommand to run. Options: 'cluster', 'actors', 'metrics', 'watch' (required)
        seeds: Comma-separated list of seed nodes (required)
        timeout: Request timeout in seconds (default: 10.0)
        best_effort: Continue even if some nodes fail (default: False)
        top: (actors only) Show top N actors by instance count
        filter: (actors only) Filter actor names by substring
        all_actors: (actors only) Include internal/system actors
        raw: (metrics only) Output raw metrics (default: True). If False, show summary only
        interval: (watch only) Refresh interval in seconds (default: 1.0)
        kind: (watch only) What to watch: 'cluster', 'actors', 'metrics', 'all' (default: 'all')
        max_rounds: (watch only) Maximum number of refresh rounds (None = infinite)

    Examples:
        # Inspect cluster members
        pulsing inspect cluster --seeds 127.0.0.1:8000

        # Inspect actors distribution
        pulsing inspect actors --seeds 127.0.0.1:8000 --top 10

        # Inspect metrics
        pulsing inspect metrics --seeds 127.0.0.1:8000 --raw False

        # Watch cluster changes
        pulsing inspect watch --seeds 127.0.0.1:8000 --interval 2.0 --kind cluster
    """
    from .inspect import (
        inspect_actors,
        inspect_cluster,
        inspect_metrics,
        inspect_watch,
    )

    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]

    if subcommand == "cluster":
        if not seed_list:
            print("Error: --seeds is required for 'inspect cluster' command")
            return
        inspect_cluster(seed_list, timeout=timeout, best_effort=best_effort)
    elif subcommand == "actors":
        if endpoint and seed_list:
            print("Error: Cannot specify both --endpoint and --seeds.")
            print("Use --endpoint for single node, --seeds for cluster.")
            return
        inspect_actors(
            seeds=seed_list if not endpoint else None,
            endpoint=endpoint,
            timeout=timeout,
            best_effort=best_effort,
            top=top,
            filter=filter,
            all_actors=all_actors,
            json_output=json,
            detailed=detailed,
        )
    elif subcommand == "metrics":
        inspect_metrics(seed_list, timeout=timeout, best_effort=best_effort, raw=raw)
    elif subcommand == "watch":
        inspect_watch(
            seed_list,
            timeout=timeout,
            best_effort=best_effort,
            interval=interval,
            kind=kind,
            max_rounds=max_rounds,
        )
    else:
        print(f"Error: Unknown subcommand '{subcommand}'")
        print("Supported subcommands: cluster, actors, metrics, watch")


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


@hp.param("examples")
def examples(name: str | None = None):
    """
    List or view Pulsing built-in examples.

    Lists all available examples when called without arguments;
    shows detailed description, usage, and source path when given a name.

    Args:
        name: Example name (optional). If omitted, lists all examples.

    Examples:
        # List all examples
        pulsing examples

        # View details of a specific example
        pulsing examples counting_game
    """
    from pulsing.examples import get_example_detail, list_examples

    if name is None:
        all_examples = list_examples()
        if not all_examples:
            print("No examples available.")
            return
        print("Available examples:\n")
        max_name_len = max(len(n) for n, _, _ in all_examples)
        for n, summary, filepath in all_examples:
            print(f"  {n:<{max_name_len}}  {summary}")
        print("\nUse 'pulsing examples <name>' for details.")
        return

    detail = get_example_detail(name)
    if detail is None:
        print(f"Unknown example: '{name}'")
        print("Use 'pulsing examples' to see all available examples.")
        return

    summary, docstring, filepath = detail
    print(f"{'=' * 60}")
    print(f"  {summary}")
    print(f"{'=' * 60}\n")
    if docstring:
        print(docstring)
        print()
    print(f"Source path:\n  {filepath}\n")
    print(f"Quick run:\n  python -m pulsing.examples.{name}")


def _collect_key_value_pairs(tokens: list[str]) -> dict:
    """Collect --key value pairs from token list into a dict. Keys normalized to snake_case."""
    extra = {}
    i = 0
    while i < len(tokens):
        a = tokens[i]
        if a.startswith("--") and not a.startswith("---") and len(a) > 2:
            key = a[2:].replace("-", "_")
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                extra[key] = tokens[i + 1]
                i += 2
                continue
        i += 1
    return extra


def _actor_argv_rewrite(argv: list[str]) -> list[str]:
    """Pre-parse 'pulsing actor ...' argv: split on \"--\" only.

    - Before \"--\": entire token list is passed through to the actor subcommand (no name-based parsing).
    - After \"--\": collected as --key value and injected as actor.extra_kwargs (constructor args).
    - If there is no \"--\", argv is left unchanged.
    """
    import json

    if len(argv) < 2 or argv[1] != "actor":
        return argv
    rest = argv[2:]
    if "--" not in rest:
        return argv
    dash_idx = rest.index("--")
    before, after = rest[:dash_idx], rest[dash_idx + 1 :]
    extra = _collect_key_value_pairs(after)
    if not extra:
        return argv
    return (
        [argv[0], "actor"] + before + ["-D", f"actor.extra_kwargs={json.dumps(extra)}"]
    )


def main():
    import sys

    # Make `pulsing examples <name>` work with positional arguments
    if (
        len(sys.argv) >= 3
        and sys.argv[1] == "examples"
        and not sys.argv[2].startswith("-")
    ):
        sys.argv = [sys.argv[0], "examples", "--name", sys.argv[2]] + sys.argv[3:]

    # Pre-parse 'actor' so --model_name my-llm etc. become actor.extra_kwargs (avoids required kwargs positional)
    sys.argv = _actor_argv_rewrite(sys.argv)

    hp.launch()


if __name__ == "__main__":
    main()
