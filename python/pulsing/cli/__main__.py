import hyperparameter as hp
import uvloop


@hp.param("actor")
def actor(
    type: str,
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
    scheduler: str = "random",
):
    """
    Start an Actor-based service.

    This command starts actors based on the Pulsing Actor System.
    Supported actor types:
    - router: RoundRobin load balancing router (with OpenAI-compatible HTTP API)
    - transformers: Transformers-based inference worker
    - vllm: vLLM-based high-performance inference worker

    Args:
        type: Actor type. Options: 'router', 'transformers', 'vllm'
        namespace: Service namespace. Default: 'pulsing'
        addr: Actor System bind address (e.g., '0.0.0.0:8000')
        seeds: Comma-separated list of seed nodes (e.g., '192.168.1.1:8000,192.168.1.2:8000')
        model: Model path (required for 'transformers' and 'vllm' type)
        model_name: Model name for OpenAI API. Default: 'pulsing-model'
        device: Device for inference ('cuda', 'cpu', 'mps'). Default: 'cuda'
        max_new_tokens: Max tokens to generate. Default: 512
        role: Worker role for vLLM ('aggregated', 'prefill', 'decode'). Default: 'aggregated'
        scheduler: Scheduler algorithm for router. Options: 'round_robin', 'random', 'least_connection'. Default: 'random'
        preload_model: Preload model on startup. Default: False
        http_host: HTTP server host (for router). Default: '0.0.0.0'
        http_port: HTTP server port (for router). Default: 8080

    Examples:
        # Start a router with OpenAI-compatible API on port 8080
        pulsing actor --type router --http_port 8080 --model_name my-llm

        # Test with curl
        curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"my-llm","messages":[{"role":"user","content":"Hello"}]}'

        # Start a vLLM worker
        pulsing actor --type vllm --model Qwen/Qwen2.5-0.5B --addr 0.0.0.0:8001 --seeds 127.0.0.1:8000

        # Start a transformers worker
        pulsing actor --type transformers --model Qwen/Qwen3-0.6B --addr 0.0.0.0:8001 --seeds 192.168.1.100:8000

        # Start worker on CPU
        pulsing actor --type transformers --model gpt2 --device cpu
    """
    # Parse seeds
    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]

    if type == "router":
        _start_router_actor(
            namespace, addr, seed_list, http_host, http_port, model_name, scheduler
        )
    elif type == "transformers":
        if not model:
            raise ValueError("--model is required for 'transformers' actor type")
        _start_transformers_actor(
            model=model,
            namespace=namespace,
            addr=addr,
            seeds=seed_list,
            device=device,
            max_new_tokens=max_new_tokens,
            preload_model=preload_model,
        )
    elif type == "vllm":
        if not model:
            raise ValueError("--model is required for 'vllm' actor type")
        _start_vllm_actor(
            model=model,
            namespace=namespace,
            addr=addr,
            seeds=seed_list,
            max_new_tokens=max_new_tokens,
            role=role,
        )
    else:
        raise ValueError(
            f"Unknown actor type: {type}. Supported types: router, transformers, vllm"
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
    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
    _inspect_system(seed_list)


def _inspect_system(seeds: list):
    """Inspect the actor system state"""
    import asyncio

    from pulsing.actor import SystemConfig, create_actor_system

    async def run():
        if not seeds:
            print("Error: --seeds is required for 'inspect' command")
            return

        print(f"Connecting to cluster via seeds: {seeds}...")
        # If seeds are local, bind to 127.0.0.1 to ensure connectivity
        if any(s.startswith("127.0.0.1") or s.startswith("localhost") for s in seeds):
            config = SystemConfig.with_addr("127.0.0.1:0").with_seeds(seeds)
        else:
            config = SystemConfig.standalone().with_seeds(seeds)
        system = await create_actor_system(config)

        # Give some time for discovery
        await asyncio.sleep(1.5)

        members = await system.members()
        print(f"\nCluster Status: {len(members)} nodes found")
        print("=" * 60)

        # Get all named actors automatically
        all_named_actors = {}
        try:
            for info in await system.all_named_actors():
                path = str(info.get("path", ""))
                name = path[7:] if path.startswith("actors/") else path
                if info.get("instance_count", 0) > 0:
                    try:
                        instances = await system.get_named_instances(name)
                        if instances:
                            all_named_actors[name] = instances
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [Warning] Failed to get all named actors: {e}")

        # Group named actors by node
        node_actors = {}
        for name, instances in all_named_actors.items():
            for inst in instances:
                node_actors.setdefault(str(inst.get("node_id")), []).append(name)

        # Display nodes and their actors
        for member in members:
            node_id = str(member.get("node_id"))
            print(f"\nNode: {node_id} ({member.get('addr')}) [{member.get('status')}]")

            if member.get("status") != "Alive":
                print("  [Node is not alive]")
                continue

            actors = node_actors.get(node_id, [])
            if not actors:
                print("  [No named actors on this node]")
                continue

            # Group by base type
            actor_groups = {}
            for name in actors:
                base = name.rsplit("_", 1)[0] if "_" in name else name
                actor_groups.setdefault(base, []).append(name)

            print(f"  Named Actors ({len(actors)}):")
            for base, names in sorted(actor_groups.items()):
                if len(names) == 1:
                    print(f"    - actors/{names[0]}")
                else:
                    print(f"    - actors/{base}_* ({len(names)} instances)")
                    for name in sorted(names)[:5]:
                        print(f"        • {name}")
                    if len(names) > 5:
                        print(f"        ... and {len(names) - 5} more")

        # Summary
        if all_named_actors:
            total = sum(len(instances) for instances in all_named_actors.values())
            print(
                f"\nTotal Named Actors: {len(all_named_actors)} types, {total} instances"
            )

        print("\n" + "=" * 60)
        await system.shutdown()

    uvloop.run(run())


def _start_router_actor(
    namespace: str,
    addr: str | None,
    seeds: list,
    http_host: str,
    http_port: int,
    model_name: str,
    scheduler_type: str,
):
    """Start Router with OpenAI-compatible API"""
    from pulsing.actor import SystemConfig, create_actor_system
    from pulsing.actor.helpers import run_until_signal

    from ..actors import LeastConnectionScheduler, RandomScheduler, RoundRobinScheduler
    from ..actors.router import start_router, stop_router

    # 选择调度器类
    scheduler_map = {
        "round_robin": RoundRobinScheduler,
        "random": RandomScheduler,
        "least_connection": LeastConnectionScheduler,
    }
    scheduler_class = scheduler_map.get(scheduler_type)
    if not scheduler_class:
        raise ValueError(
            f"Unknown scheduler: {scheduler_type}. Options: {list(scheduler_map.keys())}"
        )

    print(f"Starting Router (namespace={namespace}, model={model_name})")
    print(f"  Actor System addr: {addr or 'auto'}")
    print(f"  HTTP API: http://{http_host}:{http_port}")
    print(f"  Scheduler: {scheduler_type}")

    async def run():
        # 1. 创建 ActorSystem
        if addr:
            config = SystemConfig.with_addr(addr)
        else:
            config = SystemConfig.standalone()

        if seeds:
            config = config.with_seeds(seeds)

        system = await create_actor_system(config)
        print(f"[Router] ActorSystem started at {system.addr}")

        # 2. 启动 Router HTTP 服务器
        runner = await start_router(
            system,
            http_host=http_host,
            http_port=http_port,
            model_name=model_name,
            scheduler_class=scheduler_class,
        )

        # 3. 运行直到收到信号
        try:
            await run_until_signal(system, "router")
        finally:
            await stop_router(runner)

    uvloop.run(run())


def _start_transformers_actor(
    model: str,
    namespace: str,
    addr: str | None,
    seeds: list,
    device: str,
    max_new_tokens: int,
    preload_model: bool,
):
    """Start Transformers Worker"""
    from pulsing.actor.helpers import spawn_and_run

    from ..actors import GenerationConfig, TransformersWorker

    print(f"Starting Transformers Worker (model={model}, namespace={namespace})")
    print(f"  Device: {device}")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Preload: {preload_model}")

    async def run():
        # 创建 Worker Actor
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens)
        worker = TransformersWorker(
            model_name=model,
            device=device,
            gen_config=gen_config,
            preload=preload_model,
        )

        # spawn 并运行
        await spawn_and_run(
            worker,
            name="worker",
            addr=addr,
            seeds=seeds if seeds else None,
            public=True,
        )

    uvloop.run(run())


def _start_vllm_actor(
    model: str,
    namespace: str,
    addr: str | None,
    seeds: list,
    max_new_tokens: int,
    role: str = "aggregated",
):
    """Start vLLM Worker"""
    from pulsing.actor.helpers import spawn_and_run

    from ..actors import VllmWorker

    print(f"Starting vLLM Worker (model={model}, namespace={namespace}, role={role})")
    print(f"  Max tokens: {max_new_tokens}")

    async def run():
        # 创建 Worker Actor
        worker = VllmWorker(
            model=model,
            role=role,
            max_new_tokens=max_new_tokens,
        )

        # spawn 并运行
        await spawn_and_run(
            worker,
            name="worker",
            addr=addr,
            seeds=seeds if seeds else None,
            public=True,
        )

    uvloop.run(run())


@hp.param("bench")
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
    from pulsing._core import benchmark_main

    config = {
        "tokenizer_name": tokenizer_name,
        "max_vus": max_vus,
        "duration": duration,
        "num_rates": num_rates,
        "benchmark_kind": benchmark_kind,
        "warmup": warmup,
        "url": url,
        "api_key": api_key,
        "dataset": dataset,
        "dataset_file": dataset_file,
    }

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

    import uvloop

    uvloop.run(benchmark_main(config))


def main():
    hp.launch()


if __name__ == "__main__":
    main()
