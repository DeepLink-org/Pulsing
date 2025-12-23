import asyncio
import os
import sys
from typing import Optional

import uvloop
import hyperparameter as hp

from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    KvRouterConfig,
)
from dynamo.llm import RouterConfig as DynamoRouterConfig
from dynamo.llm import (
    RouterMode,
    make_engine,
    run_input,
)

from .runtime import create_runtime, setup_signal_handlers


def _to_bool(value) -> bool:
    """Convert value to bool, handling string 'true'/'false'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


@hp.param("frontend")
def frontend(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    namespace: Optional[str] = None,
    http_host: Optional[str] = None,
    http_port: Optional[int] = None,
    tls_cert_path: Optional[str] = None,
    tls_key_path: Optional[str] = None,
    router_mode: str = "kv",
    kv_cache_block_size: int = 16,
    busy_threshold: Optional[float] = None,
    enforce_disagg: bool = False,
    kv_overlap_score_weight: float = 1.0,
    router_temperature: float = 0.0,
    use_kv_events: bool = True,
    router_replica_sync: bool = False,
    router_snapshot_threshold: Optional[int] = None,
    router_reset_states: bool = False,
    router_track_active_blocks: bool = True,
    kserve_grpc_server: bool = False,
    interactive: bool = False,
):
    """
    Start a frontend node.

    Args:
        model_name: Model name as a string (e.g., 'Llama-3.2-1B-Instruct')
        model_path: Path to model directory on disk (e.g., /tmp/model_cache/llama3.2_1B/)
        namespace: Namespace for model discovery scoping. If not set, discovers from all namespaces.
        http_host: HTTP host for the engine. Default: '0.0.0.0'
        http_port: HTTP port for the engine. Default: 8000
        tls_cert_path: TLS certificate path (PEM format). Must be used with tls_key_path.
        tls_key_path: TLS certificate key path (PEM format). Must be used with tls_cert_path.
        router_mode: How to route the request. Options: 'round-robin', 'random', 'kv'. Default: 'kv'
        kv_cache_block_size: KV cache block size. Default: 16
        busy_threshold: Threshold (0.0-1.0) for worker busy detection based on KV cache usage.
        enforce_disagg: Enforce disaggregated prefill-decode mode.
        kv_overlap_score_weight: KV Router: Weight for overlap score. Default: 1.0
        router_temperature: KV Router: Temperature for worker sampling. Default: 0.0 (deterministic)
        use_kv_events: KV Router: Enable KV events. Default: True
        router_replica_sync: KV Router: Enable replica synchronization across routers.
        router_snapshot_threshold: KV Router: Messages before triggering snapshot.
        router_reset_states: KV Router: Reset state on startup. WARNING: affects existing routers.
        router_track_active_blocks: KV Router: Track active blocks for load balancing. Default: True
        kserve_grpc_server: Start KServe gRPC server instead of HTTP.
        interactive: Interactive text chat mode instead of HTTP server.
    """
    # Validate TLS args
    if bool(tls_cert_path) ^ bool(tls_key_path):
        raise ValueError("--tls_cert_path and --tls_key_path must be provided together")

    print("Running frontend...")

    async def run():
        # Create runtime (reads runtime.* params from param_scope)
        runtime = create_runtime()
        setup_signal_handlers(runtime)

        # Build router config
        if router_mode == "kv":
            router_mode_enum = RouterMode.KV
            kv_router_config = KvRouterConfig(
                overlap_score_weight=float(kv_overlap_score_weight),
                router_temperature=float(router_temperature),
                use_kv_events=_to_bool(use_kv_events),
                router_replica_sync=_to_bool(router_replica_sync),
                router_snapshot_threshold=int(router_snapshot_threshold) if router_snapshot_threshold else 1000000,
                router_reset_states=_to_bool(router_reset_states),
                router_track_active_blocks=_to_bool(router_track_active_blocks),
            )
        elif router_mode == "random":
            router_mode_enum = RouterMode.Random
            kv_router_config = None
        else:
            router_mode_enum = RouterMode.RoundRobin
            kv_router_config = None

        kwargs = {
            "http_host": http_host or os.environ.get("DYN_HTTP_HOST", "0.0.0.0"),
            "http_port": int(http_port) if http_port else int(os.environ.get("DYN_HTTP_PORT", "8000")),
            "kv_cache_block_size": int(kv_cache_block_size) if kv_cache_block_size else 16,
            "router_config": DynamoRouterConfig(
                router_mode_enum,
                kv_router_config,
                float(busy_threshold) if busy_threshold else None,
                _to_bool(enforce_disagg),
            ),
        }

        if model_name:
            kwargs["model_name"] = model_name
        if model_path:
            kwargs["model_path"] = model_path
        if namespace:
            kwargs["namespace"] = namespace
        if tls_cert_path:
            kwargs["tls_cert_path"] = tls_cert_path
        if tls_key_path:
            kwargs["tls_key_path"] = tls_key_path

        e = EntrypointArgs(EngineType.Dynamic, **kwargs)
        engine = await make_engine(runtime, e)

        try:
            if interactive:
                await run_input(runtime, "text", engine)
            elif kserve_grpc_server:
                await run_input(runtime, "grpc", engine)
            else:
                await run_input(runtime, "http", engine)
        except asyncio.exceptions.CancelledError:
            pass

    uvloop.run(run())


@hp.param("vllm")
def vllm(model: str):
    """
    Start a vLLM backend worker.

    Args:
        model: Model path or HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B')
    """
    from hyperparameter import param_scope
    
    try:
        from ..vllm import start_vllm_worker
    except ImportError as e:
        raise ImportError(
            "vLLM backend requires vLLM dependencies. "
            "Please ensure vLLM and related packages are installed."
        ) from e

    print("Running vLLM backend worker...")
    # Pass model to vllm.worker namespace
    with hp.scope(**{"vllm.worker.model": model}):
        start_vllm_worker()


@hp.param("transformers")
def transformers(model: str):
    """
    Start a Transformers backend worker.

    Args:
        model: Model path or HuggingFace model name (e.g., 'gpt2')
    """
    from hyperparameter import param_scope
    
    try:
        from .transformers_backend import start_transformers_worker
    except ImportError as e:
        raise ImportError(
            "Transformers backend requires 'transformers' and 'torch'. "
            "Please install them first."
        ) from e

    print("Running Transformers backend worker...")
    # Pass model to backend.transformers namespace (matches @hp.param in transformers_backend.py)
    with hp.scope(**{"backend.transformers.model": model}):
        start_transformers_worker()


@hp.param("router")
def router(endpoint: str, block_size: int = 128):
    """
    Start a standalone KV-aware router worker.

    This service provides a standalone KV-aware router for any set of workers
    in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing
    to prefill workers) or any other scenario requiring intelligent KV cache-aware
    routing decisions.

    Args:
        endpoint: Full endpoint path for workers in format namespace.component.endpoint
                  (e.g., 'dynamo.prefill.generate')
        block_size: KV cache block size for routing decisions. Default: 128
    """
    try:
        from ..router import start_router_worker
    except ImportError as e:
        raise ImportError(
            "Router requires Dynamo LLM dependencies. "
            "Please ensure dynamo.llm is installed."
        ) from e

    print("Running standalone router worker...")
    start_router_worker(endpoint=endpoint, block_size=block_size)


@hp.param("bench")
def bench():
    """
    Run benchmarks.

    This command wraps the dynamo benchmark tool.
    All arguments after 'bench' are passed to the benchmark runner.

    Examples:
        pulsing bench --help
    """
    from dynamo._core import benchmark_main

    cmd_args = []
    if "bench" in sys.argv:
        try:
            idx = sys.argv.index("bench")
            cmd_args = sys.argv[idx + 1 :]
        except ValueError:
            pass

    benchmark_main(["pulsing-bench"] + cmd_args)


@hp.param("actor")
def actor(
    type: str,
    namespace: str = "dynamo",
    addr: Optional[str] = None,
    seeds: Optional[str] = None,
    model: Optional[str] = None,
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
        namespace: Service namespace. Default: 'dynamo'
        addr: Actor System bind address (e.g., '0.0.0.0:8000')
        seeds: Comma-separated list of seed nodes (e.g., '192.168.1.1:8000,192.168.1.2:8000')
        model: Model path (required for 'transformers' and 'vllm' type)
        model_name: Model name for OpenAI API. Default: 'pulsing-model'
        device: Device for inference ('cuda', 'cpu', 'mps'). Default: 'cuda'
        max_new_tokens: Max tokens to generate. Default: 512
        role: Worker role for vLLM ('aggregated', 'prefill', 'decode'). Default: 'aggregated'
        scheduler: Scheduler algorithm for router. Options: 'round_robin', 'random', 'least_connection'. Default: 'round_robin'
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
        _start_router_actor(namespace, addr, seed_list, http_host, http_port, model_name, scheduler)
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
        raise ValueError(f"Unknown actor type: {type}. Supported types: router, transformers, vllm")


def _start_router_actor(
    namespace: str,
    addr: Optional[str],
    seeds: list,
    http_host: str,
    http_port: int,
    model_name: str,
    scheduler_type: str,
):
    """Start Router with OpenAI-compatible API"""
    from pulsing.actor import ActorSystem, SystemConfig
    from pulsing.actor.helpers import run_until_signal
    from ..actors.router import start_router, stop_router
    from ..actors import RoundRobinScheduler, RandomScheduler, LeastConnectionScheduler
    
    # 选择调度器类
    scheduler_map = {
        "round_robin": RoundRobinScheduler,
        "random": RandomScheduler,
        "least_connection": LeastConnectionScheduler,
    }
    scheduler_class = scheduler_map.get(scheduler_type)
    if not scheduler_class:
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Options: {list(scheduler_map.keys())}")

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
        
        system = await ActorSystem.create(config)
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
    addr: Optional[str],
    seeds: list,
    device: str,
    max_new_tokens: int,
    preload_model: bool,
):
    """Start Transformers Worker"""
    from pulsing.actor.helpers import spawn_and_run
    from ..actors import TransformersWorker, GenerationConfig

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
    addr: Optional[str],
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


def main():
    hp.launch()


if __name__ == "__main__":
    main()
