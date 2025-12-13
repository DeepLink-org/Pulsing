import asyncio
import os
import sys
from typing import Optional

import uvloop
from hyperparameter import auto_param, run_cli

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


@auto_param("frontend")
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
                overlap_score_weight=kv_overlap_score_weight,
                router_temperature=router_temperature,
                use_kv_events=use_kv_events,
                router_replica_sync=router_replica_sync,
                router_snapshot_threshold=router_snapshot_threshold or 1000000,
                router_reset_states=router_reset_states,
                router_track_active_blocks=router_track_active_blocks,
            )
        elif router_mode == "random":
            router_mode_enum = RouterMode.Random
            kv_router_config = None
        else:
            router_mode_enum = RouterMode.RoundRobin
            kv_router_config = None

        kwargs = {
            "http_host": http_host or os.environ.get("DYN_HTTP_HOST", "0.0.0.0"),
            "http_port": http_port or int(os.environ.get("DYN_HTTP_PORT", "8000")),
            "kv_cache_block_size": kv_cache_block_size,
            "router_config": DynamoRouterConfig(
                router_mode_enum,
                kv_router_config,
                busy_threshold,
                enforce_disagg,
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


@auto_param("vllm")
def vllm(model: str):
    """
    Start a vLLM backend worker.

    Args:
        model: Model path or HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B')
    """
    try:
        from .vllm_backend import start_vllm_worker
    except ImportError as e:
        raise ImportError(
            "vLLM backend requires vLLM dependencies. "
            "Please ensure vLLM and related packages are installed."
        ) from e

    print("Running vLLM backend worker...")
    start_vllm_worker()


@auto_param("transformers")
def transformers(model: str):
    """
    Start a Transformers backend worker.

    Args:
        model: Model path or HuggingFace model name (e.g., 'gpt2')
    """
    try:
        from .transformers_backend import start_transformers_worker
    except ImportError as e:
        raise ImportError(
            "Transformers backend requires 'transformers' and 'torch'. "
            "Please install them first."
        ) from e

    print("Running Transformers backend worker...")
    start_transformers_worker()


@auto_param("bench")
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


def main():
    run_cli()


if __name__ == "__main__":
    main()
