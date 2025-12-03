from typing import Optional
import fire
import uvloop
import asyncio
import os
import signal

from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    KvRouterConfig,
    RouterConfig,
    RouterMode,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime


async def graceful_shutdown(runtime):
    runtime.shutdown()


class CLI:
    """
    Pulsing CLI - Distributed Inference Framework command-line interface.

    Available commands:
        frontend    Start a frontend node (router + HTTP server)
        vllm        Start a vLLM backend worker
        transformers Start a Transformers backend worker
        bench       Run benchmarks

    Examples:
        # Start frontend
        python -m pulsing.cli frontend --model-name "Llama-3.2-1B-Instruct" --model-path /tmp/model

        # Start vLLM backend
        python -m pulsing.cli vllm --model Qwen/Qwen3-0.6B

        # Start Transformers backend
        python -m pulsing.cli transformers --model gpt2

    Use 'python -m pulsing.cli <command> --help' for detailed help on each command.
    """
    def frontend(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        namespace: Optional[str] = None,
        request_plane: str = "http",
        store_kv: str = "file",
        router_mode: str = "kv",
        http_host: Optional[str] = None,
        http_port: Optional[int] = None,
        kv_cache_block_size: int = 16,
        busy_threshold: Optional[float] = None,
        enforce_disagg: bool = False,
        kv_overlap_score_weight: float = 1.0,
        router_temperature: float = 0.0,
        use_kv_events: bool = False,
        router_replica_sync: bool = False,
        router_snapshot_threshold: Optional[int] = None,
        router_reset_states: bool = False,
        router_track_active_blocks: bool = True,
        interactive: bool = False,
    ):
        """
        Start a frontend node.

        Args:
            model_name: Model name as a string (e.g., 'Llama-3.2-1B-Instruct')
            model_path: Path to model directory on disk (e.g., /tmp/model_cache/lama3.2_1B/)
            namespace: Namespace for service discovery. Can be set via DYN_NAMESPACE env var.
                      Default: 'dynamo'
            request_plane: Determines how requests are distributed from routers to workers.
                          Options: 'nats', 'http', 'tcp'. 'tcp' is fastest.
                          Default: 'http'
            store_kv: Which key-value backend to use. Options: 'etcd', 'mem', 'file'.
                     Etcd uses the ETCD_* env vars for connection details.
                     File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.
                     Default: 'file'
            router_mode: How to route the request. Options: 'round-robin', 'random', 'kv'.
                        Can be set via DYN_ROUTER_MODE env var.
                        Default: 'kv'
            http_host: HTTP host for the engine. Can be set via DYN_HTTP_HOST env var.
                      Default: '0.0.0.0' (from env or default)
            http_port: HTTP port for the engine. Can be set via DYN_HTTP_PORT env var.
                      Default: 8080 (from env or default)
            kv_cache_block_size: KV cache block size (u32). Can be set via DYN_KV_CACHE_BLOCK_SIZE env var.
                                Default: 16
            busy_threshold: Threshold (0.0-1.0) for determining when a worker is considered busy
                           based on KV cache usage. If not set, busy detection is disabled.
                           Default: None
            enforce_disagg: Enforce disaggregated prefill-decode. When set, unactivated prefill router
                           will return an error instead of falling back to decode-only mode.
                           Default: False
            kv_overlap_score_weight: KV Router: Weight for overlap score in worker selection.
                                    Higher values prioritize KV cache reuse.
                                    Default: 1.0
            router_temperature: KV Router: Temperature for worker sampling via softmax.
                              Higher values promote more randomness, and 0 fallbacks to deterministic.
                              Default: 0.0
            use_kv_events: KV Router: Enable KV events. When set, uses ApproxKvRouter for predicting
                          block creation/deletion based only on incoming requests at a timer.
                          By default, KV events are disabled.
                          Default: False
            router_replica_sync: KV Router: Enable replica synchronization across multiple router instances.
                               When true, routers will publish and subscribe to events to maintain consistent state.
                               Default: False
            router_snapshot_threshold: KV Router: Number of messages in stream before triggering a snapshot.
                                     If None, uses default value (1000000).
                                     Default: None
            router_reset_states: KV Router: Reset router state on startup, purging stream and object store.
                               By default, states are persisted. WARNING: This can affect existing router replicas.
                               Default: False
            router_track_active_blocks: KV Router: Enable tracking of active blocks (blocks being used
                                        for ongoing generation). By default, active blocks are tracked for load balancing.
                                        Default: True
            interactive: Interactive text chat mode instead of HTTP server.
                        Default: False
        """
        print("Running frontend...")

        async def run():
            loop = asyncio.get_running_loop()
            runtime = DistributedRuntime(loop, store_kv, request_plane)

            def signal_handler():
                asyncio.create_task(graceful_shutdown(runtime))

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)

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

            # Get namespace from parameter, env var, or default
            effective_namespace = namespace or os.environ.get("DYN_NAMESPACE", "dynamo")

            kwargs = {
                "http_host": http_host or os.environ.get("DYN_HTTP_HOST", "0.0.0.0"),
                "http_port": http_port or int(os.environ.get("DYN_HTTP_PORT", "8080")),
                "kv_cache_block_size": kv_cache_block_size,
                "router_config": RouterConfig(
                    router_mode_enum,
                    kv_router_config,
                    busy_threshold,
                    enforce_disagg,
                ),
                "namespace": effective_namespace,
            }

            if model_name:
                kwargs["model_name"] = model_name
            if model_path:
                kwargs["model_path"] = model_path

            e = EntrypointArgs(EngineType.Dynamic, **kwargs)
            engine = await make_engine(runtime, e)

            try:
                if interactive:
                    await run_input(runtime, "text", engine)
                else:
                    await run_input(runtime, "http", engine)
            except asyncio.exceptions.CancelledError:
                pass

        uvloop.run(run())

    def vllm(
        self,
        model: str,
        component: Optional[str] = None,
        endpoint: str = "generate",
        request_plane: str = "http",
        store_kv: str = "file",
        is_prefill_worker: bool = False,
        is_decode_worker: bool = False,
        migration_limit: int = 0,
        connector: Optional[list] = None,
        tool_call_parser: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
        custom_jinja_template: Optional[str] = None,
        multimodal_processor: bool = False,
        multimodal_encode_worker: bool = False,
        multimodal_worker: bool = False,
        multimodal_decode_worker: bool = False,
        multimodal_encode_prefill_worker: bool = False,
        mm_prompt_template: str = "USER: <image>\n<prompt> ASSISTANT:",
        served_model_name: Optional[str] = None,
        block_size: Optional[int] = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        **kwargs,
    ):
        """
        Start a vLLM backend worker.

        Args:
            model: Model path or HuggingFace model name (e.g., 'Qwen/Qwen3-0.6B')
            component: Component name. Auto-set based on worker type if not provided.
            endpoint: Endpoint name. Default: 'generate'
            request_plane: Determines how requests are distributed. Options: 'nats', 'http', 'tcp'.
                          Default: 'nats'
            store_kv: Key-value backend. Options: 'etcd', 'mem', 'file'. Default: 'etcd'
            is_prefill_worker: Enable prefill functionality for this worker.
            is_decode_worker: Mark this as a decode worker which does not publish KV events.
            migration_limit: Maximum number of times a request may be migrated to a different engine worker.
            connector: List of connectors to use (e.g., ['nixl', 'lmcache']). Options: nixl, lmcache, kvbm, null, none.
                      Default: ['nixl']
            tool_call_parser: Tool call parser name for the model.
            reasoning_parser: Reasoning parser name for the model.
            custom_jinja_template: Path to a custom Jinja template file to override the model's default chat template.
            multimodal_processor: Run as multimodal processor component for handling multimodal requests.
            multimodal_encode_worker: Run as multimodal encode worker component for processing images/videos.
            multimodal_worker: Run as multimodal worker component for LLM inference with multimodal data.
            multimodal_decode_worker: Run as multimodal decode worker in disaggregated mode.
            multimodal_encode_prefill_worker: Run as unified encode+prefill+decode worker for models requiring integrated image encoding.
            mm_prompt_template: Multi-modal prompt template. Default: 'USER: <image>\\n<prompt> ASSISTANT:'
            served_model_name: Name to serve the model as. If not set, uses model name.
            block_size: KV cache block size. Default: 16
            tensor_parallel_size: Number of tensor parallel replicas. Default: 1
            pipeline_parallel_size: Number of pipeline parallel stages. Default: 1
            max_model_len: Maximum sequence length that the model can accept.
            gpu_memory_utilization: Fraction of GPU memory to use. Default: 0.9
            enable_prefix_caching: Enable prefix caching. Default: True
            **kwargs: Additional vLLM AsyncEngineArgs parameters (e.g., max_num_seqs, max_num_batched_tokens, etc.)
        """
        # Lazy import to avoid loading vLLM dependencies until needed
        try:
            from .vllm_backend import start_vllm_worker
        except ImportError as e:
            raise ImportError(
                "vLLM backend requires vLLM dependencies. "
                "Please ensure vLLM and related packages are installed."
            ) from e

        print("Running vLLM backend worker...")
        start_vllm_worker(
            model=model,
            component=component,
            endpoint=endpoint,
            request_plane=request_plane,
            store_kv=store_kv,
            is_prefill_worker=is_prefill_worker,
            is_decode_worker=is_decode_worker,
            migration_limit=migration_limit,
            connector=connector,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            custom_jinja_template=custom_jinja_template,
            multimodal_processor=multimodal_processor,
            multimodal_encode_worker=multimodal_encode_worker,
            multimodal_worker=multimodal_worker,
            multimodal_decode_worker=multimodal_decode_worker,
            multimodal_encode_prefill_worker=multimodal_encode_prefill_worker,
            mm_prompt_template=mm_prompt_template,
            served_model_name=served_model_name,
            block_size=block_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            **kwargs,
        )

    def transformers(
        self,
        model: str,
        component: str = "backend",
        endpoint: str = "generate",
        namespace: Optional[str] = None,
        request_plane: str = "http",
        store_kv: str = "file",
        device: str = "cuda",
        max_new_tokens: int = 512,
        served_model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Start a Transformers backend worker.

        Args:
            model: Model path or HuggingFace model name (e.g., 'gpt2')
            component: Component name. Default: 'backend'
            endpoint: Endpoint name. Default: 'generate'
            namespace: Namespace for service discovery. Can be set via DYN_NAMESPACE env var.
                      Default: 'dynamo'
            request_plane: Determines how requests are distributed. Options: 'nats', 'http', 'tcp'.
                          Default: 'http'
            store_kv: Key-value backend. Options: 'etcd', 'mem', 'file'. Default: 'file'
            device: Device to run the model on. Options: 'cuda', 'cpu', 'mps'. Default: 'cuda'
            max_new_tokens: Maximum number of new tokens to generate. Default: 512
            served_model_name: Name to serve the model as. If not set, uses model name.
            **kwargs: Additional parameters passed to the worker
        """
        # Lazy import to avoid loading Transformers dependencies until needed
        try:
            from .transformers_backend import start_transformers_worker
        except ImportError as e:
            raise ImportError(
                "Transformers backend requires 'transformers' and 'torch'. "
                "Please install them first."
            ) from e

        print("Running Transformers backend worker...")
        start_transformers_worker(
            model=model,
            component=component,
            endpoint=endpoint,
            namespace=namespace,
            request_plane=request_plane,
            store_kv=store_kv,
            device=device,
            max_new_tokens=max_new_tokens,
            served_model_name=served_model_name,
            **kwargs,
        )

    def bench(self, *args, **kwargs):
        """
        Run benchmarks.

        This command acts as a wrapper around the dynamo benchmark tool.
        All arguments provided after 'bench' are passed directly to the benchmark runner.

        Examples:
            python -m pulsing.cli bench --help
        """
        import sys
        from dynamo._core import benchmark_main

        # Extract arguments after 'bench' to pass raw flags/args to the underlying tool
        # Fire consumes args/kwargs, so we look at sys.argv to preserve flags like --help
        cmd_args = []
        if "bench" in sys.argv:
            try:
                idx = sys.argv.index("bench")
                cmd_args = sys.argv[idx + 1 :]
            except ValueError:
                pass
        else:
            # Fallback if 'bench' is not strictly in argv (e.g. alias)
            # This might miss flags consumed by Fire
            cmd_args = list(args)

        benchmark_main(["pulsing-bench"] + cmd_args)

def main():
    fire.core.Display = lambda lines, out: out.write("\n".join(lines) + "\n")
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
