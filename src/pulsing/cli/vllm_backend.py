"""vLLM backend worker implementation with lazy loading."""

from typing import Optional, List
import argparse
import asyncio
import os
import signal

import uvloop

from dynamo.llm import fetch_llm
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.args import Config, overwrite_args
from dynamo.vllm.main import (
    init,
    init_prefill,
    init_multimodal_processor,
    init_multimodal_encode_worker,
    init_multimodal_worker,
)
from vllm.engine.arg_utils import AsyncEngineArgs


async def graceful_shutdown(runtime):
    """Shutdown dynamo distributed runtime."""
    runtime.shutdown()


async def run_vllm_worker(
    model: str,
    namespace: Optional[str] = None,
    component: Optional[str] = None,
    endpoint: str = "generate",
    request_plane: str = "nats",
    store_kv: str = "etcd",
    is_prefill_worker: bool = False,
    is_decode_worker: bool = False,
    migration_limit: int = 0,
    connector: Optional[List[str]] = None,
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
    Run a vLLM backend worker.

    This function is called lazily to avoid importing vLLM dependencies
    until actually needed.
    """
    # Get namespace from env or default
    namespace = namespace or os.environ.get("DYN_NAMESPACE", "dynamo")

    # Build AsyncEngineArgs
    # Create an argparse.Namespace with all the engine args
    engine_args_dict = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching,
        **kwargs,
    }

    if block_size is not None:
        engine_args_dict["block_size"] = block_size
    if max_model_len is not None:
        engine_args_dict["max_model_len"] = max_model_len
    if served_model_name:
        engine_args_dict["served_model_name"] = [served_model_name]

    # Create argparse.Namespace for AsyncEngineArgs
    engine_args_namespace = argparse.Namespace(**engine_args_dict)
    engine_args = AsyncEngineArgs.from_cli_args(engine_args_namespace)

    # Create Config object
    config = Config()
    config.model = model
    config.served_model_name = served_model_name
    config.namespace = namespace
    config.store_kv = store_kv
    config.request_plane = request_plane
    config.is_prefill_worker = is_prefill_worker
    config.is_decode_worker = is_decode_worker
    config.migration_limit = migration_limit
    config.tool_call_parser = tool_call_parser
    config.reasoning_parser = reasoning_parser
    config.custom_jinja_template = custom_jinja_template
    config.multimodal_processor = multimodal_processor
    config.multimodal_encode_worker = multimodal_encode_worker
    config.multimodal_worker = multimodal_worker
    config.multimodal_decode_worker = multimodal_decode_worker
    config.multimodal_encode_prefill_worker = multimodal_encode_prefill_worker
    config.mm_prompt_template = mm_prompt_template
    config.engine_args = engine_args

    # Set connector list
    if connector is None:
        connector = ["nixl"]
    config.connector_list = [c.lower() for c in connector]

    # Set component and endpoint based on worker type
    if component:
        config.component = component
        config.endpoint = endpoint
    elif multimodal_processor:
        config.component = "processor"
        config.endpoint = "generate"
    elif multimodal_encode_worker:
        config.component = "encoder"
        config.endpoint = "generate"
    elif multimodal_encode_prefill_worker:
        config.component = "encoder"
        config.endpoint = "generate"
    elif multimodal_decode_worker:
        config.component = "decoder"
        config.endpoint = "generate"
    elif multimodal_worker and is_prefill_worker:
        config.component = "backend"
        config.endpoint = "generate"
    elif is_prefill_worker:
        config.component = "prefill"
        config.endpoint = "generate"
    else:
        config.component = "backend"
        config.endpoint = "generate"

    # Set default block_size if not provided
    if config.engine_args.block_size is None:
        config.engine_args.block_size = 16

    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, config.store_kv, config.request_plane)

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    overwrite_args(config)

    # Download the model if necessary
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model
    if not os.path.exists(config.model):
        config.model = config.engine_args.model = await fetch_llm(config.model)

    # Route to appropriate initialization based on config flags
    if config.multimodal_processor:
        await init_multimodal_processor(runtime, config)
    elif config.multimodal_encode_worker:
        await init_multimodal_encode_worker(runtime, config)
    elif (
        config.multimodal_worker
        or config.multimodal_decode_worker
        or config.multimodal_encode_prefill_worker
    ):
        await init_multimodal_worker(runtime, config)
    elif config.is_prefill_worker:
        await init_prefill(runtime, config)
    else:
        await init(runtime, config)


def start_vllm_worker(**kwargs):
    """Entry point for starting vLLM worker with uvloop."""
    uvloop.run(run_vllm_worker(**kwargs))

