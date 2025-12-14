"""vLLM backend worker implementation with lazy loading."""

import os

import uvloop
from hyperparameter import auto_param, param_scope

from dynamo.llm import fetch_llm

from .runtime import create_runtime, setup_signal_handlers
from .vllm_init import (
    init_decode_worker,
    init_multimodal_encode_worker_worker,
    init_multimodal_processor_worker,
    init_multimodal_worker_worker,
    init_prefill_worker,
)


@auto_param("vllm.worker")
async def run_vllm_worker(
    model: str = None,
    component: str = "backend",
    endpoint: str = "generate",
    is_prefill_worker: bool = False,
    is_decode_worker: bool = False,
    multimodal_processor: bool = False,
    multimodal_encode_worker: bool = False,
    multimodal_worker: bool = False,
    multimodal_decode_worker: bool = False,
    multimodal_encode_prefill_worker: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    block_size: int = 16,
    enable_prefix_caching: bool = True,
    served_model_name: str = None,
    connector: list = None,
    migration_limit: int = 0,
    tool_call_parser: str = None,
    reasoning_parser: str = None,
    custom_jinja_template: str = None,
    mm_prompt_template: str = "USER: <image>\n<prompt> ASSISTANT:",
):
    """
    vLLM worker configuration.

    Args:
        model: Model path or HuggingFace model name
        component: Component name. Default: 'backend'
        endpoint: Endpoint name. Default: 'generate'
        is_prefill_worker: Enable prefill functionality
        is_decode_worker: Mark as decode worker (no KV events)
        multimodal_processor: Run as multimodal processor
        multimodal_encode_worker: Run as multimodal encode worker
        multimodal_worker: Run as multimodal worker
        multimodal_decode_worker: Run as multimodal decode worker
        multimodal_encode_prefill_worker: Run as encode+prefill+decode worker
        tensor_parallel_size: Number of TP replicas. Default: 1
        pipeline_parallel_size: Number of PP stages. Default: 1
        gpu_memory_utilization: GPU memory fraction. Default: 0.9
        max_model_len: Maximum sequence length
        block_size: KV cache block size. Default: 16
        enable_prefix_caching: Enable prefix caching. Default: True
        served_model_name: Name to serve as
        connector: Connectors list. Default: ['nixl']
        migration_limit: Max migration times. Default: 0
        tool_call_parser: Tool call parser name
        reasoning_parser: Reasoning parser name
        custom_jinja_template: Custom Jinja template path
        mm_prompt_template: Multi-modal prompt template
    """
    # Download the model if necessary
    if not os.path.exists(model):
        model = await fetch_llm(model)

    # Update model path in param_scope for init functions to use
    with param_scope(**{"vllm.model": model}):
        runtime = create_runtime()
        setup_signal_handlers(runtime)

        # Route to appropriate initialization based on flags
        if multimodal_processor:
            await init_multimodal_processor_worker(runtime)
        elif multimodal_encode_worker:
            await init_multimodal_encode_worker_worker(runtime)
        elif (
            multimodal_worker
            or multimodal_decode_worker
            or multimodal_encode_prefill_worker
        ):
            await init_multimodal_worker_worker(runtime)
        elif is_prefill_worker:
            await init_prefill_worker(runtime)
        else:
            await init_decode_worker(runtime)


def start_vllm_worker(**kwargs):
    """Entry point for starting vLLM worker with uvloop."""
    # Parameters are available in param_scope via @auto_param("vllm") in __main__.py
    uvloop.run(run_vllm_worker())
