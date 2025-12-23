"""vLLM backend worker implementation with lazy loading."""

import os

import hyperparameter as hp
import uvloop

from dynamo.llm import fetch_llm

from ..cli.runtime import create_runtime, setup_signal_handlers
from .init import (
    init_decode_worker,
    init_multimodal_encode_worker_worker,
    init_multimodal_processor_worker,
    init_multimodal_worker_worker,
    init_prefill_worker,
)


@hp.param("vllm.worker")
async def run_vllm_worker(
    model: str = None,
    is_prefill_worker: bool = False,
    multimodal_processor: bool = False,
    multimodal_encode_worker: bool = False,
    multimodal_worker: bool = False,
    multimodal_decode_worker: bool = False,
    multimodal_encode_prefill_worker: bool = False,
):
    """
    vLLM worker entry point.

    Args:
        model: Model path or HuggingFace model name
        is_prefill_worker: Enable prefill functionality
        multimodal_processor: Run as multimodal processor
        multimodal_encode_worker: Run as multimodal encode worker
        multimodal_worker: Run as multimodal worker
        multimodal_decode_worker: Run as multimodal decode worker
        multimodal_encode_prefill_worker: Run as encode+prefill+decode worker

    Note:
        Other vLLM parameters (tensor_parallel_size, gpu_memory_utilization, etc.)
        are automatically read from param_scope by the init functions via @hp.param.
    """
    # Download the model if necessary
    if not os.path.exists(model):
        model = await fetch_llm(model)

    # Update model path in param_scope for init functions to use
    with hp.scope(**{"vllm.worker.model": model}):
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
    # Parameters are available in param_scope via @hp.param("vllm") in __main__.py
    uvloop.run(run_vllm_worker())
