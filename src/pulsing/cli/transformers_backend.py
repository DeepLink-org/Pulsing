"""Transformers backend worker implementation."""

import asyncio
import os
from typing import AsyncGenerator, Optional

import hyperparameter as hp
import uvloop

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    fetch_llm,
    register_llm,
)

from .runtime import create_runtime, setup_signal_handlers


@hp.param("backend.transformers")
async def run_transformers_worker(
    model: str = None,
    component: str = "backend",
    endpoint: str = "generate",
    device: str = "cuda",
    max_new_tokens: int = 512,
    served_model_name: Optional[str] = None,
    block_size: int = 16,
    **kwargs,
):
    """
    Transformers worker configuration.

    Args:
        model: Model name or path (e.g., "gpt2" or "/path/to/model")
        component: Component name. Default: 'backend'
        endpoint: Endpoint name. Default: 'generate'
        device: Device to run on ('cuda', 'cpu', 'mps'). Default: 'cuda'
        max_new_tokens: Max tokens to generate. Default: 512
        served_model_name: Name to serve as. Default: model name
        block_size: KV cache block size. Default: 16
    """
    # Lazy import heavy dependencies
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Transformers backend requires 'transformers' and 'torch'. "
            "Please install them first."
        ) from e

    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    served_model_name = served_model_name or model

    # Create runtime (reads runtime.* from param_scope)
    runtime = create_runtime()
    setup_signal_handlers(runtime)

    # Download/resolve model path
    if not os.path.exists(model):
        model_path = await fetch_llm(model)
    else:
        model_path = model

    print(f"Loading model {model} from {model_path} on {device}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model_kwargs = {"device_map": "auto"} if device == "cuda" else {}

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, **model_kwargs
    )

    if device != "cuda":
        hf_model.to(device)

    hf_model.eval()
    print("Model loaded successfully.")

    # Create component and service
    comp = runtime.namespace(namespace).component(component)
    try:
        await comp.create_service()
    except Exception as e:
        print(f"Warning: create_service failed: {e}")

    generate_endpoint = comp.endpoint(endpoint)

    # Get event loop for executor calls
    loop = asyncio.get_running_loop()

    # Request handler
    async def generate_handler(request, context) -> AsyncGenerator[dict, None]:
        request_id = context.id() if hasattr(context, "id") else "unknown"
        print(f"Processing request ID: {request_id}")

        try:
            # Extract prompt
            prompt = request.get("prompt", "")
            token_ids = request.get("token_ids", [])

            if not prompt and not token_ids and "messages" in request:
                msgs = request["messages"]
                if isinstance(msgs, list) and len(msgs) > 0:
                    prompt = msgs[-1].get("content", "")

            # Prepare inputs
            if token_ids:
                inputs = {
                    "input_ids": torch.tensor([token_ids], device=hf_model.device)
                }
            elif prompt:
                inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
            else:
                yield {
                    "finish_reason": "error",
                    "message": "No prompt or token_ids provided",
                }
                return

            # Generate in executor to avoid blocking
            outputs = await loop.run_in_executor(
                None,
                lambda: hf_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                ),
            )

            # Decode and stream tokens
            full_ids = outputs[0]
            input_len = inputs["input_ids"].shape[1]
            new_tokens = full_ids[input_len:]

            for i in range(len(new_tokens)):
                token_id = new_tokens[i].item()
                yield {
                    "token_ids": [token_id],
                    "text": tokenizer.decode([token_id]),
                    "finish_reason": None,
                }

            yield {
                "token_ids": [],
                "finish_reason": "stop",
                "completion_usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": len(new_tokens),
                    "total_tokens": len(full_ids),
                },
            }

        except Exception as e:
            print(f"Error during generation: {e}")
            yield {"finish_reason": "error", "message": str(e)}

    # Register with Dynamo
    runtime_config = ModelRuntimeConfig()
    runtime_config.total_kv_blocks = 1000
    runtime_config.max_num_seqs = 1
    runtime_config.max_num_batched_tokens = 2048
    runtime_config.data_parallel_size = 1

    await register_llm(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        generate_endpoint,
        model,
        served_model_name,
        kv_cache_block_size=block_size,
        migration_limit=0,
        runtime_config=runtime_config,
    )

    print(f"Worker {component} listening on {namespace}.{component}.{endpoint}")
    print(f"Registered model: {served_model_name}")
    print("Waiting for requests...")

    # Serve
    try:
        await generate_endpoint.serve_endpoint(
            generate_handler, metrics_labels=[("model", served_model_name)]
        )
    except KeyboardInterrupt:
        print("Received interrupt signal, shutting down...")
    except Exception as e:
        print(f"Failed to serve endpoint: {e}")
        raise
    finally:
        print("Worker cleanup complete")


def start_transformers_worker():
    """Entry point for starting transformers worker with uvloop."""
    # Parameters are available in param_scope via @hp.param("transformers")
    uvloop.run(run_transformers_worker())
