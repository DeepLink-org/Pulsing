"""
Standalone Router Handler

Handles routing requests to workers using KV-aware routing.
"""

import logging
from typing import Optional

from dynamo.llm import KvPushRouter, KvRouterConfig
from dynamo.runtime import Client, DistributedRuntime

logger = logging.getLogger(__name__)


class StandaloneRouterHandler:
    """Handles routing requests to workers using KV-aware routing."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        worker_endpoint_path: str,
        block_size: int,
        kv_router_config: KvRouterConfig,
    ):
        self.runtime = runtime
        self.worker_endpoint_path = worker_endpoint_path
        self.block_size = block_size
        self.kv_router_config = kv_router_config
        self.kv_push_router: Optional[KvPushRouter] = None
        self.worker_client: Optional[Client] = None

    async def initialize(self):
        """Initialize the KV router for workers."""
        try:
            # Parse endpoint path (format: namespace.component.endpoint)
            parts = self.worker_endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid endpoint path format: {self.worker_endpoint_path}. "
                    "Expected format: namespace.component.endpoint"
                )
            namespace, component, endpoint = parts

            # Get worker endpoint
            worker_endpoint = (
                self.runtime.namespace(namespace)
                .component(component)
                .endpoint(endpoint)
            )

            self.worker_client = await worker_endpoint.client()

            # Create KvPushRouter with specified configuration
            self.kv_push_router = KvPushRouter(
                endpoint=worker_endpoint,
                block_size=self.block_size,
                kv_router_config=self.kv_router_config,
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvPushRouter: {e}")
            raise

    async def generate(self, request):
        """
        Generate tokens using the KV-aware router.

        This endpoint routes the request to the best worker and streams back results.
        Wraps the request into PreprocessedRequest format and wraps worker responses
        into LLMEngineOutput format.
        """
        if self.kv_push_router is None:
            logger.error("KvPushRouter not initialized - cannot process request")
            raise RuntimeError("Router not initialized")

        # Wrap incoming request into PreprocessedRequest format for KvPushRouter
        preprocessed_request = {
            "model": request.get("model", "unknown"),
            "token_ids": request["token_ids"],
            "stop_conditions": request.get("stop_conditions", {}),
            "sampling_options": request.get("sampling_options", {}),
            "output_options": request.get("output_options", {}),
            "eos_token_ids": request.get("eos_token_ids", []),
            "annotations": request.get("annotations", []),
            "disaggregated_params": request.get("disaggregated_params"),
            "dp_rank": request.get("dp_rank"),
            "extra_args": request.get("extra_args", {}),
        }

        # Route and process through KvPushRouter
        async for worker_output in await self.kv_push_router.generate_from_request(
            preprocessed_request
        ):
            # Wrap worker output into LLMEngineOutput format
            llm_engine_output = {
                "token_ids": worker_output.get("token_ids", []),
                "tokens": worker_output.get("tokens"),
                "text": worker_output.get("text"),
                "cum_log_probs": worker_output.get("cum_log_probs"),
                "log_probs": worker_output.get("log_probs"),
                "top_logprobs": worker_output.get("top_logprobs"),
                "finish_reason": worker_output.get("finish_reason"),
                "index": worker_output.get("index"),
                "disaggregated_params": worker_output.get("disaggregated_params"),
                "extra_args": worker_output.get("extra_args"),
                "completion_usage": worker_output.get("completion_usage"),
            }
            yield llm_engine_output

    async def best_worker_id(self, token_ids, router_config_override=None):
        """
        Get the best worker ID for a given set of tokens without actually routing.

        This method returns the worker ID that would be selected based on KV cache
        overlap, but does NOT actually route the request or update router states.
        It's useful for debugging, monitoring, or implementing custom routing logic.
        """
        if self.kv_push_router is None:
            logger.error("KvPushRouter not initialized - cannot get best worker")
            raise RuntimeError("Router not initialized")

        result = await self.kv_push_router.best_worker_id(
            token_ids, router_config_override
        )

        yield result
