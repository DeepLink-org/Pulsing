"""
Standalone Router Worker

Provides the worker function for the standalone KV-aware router service.
"""

import asyncio
import logging

import uvloop
import hyperparameter as hp

from dynamo.llm import KvRouterConfig

from ..cli.runtime import create_runtime, setup_signal_handlers
from .handler import StandaloneRouterHandler

logger = logging.getLogger(__name__)


def _to_bool(value) -> bool:
    """Convert value to bool, handling string 'true'/'false'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


@hp.param("router.kv")
def _build_kv_router_config(
    overlap_score_weight: float = 1.0,
    temperature: float = 0.0,
    use_kv_events: bool = True,
    replica_sync: bool = False,
    snapshot_threshold: int = 1000000,
    reset_states: bool = False,
    track_active_blocks: bool = True,
) -> KvRouterConfig:
    """
    Build KvRouterConfig from hyperparameter settings.

    Args:
        overlap_score_weight: Weight for overlap score in worker selection.
                              Higher values prioritize KV cache reuse. Default: 1.0
        temperature: Temperature for worker sampling via softmax.
                     Higher values promote randomness, 0 is deterministic. Default: 0.0
        use_kv_events: Enable KV events. When False, uses ApproxKvRouter. Default: True
        replica_sync: Enable replica synchronization across router instances. Default: False
        snapshot_threshold: Messages in stream before triggering a snapshot. Default: 1000000
        reset_states: Reset router state on startup. WARNING: affects existing routers. Default: False
        track_active_blocks: Track active blocks for load balancing. Default: True
    """
    return KvRouterConfig(
        overlap_score_weight=float(overlap_score_weight),
        router_temperature=float(temperature),
        use_kv_events=_to_bool(use_kv_events),
        router_replica_sync=_to_bool(replica_sync),
        router_snapshot_threshold=int(snapshot_threshold),
        router_reset_states=_to_bool(reset_states),
        router_track_active_blocks=_to_bool(track_active_blocks),
    )


def start_router_worker(endpoint: str, block_size: int = 128):
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

    Advanced KV router options can be configured via hyperparameter:
        - router.kv.overlap_score_weight: Weight for overlap score (default: 1.0)
        - router.kv.temperature: Sampling temperature (default: 0.0)
        - router.kv.use_kv_events: Enable KV events (default: True)
        - router.kv.replica_sync: Enable replica sync (default: False)
        - router.kv.snapshot_threshold: Snapshot threshold (default: 1000000)
        - router.kv.reset_states: Reset states on startup (default: False)
        - router.kv.track_active_blocks: Track active blocks (default: True)
    """
    # Build KV router config from hyperparameter settings
    kv_router_config = _build_kv_router_config()

    async def _run_worker():
        # Create runtime (reads runtime.* params from param_scope)
        runtime = create_runtime()
        setup_signal_handlers(runtime)

        # Parse endpoint path to get namespace for service registration
        endpoint_parts = endpoint.split(".")
        if len(endpoint_parts) != 3:
            raise ValueError(
                f"Invalid endpoint path format: {endpoint}. "
                "Expected format: namespace.component.endpoint"
            )
        namespace = endpoint_parts[0]

        logger.info("Starting Standalone Router Service")
        logger.debug(
            f"Configuration: endpoint={endpoint}, block_size={block_size}, "
            f"kv_router_config={kv_router_config}"
        )

        # Create service component - use "router" as component name
        component = runtime.namespace(namespace).component("router")
        await component.create_service()

        # Create handler
        handler = StandaloneRouterHandler(
            runtime, endpoint, int(block_size), kv_router_config
        )
        await handler.initialize()

        # Expose endpoints
        generate_endpoint = component.endpoint("generate")
        best_worker_endpoint = component.endpoint("best_worker_id")

        logger.debug("Starting to serve endpoints...")

        # Serve both endpoints concurrently
        try:
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    graceful_shutdown=True,
                    metrics_labels=[("service", "router")],
                ),
                best_worker_endpoint.serve_endpoint(
                    handler.best_worker_id,
                    graceful_shutdown=True,
                    metrics_labels=[("service", "router")],
                ),
            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Failed to serve endpoint: {e}")
            raise
        finally:
            logger.info("Standalone Router Service shutting down")

    uvloop.run(_run_worker())
