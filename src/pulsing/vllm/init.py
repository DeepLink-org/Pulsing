"""vLLM initialization functions with hyperparameter integration."""

import asyncio
import logging
import os
import tempfile
from typing import Optional

from hyperparameter import auto_param, param_scope
from prometheus_client import REGISTRY
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.args import ENABLE_LMCACHE, Config
from dynamo.vllm.handlers import DecodeWorkerHandler, PrefillWorkerHandler
from dynamo.vllm.health_check import (
    VllmHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
)
from dynamo.vllm.main import (
    get_engine_cache_info,
    setup_kv_event_publisher,
    setup_lmcache_environment,
    setup_vllm_engine,
)
from dynamo.vllm.multimodal_handlers import (
    EncodeWorkerHandler,
    MultimodalDecodeWorkerHandler,
    MultimodalPDWorkerHandler,
    ProcessorHandler,
)
from dynamo.vllm.publisher import StatLoggerFactory

logger = logging.getLogger(__name__)


def _build_engine_args(
    model: str,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    block_size: Optional[int] = None,
    max_model_len: Optional[int] = None,
    served_model_name: Optional[str] = None,
    **kwargs,
) -> AsyncEngineArgs:
    """Build AsyncEngineArgs from parameters."""
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

    from argparse import Namespace

    return AsyncEngineArgs.from_cli_args(Namespace(**engine_args_dict))


def _setup_vllm_engine(
    engine_args: AsyncEngineArgs,
    served_model_name: str,
    connector_list: list,
    stat_logger=None,
):
    """Setup vLLM engine."""
    prometheus_temp_dir = None
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        prometheus_temp_dir = tempfile.TemporaryDirectory(prefix="vllm_prometheus_")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_temp_dir.name
        logger.debug(
            f"Created PROMETHEUS_MULTIPROC_DIR at: {os.environ['PROMETHEUS_MULTIPROC_DIR']}"
        )

    setup_multiprocess_prometheus()
    logger.debug(
        f"Prometheus multiproc dir set to: {os.environ.get('PROMETHEUS_MULTIPROC_DIR')}"
    )

    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if ENABLE_LMCACHE:
        setup_lmcache_environment()
        logger.info("LMCache enabled for VllmWorker")
    else:
        logger.debug("LMCache is disabled")

    default_sampling_params = (
        engine_args.create_model_config().get_diff_sampling_param()
    )

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    consolidator_endpoints = None
    if "kvbm" in [c.lower() for c in connector_list]:
        try:
            from kvbm.vllm_integration.consolidator_config import (
                get_consolidator_endpoints,
            )

            consolidator_endpoints = get_consolidator_endpoints(vllm_config)
        except Exception as e:
            logger.warning(
                f"KVBM connector is enabled but failed to get consolidator endpoints: {e}. "
                "Continuing without KV event consolidation."
            )
    vllm_config.consolidator_endpoints = consolidator_endpoints

    factory = []
    if stat_logger:
        factory.append(stat_logger)

    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        stat_loggers=factory,
        disable_log_requests=engine_args.disable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    if ENABLE_LMCACHE:
        logger.info(
            f"VllmWorker for {served_model_name} has been initialized with LMCache"
        )
    else:
        logger.info(f"VllmWorker for {served_model_name} has been initialized")

    return engine_client, vllm_config, default_sampling_params, prometheus_temp_dir


def _setup_kv_event_publisher(
    engine_args: AsyncEngineArgs,
    is_decode_worker: bool,
    component,
    generate_endpoint,
    vllm_config,
    consolidator_enabled: bool = False,
    consolidator_port: Optional[int] = 5558,
) -> Optional[list]:
    """Setup KV event publisher."""
    if not engine_args.enable_prefix_caching:
        return None

    if is_decode_worker:
        logger.info("Skipping KV event publisher setup for decode worker")
        return None

    if engine_args.kv_events_config is None:
        return None

    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    kv_publishers = []

    for dp_rank in range(data_parallel_size):
        if consolidator_enabled:
            zmq_endpoint = f"tcp://127.0.0.1:{consolidator_port}"
            logger.info(
                f"KV event publisher for dp_rank={dp_rank} subscribing to consolidator at {zmq_endpoint}"
            )
        else:
            zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
                engine_args.kv_events_config.endpoint,
                data_parallel_rank=dp_rank,
            ).replace("*", "127.0.0.1")
            logger.info(
                f"KV event publisher for dp_rank={dp_rank} subscribing to vLLM at {zmq_endpoint}"
            )

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=generate_endpoint.connection_id(),
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
        )
        kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)
        kv_publishers.append(kv_publisher)
        logger.info(
            f"Worker reading KV events for dp_rank={dp_rank} from {zmq_endpoint}"
        )

    return kv_publishers if kv_publishers else None


async def _register_vllm_model(
    model_input: ModelInput,
    model_type: ModelType,
    generate_endpoint,
    model: str,
    served_model_name: str,
    engine_client: AsyncLLM,
    vllm_config,
    engine_args: AsyncEngineArgs,
    migration_limit: int,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    custom_jinja_template: Optional[str] = None,
):
    """Register vLLM model."""
    runtime_config = ModelRuntimeConfig()

    logger.info(
        f"Getting engine runtime configuration metadata from vLLM engine for {model_type}..."
    )
    runtime_values = get_engine_cache_info(engine_client)
    runtime_config.total_kv_blocks = runtime_values["num_gpu_blocks"]
    runtime_config.max_num_seqs = runtime_values["max_num_seqs"]
    runtime_config.max_num_batched_tokens = runtime_values["max_num_batched_tokens"]

    if model_type != ModelType.Prefill:
        runtime_config.tool_call_parser = tool_call_parser
        runtime_config.reasoning_parser = reasoning_parser

    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    runtime_config.data_parallel_size = data_parallel_size

    await register_llm(
        model_input,
        model_type,
        generate_endpoint,
        model,
        served_model_name,
        kv_cache_block_size=engine_args.block_size,
        migration_limit=migration_limit,
        runtime_config=runtime_config,
        custom_template_path=custom_jinja_template,
    )


@auto_param("vllm.worker")
async def init_decode_worker(
    runtime: DistributedRuntime,
    model: str = None,
    component: str = "backend",
    endpoint: str = "generate",
    served_model_name: Optional[str] = None,
    migration_limit: int = 0,
    tool_call_parser: Optional[str] = None,
    reasoning_parser: Optional[str] = None,
    custom_jinja_template: Optional[str] = None,
    connector: Optional[list] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    block_size: Optional[int] = None,
    **kwargs,
):
    """
    Initialize and serve a decode worker.

    Args:
        runtime: DistributedRuntime instance
        model: Model path or HuggingFace model name
        component: Component name. Default: 'backend'
        endpoint: Endpoint name. Default: 'generate'
        served_model_name: Name to serve the model as
        migration_limit: Max migration times
        tool_call_parser: Tool call parser name
        reasoning_parser: Reasoning parser name
        custom_jinja_template: Custom Jinja template path
        connector: List of connectors. Default: ['nixl']
        tensor_parallel_size: TP size. Default: 1
        pipeline_parallel_size: PP size. Default: 1
        max_model_len: Max sequence length
        gpu_memory_utilization: GPU memory fraction. Default: 0.9
        enable_prefix_caching: Enable prefix caching. Default: True
        block_size: KV cache block size
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    connector_list = [c.lower() for c in (connector or ["nixl"])]
    served_model_name = served_model_name or model
    block_size = block_size or 16

    engine_args = _build_engine_args(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        block_size=block_size,
        max_model_len=max_model_len,
        served_model_name=served_model_name,
        **kwargs,
    )

    component_obj = runtime.namespace(namespace).component(component)
    await component_obj.create_service()

    generate_endpoint = component_obj.endpoint(endpoint)
    clear_endpoint = component_obj.endpoint("clear_kv_blocks")

    factory = StatLoggerFactory(
        component_obj,
        engine_args.data_parallel_rank or 0,
        metrics_labels=[("model", served_model_name)],
    )

    engine_client, vllm_config, default_sampling_params, prometheus_temp_dir = (
        _setup_vllm_engine(engine_args, served_model_name, connector_list, factory)
    )

    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.set_request_total_slots_all(vllm_config.scheduler_config.max_num_seqs)
    factory.init_publish()

    handler = DecodeWorkerHandler(
        runtime,
        component_obj,
        engine_client,
        default_sampling_params,
        getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
    )
    handler.add_temp_dir(prometheus_temp_dir)

    consolidator_enabled = False
    consolidator_port = None
    if (
        hasattr(vllm_config, "consolidator_endpoints")
        and vllm_config.consolidator_endpoints
    ):
        consolidator_output_endpoint = vllm_config.consolidator_endpoints[2]
        consolidator_port = int(consolidator_output_endpoint.split(":")[-1])
        consolidator_enabled = True

    kv_publishers = _setup_kv_event_publisher(
        engine_args,
        False,
        component_obj,
        generate_endpoint,
        vllm_config,
        consolidator_enabled=consolidator_enabled,
        consolidator_port=consolidator_port,
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    if not engine_args.disable_log_stats:
        register_engine_metrics_callback(
            endpoint=generate_endpoint, registry=REGISTRY, metric_prefix_filter="vllm:"
        )

    if not engine_args.data_parallel_rank:
        await _register_vllm_model(
            ModelInput.Tokens,
            ModelType.Chat | ModelType.Completions,
            generate_endpoint,
            model,
            served_model_name,
            engine_client,
            vllm_config,
            engine_args,
            migration_limit,
            tool_call_parser,
            reasoning_parser,
            custom_jinja_template,
        )

    health_check_payload = VllmHealthCheckPayload(engine_client).to_dict()

    try:
        logger.debug("Starting serve_endpoint for decode worker")
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=migration_limit <= 0,
                metrics_labels=[("model", served_model_name)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", served_model_name)],
            ),
        )
        logger.debug("serve_endpoint completed for decode worker")
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.debug("Cleaning up decode worker")
        handler.cleanup()


@auto_param("vllm.worker")
async def init_prefill_worker(
    runtime: DistributedRuntime,
    model: str = None,
    component: str = "prefill",
    endpoint: str = "generate",
    served_model_name: Optional[str] = None,
    connector: Optional[list] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    block_size: Optional[int] = None,
    **kwargs,
):
    """
    Initialize and serve a prefill worker.

    Args:
        runtime: DistributedRuntime instance
        model: Model path or HuggingFace model name
        component: Component name. Default: 'prefill'
        endpoint: Endpoint name. Default: 'generate'
        served_model_name: Name to serve the model as
        connector: List of connectors. Default: ['nixl']
        tensor_parallel_size: TP size. Default: 1
        pipeline_parallel_size: PP size. Default: 1
        max_model_len: Max sequence length
        gpu_memory_utilization: GPU memory fraction. Default: 0.9
        enable_prefix_caching: Enable prefix caching. Default: True
        block_size: KV cache block size
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    connector_list = [c.lower() for c in (connector or ["nixl"])]
    served_model_name = served_model_name or model
    block_size = block_size or 16

    engine_args = _build_engine_args(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        block_size=block_size,
        max_model_len=max_model_len,
        served_model_name=served_model_name,
        **kwargs,
    )

    component_obj = runtime.namespace(namespace).component(component)
    await component_obj.create_service()

    generate_endpoint = component_obj.endpoint(endpoint)
    clear_endpoint = component_obj.endpoint("clear_kv_blocks")

    engine_client, vllm_config, default_sampling_params, prometheus_temp_dir = (
        _setup_vllm_engine(engine_args, served_model_name, connector_list)
    )

    handler = PrefillWorkerHandler(
        runtime,
        component_obj,
        engine_client,
        default_sampling_params,
        getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
    )
    handler.add_temp_dir(prometheus_temp_dir)

    consolidator_enabled = False
    consolidator_port = None
    if (
        hasattr(vllm_config, "consolidator_endpoints")
        and vllm_config.consolidator_endpoints
    ):
        consolidator_output_endpoint = vllm_config.consolidator_endpoints[2]
        consolidator_port = int(consolidator_output_endpoint.split(":")[-1])
        consolidator_enabled = True

    kv_publishers = _setup_kv_event_publisher(
        engine_args,
        False,
        component_obj,
        generate_endpoint,
        vllm_config,
        consolidator_enabled=consolidator_enabled,
        consolidator_port=consolidator_port,
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    if not engine_args.disable_log_stats:
        register_engine_metrics_callback(
            endpoint=generate_endpoint, registry=REGISTRY, metric_prefix_filter="vllm:"
        )

    if not engine_args.data_parallel_rank:
        await _register_vllm_model(
            ModelInput.Tokens,
            ModelType.Prefill,
            generate_endpoint,
            model,
            served_model_name,
            engine_client,
            vllm_config,
            engine_args,
            0,  # migration_limit=0 for prefill
        )

    health_check_payload = VllmPrefillHealthCheckPayload(engine_client).to_dict()

    try:
        logger.debug("Starting serve_endpoint for prefill worker")
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("model", served_model_name)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", served_model_name)],
            ),
        )
        logger.debug("serve_endpoint completed for prefill worker")
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.debug("Cleaning up prefill worker")
        handler.cleanup()


@auto_param("vllm.worker")
async def init_multimodal_processor_worker(
    runtime: DistributedRuntime,
    model: str = None,
    component: str = "processor",
    endpoint: str = "generate",
    served_model_name: Optional[str] = None,
    mm_prompt_template: str = "USER: <image>\n<prompt> ASSISTANT:",
    block_size: Optional[int] = None,
    **kwargs,
):
    """
    Initialize and serve a multimodal processor worker.

    Args:
        runtime: DistributedRuntime instance
        model: Model path or HuggingFace model name
        component: Component name. Default: 'processor'
        endpoint: Endpoint name. Default: 'generate'
        served_model_name: Name to serve the model as
        mm_prompt_template: Multi-modal prompt template
        block_size: KV cache block size
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    served_model_name = served_model_name or model
    block_size = block_size or 16

    engine_args = _build_engine_args(
        model=model,
        block_size=block_size,
        served_model_name=served_model_name,
        **kwargs,
    )

    component_obj = runtime.namespace(namespace).component(component)
    await component_obj.create_service()

    generate_endpoint = component_obj.endpoint(endpoint)

    # Get encode worker client
    encode_worker_client = (
        await runtime.namespace(namespace)
        .component("encoder")
        .endpoint("generate")
        .client()
    )

    handler = ProcessorHandler(engine_args, encode_worker_client, mm_prompt_template)

    logger.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    await register_llm(
        ModelInput.Text,
        ModelType.Chat,
        generate_endpoint,
        model,
        served_model_name,
        kv_cache_block_size=block_size,
    )

    logger.info("Starting to serve the processor endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


@auto_param("vllm.worker")
async def init_multimodal_encode_worker_worker(
    runtime: DistributedRuntime,
    model: str = None,
    component: str = "encoder",
    endpoint: str = "generate",
    served_model_name: Optional[str] = None,
    **kwargs,
):
    """
    Initialize and serve a multimodal encode worker.

    Args:
        runtime: DistributedRuntime instance
        model: Model path or HuggingFace model name
        component: Component name. Default: 'encoder'
        endpoint: Endpoint name. Default: 'generate'
        served_model_name: Name to serve the model as
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    served_model_name = served_model_name or model

    engine_args = _build_engine_args(
        model=model,
        served_model_name=served_model_name,
        **kwargs,
    )

    component_obj = runtime.namespace(namespace).component(component)
    await component_obj.create_service()

    generate_endpoint = component_obj.endpoint(endpoint)

    # Get PD worker client
    pd_worker_client = (
        await runtime.namespace(namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    handler = EncodeWorkerHandler(engine_args, pd_worker_client)
    await handler.async_init(runtime)

    logger.info("Waiting for PD Worker Instances ...")
    await pd_worker_client.wait_for_instances()
    logger.info("Starting to serve the encode worker endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


@auto_param("vllm.worker")
async def init_multimodal_worker_worker(
    runtime: DistributedRuntime,
    model: str = None,
    component: str = "backend",
    endpoint: str = "generate",
    served_model_name: Optional[str] = None,
    is_prefill_worker: bool = False,
    multimodal_decode_worker: bool = False,
    connector: Optional[list] = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
    block_size: Optional[int] = None,
    **kwargs,
):
    """
    Initialize and serve a multimodal worker.

    Args:
        runtime: DistributedRuntime instance
        model: Model path or HuggingFace model name
        component: Component name. Default: 'backend'
        endpoint: Endpoint name. Default: 'generate'
        served_model_name: Name to serve the model as
        is_prefill_worker: Enable prefill functionality
        multimodal_decode_worker: Run as decode worker
        connector: List of connectors. Default: ['nixl']
        tensor_parallel_size: TP size. Default: 1
        pipeline_parallel_size: PP size. Default: 1
        max_model_len: Max sequence length
        gpu_memory_utilization: GPU memory fraction. Default: 0.9
        enable_prefix_caching: Enable prefix caching. Default: True
        block_size: KV cache block size
    """
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    connector_list = [c.lower() for c in (connector or ["nixl"])]
    served_model_name = served_model_name or model
    block_size = block_size or 16

    engine_args = _build_engine_args(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        block_size=block_size,
        max_model_len=max_model_len,
        served_model_name=served_model_name,
        **kwargs,
    )

    # Create minimal Config for compatibility with existing functions
    temp_config = Config()
    temp_config.model = model
    temp_config.served_model_name = served_model_name
    temp_config.namespace = namespace
    temp_config.component = component
    temp_config.endpoint = endpoint
    temp_config.is_prefill_worker = is_prefill_worker
    temp_config.is_decode_worker = False
    temp_config.multimodal_decode_worker = multimodal_decode_worker
    temp_config.connector_list = connector_list
    temp_config.engine_args = engine_args
    temp_config.has_connector = lambda name: name.lower() in connector_list

    component_obj = runtime.namespace(namespace).component(component)
    await component_obj.create_service()

    generate_endpoint = component_obj.endpoint(endpoint)
    clear_endpoint = component_obj.endpoint("clear_kv_blocks")

    engine_client, vllm_config, default_sampling_params, prometheus_temp_dir = (
        setup_vllm_engine(temp_config)
    )

    # Set up decode worker client for disaggregated mode
    decode_worker_client = None
    if is_prefill_worker:
        decode_worker_client = (
            await runtime.namespace(namespace)
            .component("decoder")
            .endpoint("generate")
            .client()
        )
        await decode_worker_client.wait_for_instances()
        logger.info("Connected to decode worker for disaggregated mode")

    # Choose handler based on worker type
    if multimodal_decode_worker:
        handler = MultimodalDecodeWorkerHandler(
            runtime, component_obj, engine_client, temp_config
        )
    else:
        handler = MultimodalPDWorkerHandler(
            runtime, component_obj, engine_client, temp_config, decode_worker_client
        )
    handler.add_temp_dir(prometheus_temp_dir)

    await handler.async_init(runtime)

    # Set up KV event publisher for prefix caching if enabled
    kv_publisher = setup_kv_event_publisher(
        temp_config, component_obj, generate_endpoint, vllm_config
    )
    if kv_publisher:
        handler.kv_publisher = kv_publisher

    metrics_labels = [("model", model)]
    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=metrics_labels
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks, metrics_labels=metrics_labels
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()
