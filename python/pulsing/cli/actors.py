"""Pulsing CLI - Actor commands"""

import uvloop


def start_router(
    namespace: str,
    addr: str | None,
    seeds: list[str],
    http_host: str,
    http_port: int,
    model_name: str,
    scheduler_type: str,
):
    """Start Router with OpenAI-compatible API"""
    from pulsing.actor import SystemConfig, create_actor_system
    from pulsing.actor.helpers import run_until_signal

    from ..actors import (
        LeastConnectionScheduler,
        RandomScheduler,
        RoundRobinScheduler,
        StreamLoadScheduler,
    )
    from ..actors.router import start_router as start_router_service
    from ..actors.router import stop_router

    # 选择调度器类
    scheduler_map = {
        "round_robin": RoundRobinScheduler,
        "random": RandomScheduler,
        "least_connection": LeastConnectionScheduler,
        "stream_load": StreamLoadScheduler,
    }
    scheduler_class = scheduler_map.get(scheduler_type)
    if not scheduler_class:
        raise ValueError(
            f"Unknown scheduler: {scheduler_type}. Options: {list(scheduler_map.keys())}"
        )

    print(f"Starting Router (namespace={namespace}, model={model_name})")
    print(f"  Actor System addr: {addr or 'auto'}")
    print(f"  HTTP API: http://{http_host}:{http_port}")
    print(f"  Scheduler: {scheduler_type}")

    async def run():
        # 创建 ActorSystem
        if addr:
            config = SystemConfig.with_addr(addr)
        else:
            config = SystemConfig.standalone()

        if seeds:
            config = config.with_seeds(seeds)

        system = await create_actor_system(config)
        print(f"[Router] ActorSystem started at {system.addr}")

        # 启动 Router HTTP 服务器
        runner = await start_router_service(
            system,
            http_host=http_host,
            http_port=http_port,
            model_name=model_name,
            worker_name="worker",
            scheduler_type=scheduler_type,
        )

        # 运行直到收到信号
        try:
            await run_until_signal(system, "router")
        finally:
            await stop_router(runner)

    uvloop.run(run())


def start_transformers(
    model: str,
    namespace: str,
    addr: str | None,
    seeds: list[str],
    device: str,
    max_new_tokens: int,
    preload_model: bool,
):
    """Start Transformers Worker"""
    from pulsing.actor.helpers import spawn_and_run

    from ..actors import GenerationConfig, TransformersWorker

    print(f"Starting Transformers Worker (model={model}, namespace={namespace})")
    print(f"  Device: {device}")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Preload: {preload_model}")

    async def run():
        gen_config = GenerationConfig(max_new_tokens=max_new_tokens)
        worker = TransformersWorker(
            model_name=model,
            device=device,
            gen_config=gen_config,
            preload=preload_model,
        )

        await spawn_and_run(
            worker,
            name="worker",
            addr=addr,
            seeds=seeds if seeds else None,
            public=True,
        )

    uvloop.run(run())


def start_vllm(
    model: str,
    namespace: str,
    addr: str | None,
    seeds: list[str],
    max_new_tokens: int,
    role: str = "aggregated",
    mlx_device: str | None = None,
    metal_memory_fraction: float | None = None,
):
    """Start vLLM Worker

    Args:
        model: Model path or name
        namespace: Service namespace
        addr: Actor System bind address
        seeds: List of seed nodes
        max_new_tokens: Maximum tokens to generate
        role: Worker role ('aggregated', 'prefill', 'decode')
        mlx_device: MLX device type for macOS ('gpu' or 'cpu')
        metal_memory_fraction: Metal memory fraction for macOS (0.0-1.0)
    """
    from pulsing.actor.helpers import spawn_and_run

    from ..actors.vllm import VllmWorker

    print(f"Starting vLLM Worker (model={model}, namespace={namespace}, role={role})")
    print(f"  Max tokens: {max_new_tokens}")

    # 显示 macOS Metal 配置
    import platform

    if platform.system() == "Darwin":
        mlx_info = mlx_device or "gpu (default)"
        if metal_memory_fraction is not None:
            try:
                memory_value = float(metal_memory_fraction)
                memory_info = f"{memory_value:.2f}"
            except (ValueError, TypeError):
                memory_info = str(metal_memory_fraction)
        else:
            memory_info = "0.8 (default)"
        print(
            f"  macOS Metal support: MLX device={mlx_info}, memory fraction={memory_info}"
        )

    async def run():
        worker = VllmWorker(
            model=model,
            role=role,
            max_new_tokens=max_new_tokens,
            mlx_device=mlx_device,
            metal_memory_fraction=metal_memory_fraction,
        )

        await spawn_and_run(
            worker,
            name="worker",
            addr=addr,
            seeds=seeds if seeds else None,
            public=True,
        )

    uvloop.run(run())
