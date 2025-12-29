#!/usr/bin/env python3
"""vLLM Worker 高级功能示例

演示如何使用 Pulsing 中的 vLLM Worker 的高级功能：
1. Prefill/Decode 分离
2. KV Cache 管理
3. LoRA 动态加载/卸载
4. 多模态处理
5. 健康检查
"""

import asyncio
import logging

from pulsing import System
from pulsing.actor import Message
from pulsing.actors.vllm_worker import VllmWorker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_basic_generation():
    """基础文本生成示例"""
    logger.info("=== 基础文本生成示例 ===")

    system = System.create()
    await system.start()

    try:
        # 创建聚合模式的 Worker（同时执行 prefill 和 decode）
        # 使用 use_vllm_tokenizer=True 启用文本输入输出模式
        worker_ref = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            role="aggregated",
            max_new_tokens=100,
            use_vllm_tokenizer=True,  # 启用文本模式
        )

        logger.info("等待 Worker 初始化...")
        await asyncio.sleep(10)  # 给引擎一些初始化时间

        # 方式 1: 使用 prompt 字段（文本模式）
        request = Message.from_json(
            "GenerateRequest",
            {
                "prompt": "Hello, how are you?",
                "temperature": 0.7,
                "max_tokens": 50,
            },
        )

        logger.info("发送生成请求（文本模式）...")
        response = await worker_ref.send(request)
        result = response.to_json()
        logger.info(f"生成结果: {result}")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def example_token_mode_generation():
    """Token 模式生成示例（需要预先进行 tokenization）"""
    logger.info("=== Token 模式生成示例 ===")

    system = System.create()
    await system.start()

    try:
        # 创建 Worker（默认使用 token 模式）
        worker_ref = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            role="aggregated",
            max_new_tokens=100,
        )

        logger.info("等待 Worker 初始化...")
        await asyncio.sleep(10)

        # 使用 token_ids 字段（需要预先 tokenize）
        # 这里使用一些示例 token IDs
        request = Message.from_json(
            "GenerateRequest",
            {
                "token_ids": [1, 15339, 11, 1268, 525, 498, 30],  # "Hello, how are you?" 的近似 tokens
                "sampling_options": {"temperature": 0.7, "top_p": 0.9},
                "stop_conditions": {"max_tokens": 50},
            },
        )

        logger.info("发送生成请求（Token 模式）...")
        response = await worker_ref.send(request)
        result = response.to_json()
        logger.info(f"生成结果: {result}")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def example_prefill_decode_separation():
    """Prefill/Decode 分离示例"""
    logger.info("=== Prefill/Decode 分离示例 ===")

    system = System.create()
    await system.start()

    try:
        # 创建 Prefill Worker
        prefill_worker = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen3-0.6B",
            role="prefill",
        )

        # 创建 Decode Worker
        decode_worker = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen3-0.6B",
            role="decode",
        )

        logger.info("等待 Workers 初始化...")
        await asyncio.sleep(10)

        # 步骤 1: Prefill 阶段
        prefill_request = Message.from_json(
            "GenerateRequest",
            {
                "token_ids": [1, 2, 3, 4, 5],
                "sampling_options": {"temperature": 0.7},
                "stop_conditions": {"max_tokens": 1},  # Prefill 只生成 1 个 token
            },
        )

        logger.info("发送 Prefill 请求...")
        prefill_response = await prefill_worker.send(prefill_request)
        prefill_result = prefill_response.to_json()
        logger.info(f"Prefill 结果: {prefill_result}")

        # 步骤 2: Decode 阶段（使用 Prefill 的结果）
        decode_request = Message.from_json(
            "GenerateRequest",
            {
                "token_ids": prefill_result.get("token_ids", []),
                "prefill_result": prefill_result,  # 传递 Prefill 结果
                "sampling_options": {"temperature": 0.7},
                "stop_conditions": {"max_tokens": 50},
            },
        )

        logger.info("发送 Decode 请求...")
        decode_response = await decode_worker.send(decode_request)
        decode_result = decode_response.to_json()
        logger.info(f"Decode 结果: {decode_result}")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def example_kv_cache_management():
    """KV Cache 管理示例"""
    logger.info("=== KV Cache 管理示例 ===")

    system = System.create()
    await system.start()

    try:
        worker_ref = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen3-0.6B",
            role="aggregated",
        )

        logger.info("等待 Worker 初始化...")
        await asyncio.sleep(5)

        # 发送一些请求以填充 KV Cache
        request = Message.from_json(
            "GenerateRequest",
            {
                "token_ids": [1, 2, 3, 4, 5],
                "sampling_options": {"temperature": 0.7},
                "stop_conditions": {"max_tokens": 20},
            },
        )

        logger.info("发送生成请求...")
        await worker_ref.send(request)

        # 清理 KV Cache
        clear_request = Message.from_json("ClearKVCache", {})
        logger.info("清理 KV Cache...")
        clear_response = await worker_ref.send(clear_request)
        logger.info(f"清理结果: {clear_response.to_json()}")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def example_lora_management():
    """LoRA 动态加载/卸载示例"""
    logger.info("=== LoRA 管理示例 ===")

    system = System.create()
    await system.start()

    try:
        worker_ref = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen3-0.6B",
            role="aggregated",
        )

        logger.info("等待 Worker 初始化...")
        await asyncio.sleep(5)

        # 列出当前加载的 LoRAs
        list_request = Message.from_json("ListLoRAs", {})
        logger.info("列出 LoRAs...")
        list_response = await worker_ref.send(list_request)
        logger.info(f"当前 LoRAs: {list_response.to_json()}")

        # 加载 LoRA（需要实际的 LoRA 路径）
        # load_request = Message.from_json(
        #     "LoadLoRA",
        #     {
        #         "lora_name": "my_lora_adapter",
        #         "lora_path": "/path/to/lora/adapter",
        #     },
        # )
        # logger.info("加载 LoRA...")
        # load_response = await worker_ref.send(load_request)
        # logger.info(f"加载结果: {load_response.to_json()}")

        # 使用 LoRA 生成
        # request = Message.from_json(
        #     "GenerateRequest",
        #     {
        #         "model": "my_lora_adapter",  # 指定使用的 LoRA
        #         "token_ids": [1, 2, 3, 4, 5],
        #         "sampling_options": {"temperature": 0.7},
        #         "stop_conditions": {"max_tokens": 20},
        #     },
        # )
        # logger.info("使用 LoRA 生成...")
        # response = await worker_ref.send(request)
        # logger.info(f"生成结果: {response.to_json()}")

        # 卸载 LoRA
        # unload_request = Message.from_json(
        #     "UnloadLoRA",
        #     {"lora_name": "my_lora_adapter"},
        # )
        # logger.info("卸载 LoRA...")
        # unload_response = await worker_ref.send(unload_request)
        # logger.info(f"卸载结果: {unload_response.to_json()}")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def example_health_check():
    """健康检查示例"""
    logger.info("=== 健康检查示例 ===")

    system = System.create()
    await system.start()

    try:
        worker_ref = await system.spawn_actor(
            VllmWorker,
            model="Qwen/Qwen3-0.6B",
            role="aggregated",
        )

        logger.info("等待 Worker 初始化...")
        await asyncio.sleep(5)

        # 发送健康检查请求
        health_request = Message.from_json("HealthCheck", {})
        logger.info("检查 Worker 健康状态...")
        health_response = await worker_ref.send(health_request)
        health_status = health_response.to_json()

        logger.info(f"健康状态: {health_status['status']}")
        logger.info(f"角色: {health_status.get('role')}")
        logger.info(f"Worker ID: {health_status.get('worker_id')}")

        if "cache_info" in health_status:
            logger.info("缓存信息:")
            for key, value in health_status["cache_info"].items():
                logger.info(f"  {key}: {value}")

        if "model_config" in health_status:
            logger.info("模型配置:")
            for key, value in health_status["model_config"].items():
                logger.info(f"  {key}: {value}")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def example_multimodal():
    """多模态处理示例（图片输入）"""
    logger.info("=== 多模态处理示例 ===")

    system = System.create()
    await system.start()

    try:
        # 创建支持多模态的 Worker
        await system.spawn_actor(
            VllmWorker,
            model="llava-hf/llava-1.5-7b-hf",  # 需要支持多模态的模型
            role="aggregated",
            enable_multimodal=True,
        )

        logger.info("等待 Worker 初始化...")
        await asyncio.sleep(10)

        # 准备图片数据（Base64 编码）
        # 这里使用一个简单的示例图片
        # 实际使用时需要读取真实图片
        # with open("image.jpg", "rb") as f:
        #     image_data = base64.b64encode(f.read()).decode("utf-8")
        #     image_url = f"data:image/jpeg;base64,{image_data}"

        # 发送多模态请求
        # request = Message.from_json(
        #     "GenerateRequest",
        #     {
        #         "token_ids": [1, 2, 3, 4, 5],  # 包含图片占位符的 tokens
        #         "multi_modal_data": {
        #             "image_url": [{"Url": image_url}]
        #         },
        #         "sampling_options": {"temperature": 0.7},
        #         "stop_conditions": {"max_tokens": 50},
        #     },
        # )

        # logger.info("发送多模态生成请求...")
        # response = await worker_ref.send(request)
        # result = response.to_json()
        # logger.info(f"生成结果: {result}")

        logger.info("多模态示例需要实际的图片数据和多模态模型")

    finally:
        await system.stop()
        logger.info("系统已停止")


async def main():
    """运行所有示例"""
    examples = [
        ("基础文本生成", example_basic_generation),
        ("Token 模式生成", example_token_mode_generation),
        ("Prefill/Decode 分离", example_prefill_decode_separation),
        ("KV Cache 管理", example_kv_cache_management),
        ("LoRA 管理", example_lora_management),
        ("健康检查", example_health_check),
        ("多模态处理", example_multimodal),
    ]

    logger.info("vLLM Worker 高级功能示例")
    logger.info("=" * 60)

    for i, (name, example_func) in enumerate(examples, 1):
        logger.info(f"\n示例 {i}/{len(examples)}: {name}")
        try:
            # 注意：某些示例需要较长的初始化时间
            # 如果只想运行特定示例，可以注释掉其他示例
            if name in ["基础文本生成", "健康检查"]:
                await example_func()
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"示例 {name} 失败: {e}", exc_info=True)
        logger.info("-" * 60)

    logger.info("\n所有示例完成!")


if __name__ == "__main__":
    asyncio.run(main())

