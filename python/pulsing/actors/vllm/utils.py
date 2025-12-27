"""vLLM Worker Actor - 基于 vLLM V1 引擎的高性能推理 Worker

参考 Dynamo 实现，支持：
1. Prefill/Decode 分离 (PD Disaggregation)
2. 多模态输入处理（图片）
3. KV Cache 管理和清理
4. LoRA 动态加载/卸载
5. OpenAI 兼容的文本输入输出模式
6. 引擎监控和健康检查
"""

# 工具函数和辅助类

"""vLLM Worker Actor - 基于 vLLM V1 引擎的高性能推理 Worker

参考 Dynamo 实现，支持：
1. Prefill/Decode 分离 (PD Disaggregation)
2. 多模态输入处理（图片）
3. KV Cache 管理和清理
4. LoRA 动态加载/卸载
5. OpenAI 兼容的文本输入输出模式
6. 引擎监控和健康检查
"""

import asyncio
import base64
import hashlib
import logging
import os
import platform
from io import BytesIO
from typing import Any, Final

try:
    from PIL import Image
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TextPrompt, TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.exceptions import EngineDeadError

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


logger = logging.getLogger(__name__)

# 多模态数据字典键
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


# ===== 工具函数 =====


def lora_name_to_id(lora_name: str) -> int:
    """从 LoRA 名称生成确定性的整数 ID

    使用 blake2b 哈希算法生成 64 位整数 ID

    Args:
        lora_name: LoRA 适配器名称

    Returns:
        64 位整数 ID
    """
    # 使用 blake2b 生成 8 字节哈希
    hash_digest = hashlib.blake2b(lora_name.encode("utf-8"), digest_size=8).digest()
    # 转换为无符号 64 位整数
    return int.from_bytes(hash_digest, byteorder="big", signed=False)


def _is_macos() -> bool:
    """检测是否在 macOS 上运行"""
    return platform.system() == "Darwin"


class VllmEngineMonitor:
    """vLLM 引擎监控器，用于收集引擎运行时统计信息"""

    def __init__(self, engine: "AsyncLLM"):
        self.engine = engine

    def get_cache_info(self) -> dict[str, Any]:
        """获取缓存配置信息"""
        try:
            config = self.engine.vllm_config
            return {
                "num_gpu_blocks": config.cache_config.num_gpu_blocks,
                "max_num_seqs": config.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
                "block_size": config.cache_config.block_size,
            }
        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
            return {}

    def get_model_config(self) -> dict[str, Any]:
        """获取模型配置信息"""
        try:
            config = self.engine.vllm_config
            model_config = config.model_config
            return {
                "max_model_len": model_config.max_model_len,
                "vocab_size": model_config.vocab_size,
                "dtype": str(model_config.dtype),
            }
        except Exception as e:
            logger.warning(f"Failed to get model config: {e}")
            return {}

    def get_health_status(self) -> dict[str, Any]:
        """获取引擎健康状态"""
        try:
            # 基本健康检查
            cache_info = self.get_cache_info()
            model_config = self.get_model_config()

            return {
                "status": "healthy",
                "cache_info": cache_info,
                "model_config": model_config,
            }
        except Exception as e:
            logger.exception(f"Failed to get health status: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


def _setup_macos_metal_env(
    mlx_device: str | None = None,
    metal_memory_fraction: float | None = None,
) -> None:
    """设置 macOS Metal/MLX 环境变量

    Args:
        mlx_device: MLX 设备类型 ('gpu' 或 'cpu')，默认 'gpu'
        metal_memory_fraction: Metal 内存使用比例 (0.0-1.0)，默认 0.8
    """
    if not _is_macos():
        return

    # 设置 MLX 设备
    if mlx_device is None:
        mlx_device = os.environ.get("VLLM_MLX_DEVICE", "gpu")

    if "VLLM_MLX_DEVICE" not in os.environ:
        os.environ["VLLM_MLX_DEVICE"] = mlx_device
        logger.info(f"Set VLLM_MLX_DEVICE={mlx_device} for macOS Metal support")

    # 设置 Metal 内存使用比例
    if metal_memory_fraction is None:
        metal_memory_fraction_str = os.environ.get("VLLM_METAL_MEMORY_FRACTION")
        if metal_memory_fraction_str:
            metal_memory_fraction = float(metal_memory_fraction_str)
        else:
            metal_memory_fraction = 0.8  # 默认 80%

    if "VLLM_METAL_MEMORY_FRACTION" not in os.environ:
        os.environ["VLLM_METAL_MEMORY_FRACTION"] = str(metal_memory_fraction)
        logger.info(
            f"Set VLLM_METAL_MEMORY_FRACTION={metal_memory_fraction} for macOS Metal support"
        )


class ImageLoader:
    """图片加载器，支持 URL 和 Base64 编码的图片"""

    def __init__(self, cache_size: int = 100):
        self._cache: dict[str, Image.Image] = {}
        self._cache_size = cache_size

    async def load_image(self, image_source: str) -> Image.Image:
        """加载图片，支持 Data URL (Base64) 和 HTTP(S) URL

        Args:
            image_source: 图片源，可以是 data: URL 或 http(s): URL

        Returns:
            PIL Image 对象
        """
        # 检查缓存
        if image_source in self._cache:
            logger.debug(f"Image cache hit: {image_source[:80]}...")
            return self._cache[image_source]

        if image_source.startswith("data:image"):
            try:
                # data:image/png;base64,xxxx
                header, data = image_source.split(",", 1)
                image_bytes = base64.b64decode(data)
                image = await asyncio.to_thread(Image.open, BytesIO(image_bytes))
                self._add_to_cache(image_source, image)
                return image
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {e}") from e

        # 暂时不支持 HTTP URL 下载，建议由前端/Processor 转换成 Base64
        raise ValueError(f"Unsupported image source: {image_source[:20]}...")

    def _add_to_cache(self, key: str, image: Image.Image) -> None:
        """添加到缓存，如果超过大小限制则清理旧条目"""
        if len(self._cache) >= self._cache_size:
            # 简单的 FIFO 策略
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = image
