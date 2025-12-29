"""vLLM Worker 工具函数和辅助类

包含：
- VllmEngineMonitor: 引擎健康监控
- ImageLoader: 多模态图片加载
- 环境变量设置函数
"""

import asyncio
import base64
import hashlib
import logging
import os
import platform
from collections.abc import Callable
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


# ===== EngineDeadError 统一处理 =====


def handle_engine_dead_error(
    error: Exception,
    context: str,
    on_engine_dead: Callable[[], None] | None = None,
) -> None:
    """统一处理 EngineDeadError

    Args:
        error: 捕获的异常
        context: 错误发生的上下文描述
        on_engine_dead: 引擎死亡回调函数
    """
    logger.error(f"vLLM EngineDeadError during {context}: {error}")
    if on_engine_dead is not None:
        logger.warning(f"Calling on_engine_dead callback from {context}...")
        try:
            on_engine_dead()
        except Exception as callback_error:
            logger.exception(f"Error in on_engine_dead callback: {callback_error}")


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
    """vLLM 引擎监控器，用于收集引擎运行时统计信息和持续健康检查

    参考 Dynamo engine_monitor.py 实现，支持：
    1. 静态配置信息获取
    2. 持续的后台健康检查
    3. 引擎死亡检测和回调通知
    """

    # 健康检查间隔（秒）
    HEALTH_CHECK_INTERVAL = 2
    # 引擎关闭超时（秒）
    ENGINE_SHUTDOWN_TIMEOUT = 30

    def __init__(
        self,
        engine: "AsyncLLM",
        on_engine_dead: "Callable[[], None] | None" = None,
    ):
        """初始化引擎监控器

        Args:
            engine: vLLM AsyncLLM 引擎实例
            on_engine_dead: 引擎死亡时的回调函数 (可选)
        """
        self.engine = engine
        self._on_engine_dead = on_engine_dead
        self._monitor_task: asyncio.Task | None = None
        self._is_healthy = True
        self._last_error: str | None = None

    def start_monitoring(
        self, on_engine_dead: "Callable[[], None] | None" = None
    ) -> None:
        """启动后台健康检查任务

        Args:
            on_engine_dead: 引擎死亡时的回调函数 (可选，会覆盖构造函数中的设置)
        """
        if on_engine_dead is not None:
            self._on_engine_dead = on_engine_dead

        if self._monitor_task is not None and not self._monitor_task.done():
            logger.warning("Health check task already running")
            return

        self._monitor_task = asyncio.create_task(self._health_check_loop())
        logger.info(
            f"VllmEngineMonitor started, health check interval: {self.HEALTH_CHECK_INTERVAL}s"
        )

    def stop_monitoring(self) -> None:
        """停止后台健康检查任务"""
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            logger.info("VllmEngineMonitor stopped")

    async def _health_check_loop(self) -> None:
        """持续的健康检查循环 (参考 Dynamo engine_monitor.py)"""
        while True:
            try:
                # 调用 vLLM 引擎的健康检查方法
                await self.engine.check_health()
                self._is_healthy = True
                self._last_error = None
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

            except Exception as e:
                # 检查是否是 EngineDeadError
                error_name = type(e).__name__
                if error_name == "EngineDeadError" or "EngineDeadError" in str(type(e)):
                    logger.error(f"vLLM EngineDeadError detected: {e}")
                    self._is_healthy = False
                    self._last_error = str(e)

                    # 尝试关闭引擎
                    await self._shutdown_engine()

                    # 调用回调函数
                    if self._on_engine_dead is not None:
                        logger.warning("Calling on_engine_dead callback...")
                        try:
                            self._on_engine_dead()
                        except Exception as callback_error:
                            logger.exception(
                                f"Error in on_engine_dead callback: {callback_error}"
                            )

                    # 退出健康检查循环
                    break
                else:
                    # 其他异常，记录但继续检查
                    logger.warning(f"Health check failed with non-fatal error: {e}")
                    self._is_healthy = False
                    self._last_error = str(e)
                    await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

            except asyncio.CancelledError:
                logger.debug("Health check loop cancelled")
                break

    async def _shutdown_engine(self) -> None:
        """关闭 vLLM 引擎 (带超时保护)"""
        logger.warning("Initiating vLLM engine shutdown...")

        try:
            # 使用 asyncio.wait_for 添加超时保护
            await asyncio.wait_for(
                asyncio.to_thread(self.engine.shutdown),
                timeout=self.ENGINE_SHUTDOWN_TIMEOUT,
            )
            logger.info("vLLM engine shutdown completed")
        except asyncio.TimeoutError:
            logger.error(
                f"vLLM engine shutdown timed out after {self.ENGINE_SHUTDOWN_TIMEOUT}s"
            )
        except Exception as e:
            logger.warning(f"vLLM engine shutdown failed: {e}")

    @property
    def is_healthy(self) -> bool:
        """返回引擎是否健康"""
        return self._is_healthy

    @property
    def last_error(self) -> str | None:
        """返回最后一次错误信息"""
        return self._last_error

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
            cache_info = self.get_cache_info()
            model_config = self.get_model_config()

            status = "healthy" if self._is_healthy else "unhealthy"

            result = {
                "status": status,
                "cache_info": cache_info,
                "model_config": model_config,
            }

            if self._last_error:
                result["last_error"] = self._last_error

            return result
        except Exception as e:
            logger.exception(f"Failed to get health status: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def __del__(self):
        """析构函数，确保停止监控任务"""
        self.stop_monitoring()


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
