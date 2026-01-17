"""vLLM Worker utility functions and helper classes

Contains:
- VllmEngineMonitor: Engine health monitoring
- ImageLoader: Multimodal image loading
- Environment variable setup functions
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


# ===== EngineDeadError unified handling =====


def handle_engine_dead_error(
    error: Exception,
    context: str,
    on_engine_dead: Callable[[], None] | None = None,
) -> None:
    """Unified handling of EngineDeadError

    Args:
        error: Caught exception
        context: Description of the context where error occurred
        on_engine_dead: Engine dead callback function
    """
    logger.error(f"vLLM EngineDeadError during {context}: {error}")
    if on_engine_dead is not None:
        logger.warning(f"Calling on_engine_dead callback from {context}...")
        try:
            on_engine_dead()
        except Exception as callback_error:
            logger.exception(f"Error in on_engine_dead callback: {callback_error}")


# Multimodal data dictionary keys
IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"


# ===== Utility functions =====


def lora_name_to_id(lora_name: str) -> int:
    """Generate deterministic integer ID from LoRA name

    Uses blake2b hash algorithm to generate 64-bit integer ID

    Args:
        lora_name: LoRA adapter name

    Returns:
        64-bit integer ID
    """
    # Use blake2b to generate 8-byte hash
    hash_digest = hashlib.blake2b(lora_name.encode("utf-8"), digest_size=8).digest()
    # Convert to unsigned 64-bit integer
    return int.from_bytes(hash_digest, byteorder="big", signed=False)


def _is_macos() -> bool:
    """Detect if running on macOS"""
    return platform.system() == "Darwin"


class VllmEngineMonitor:
    """vLLM engine monitor for collecting engine runtime statistics and continuous health checks

    Referencing Dynamo engine_monitor.py implementation, supports:
    1. Static configuration information retrieval
    2. Continuous background health checks
    3. Engine death detection and callback notification
    """

    # Health check interval (seconds)
    HEALTH_CHECK_INTERVAL = 2
    # Engine shutdown timeout (seconds)
    ENGINE_SHUTDOWN_TIMEOUT = 30

    def __init__(
        self,
        engine: "AsyncLLM",
        on_engine_dead: "Callable[[], None] | None" = None,
    ):
        """Initialize engine monitor

        Args:
            engine: vLLM AsyncLLM engine instance
            on_engine_dead: Callback function when engine dies (optional)
        """
        self.engine = engine
        self._on_engine_dead = on_engine_dead
        self._monitor_task: asyncio.Task | None = None
        self._is_healthy = True
        self._last_error: str | None = None

    def start_monitoring(
        self, on_engine_dead: "Callable[[], None] | None" = None
    ) -> None:
        """Start background health check task

        Args:
            on_engine_dead: Callback function when engine dies (optional, overrides constructor setting)
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
        """Stop background health check task"""
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            logger.info("VllmEngineMonitor stopped")

    async def _health_check_loop(self) -> None:
        """Continuous health check loop (referencing Dynamo engine_monitor.py)"""
        while True:
            try:
                # Call vLLM engine health check method
                await self.engine.check_health()
                self._is_healthy = True
                self._last_error = None
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

            except Exception as e:
                # Check if it's EngineDeadError
                error_name = type(e).__name__
                if error_name == "EngineDeadError" or "EngineDeadError" in str(type(e)):
                    logger.error(f"vLLM EngineDeadError detected: {e}")
                    self._is_healthy = False
                    self._last_error = str(e)

                    # Try to shutdown engine
                    await self._shutdown_engine()

                    # Call callback function
                    if self._on_engine_dead is not None:
                        logger.warning("Calling on_engine_dead callback...")
                        try:
                            self._on_engine_dead()
                        except Exception as callback_error:
                            logger.exception(
                                f"Error in on_engine_dead callback: {callback_error}"
                            )

                    # Exit health check loop
                    break
                else:
                    # Other exceptions, log but continue checking
                    logger.warning(f"Health check failed with non-fatal error: {e}")
                    self._is_healthy = False
                    self._last_error = str(e)
                    await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

            except asyncio.CancelledError:
                logger.debug("Health check loop cancelled")
                break

    async def _shutdown_engine(self) -> None:
        """Shutdown vLLM engine (with timeout protection)"""
        logger.warning("Initiating vLLM engine shutdown...")

        try:
            # Use asyncio.wait_for to add timeout protection
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
        """Return whether engine is healthy"""
        return self._is_healthy

    @property
    def last_error(self) -> str | None:
        """Return last error message"""
        return self._last_error

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache configuration information"""
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
        """Get model configuration information"""
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
        """Get engine health status"""
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
        """Destructor, ensure monitoring task is stopped"""
        self.stop_monitoring()


def _setup_macos_metal_env(
    mlx_device: str | None = None,
    metal_memory_fraction: float | None = None,
) -> None:
    """Setup macOS Metal/MLX environment variables

    Args:
        mlx_device: MLX device type ('gpu' or 'cpu'), default 'gpu'
        metal_memory_fraction: Metal memory usage ratio (0.0-1.0), default 0.8
    """
    if not _is_macos():
        return

    # Set MLX device
    if mlx_device is None:
        mlx_device = os.environ.get("VLLM_MLX_DEVICE", "gpu")

    if "VLLM_MLX_DEVICE" not in os.environ:
        os.environ["VLLM_MLX_DEVICE"] = mlx_device
        logger.info(f"Set VLLM_MLX_DEVICE={mlx_device} for macOS Metal support")

    # Set Metal memory usage ratio
    if metal_memory_fraction is None:
        metal_memory_fraction_str = os.environ.get("VLLM_METAL_MEMORY_FRACTION")
        if metal_memory_fraction_str:
            metal_memory_fraction = float(metal_memory_fraction_str)
        else:
            metal_memory_fraction = 0.8  # Default 80%

    if "VLLM_METAL_MEMORY_FRACTION" not in os.environ:
        os.environ["VLLM_METAL_MEMORY_FRACTION"] = str(metal_memory_fraction)
        logger.info(
            f"Set VLLM_METAL_MEMORY_FRACTION={metal_memory_fraction} for macOS Metal support"
        )


class ImageLoader:
    """Image loader, supports URL and Base64 encoded images"""

    def __init__(self, cache_size: int = 100):
        self._cache: dict[str, Image.Image] = {}
        self._cache_size = cache_size

    async def load_image(self, image_source: str) -> Image.Image:
        """Load image, supports Data URL (Base64) and HTTP(S) URL

        Args:
            image_source: Image source, can be data: URL or http(s): URL

        Returns:
            PIL Image object
        """
        # Check cache
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

        # HTTP URL download not yet supported, recommend converting to Base64 by frontend/Processor
        raise ValueError(f"Unsupported image source: {image_source[:20]}...")

    def _add_to_cache(self, key: str, image: Image.Image) -> None:
        """Add to cache, clean old entries if size limit exceeded"""
        if len(self._cache) >= self._cache_size:
            # Simple FIFO strategy
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = image
