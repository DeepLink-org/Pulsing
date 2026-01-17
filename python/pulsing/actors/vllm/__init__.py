"""vLLM Worker - High-performance inference Worker based on vLLM V1 engine

Referencing Dynamo implementation, supports:
1. Prefill/Decode separation (PD Disaggregation)
2. Multimodal input processing (images)
3. KV Cache management and cleanup
4. LoRA dynamic loading/unloading
5. OpenAI-compatible text input/output mode
6. Engine monitoring and health checks
"""

from .worker import VllmWorker

__all__ = ["VllmWorker"]
