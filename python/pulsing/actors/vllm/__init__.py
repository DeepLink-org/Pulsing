"""vLLM Worker - 基于 vLLM V1 引擎的高性能推理 Worker

参考 Dynamo 实现，支持：
1. Prefill/Decode 分离 (PD Disaggregation)
2. 多模态输入处理（图片）
3. KV Cache 管理和清理
4. LoRA 动态加载/卸载
5. OpenAI 兼容的文本输入输出模式
6. 引擎监控和健康检查
"""

from .worker import VllmWorker

__all__ = ["VllmWorker"]
