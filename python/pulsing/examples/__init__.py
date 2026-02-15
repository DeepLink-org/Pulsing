"""
pulsing.examples — Pulsing 内置示例集

每个子模块都是一个可独立运行的示例，同时也可被测试导入复用。
"""

import importlib
import inspect
from pathlib import Path

# 注册所有 example：模块名 → 一句话摘要
_EXAMPLES = {
    "counting_game": "Pulsing + Ray 分布式报数游戏",
}


def list_examples():
    """返回 [(name, summary, module_path)] 列表"""
    result = []
    examples_dir = Path(__file__).parent
    for name, summary in _EXAMPLES.items():
        filepath = examples_dir / f"{name}.py"
        result.append((name, summary, str(filepath)))
    return result


def get_example_detail(name):
    """返回 (summary, docstring, filepath)，找不到则返回 None"""
    if name not in _EXAMPLES:
        return None
    mod = importlib.import_module(f"pulsing.examples.{name}")
    filepath = inspect.getfile(mod)
    return (_EXAMPLES[name], (mod.__doc__ or "").strip(), filepath)
