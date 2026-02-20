"""
pulsing.examples — Pulsing built-in examples collection

Each submodule is a standalone runnable example, also importable for testing.
"""

import importlib
import inspect
from pathlib import Path

# Register all examples: module name -> one-line summary
_EXAMPLES = {
    "counting_game": "Pulsing + Ray distributed counting game",
}


def list_examples():
    """Return [(name, summary, module_path)] list"""
    result = []
    examples_dir = Path(__file__).parent
    for name, summary in _EXAMPLES.items():
        filepath = examples_dir / f"{name}.py"
        result.append((name, summary, str(filepath)))
    return result


def get_example_detail(name):
    """Return (summary, docstring, filepath), or None if not found"""
    if name not in _EXAMPLES:
        return None
    mod = importlib.import_module(f"pulsing.examples.{name}")
    filepath = inspect.getfile(mod)
    return (_EXAMPLES[name], (mod.__doc__ or "").strip(), filepath)
