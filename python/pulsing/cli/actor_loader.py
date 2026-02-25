"""Generic Actor loader - dynamically load Actor or @remote classes"""

import importlib
from typing import Any, Union

from pulsing.core import Actor


def load_actor_class(class_path: str) -> Union[type[Actor], Any]:
    """Load Actor 或 @remote 类

    Args:
        class_path: 完整类路径，如 'pulsing.serving.worker.TransformersWorker'

    Returns:
        Actor 子类或带 .spawn 的 @remote 类

    Raises:
        ImportError: 模块/类无法导入
        ValueError: 既非 Actor 子类也非 @remote 类（无 .spawn）
    """
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path '{class_path}'. Expected format: 'module.path.ClassName'\n"
            f"Example: pulsing.serving.worker.TransformersWorker"
        )

    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid class path '{class_path}'. Expected format: 'module.path.ClassName'"
        )

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}': {e}\n"
            f"Make sure the module is installed and the path is correct."
        ) from e

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}'.\n"
            f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}"
        )

    actor_class = getattr(module, class_name)

    # Actor 子类 或 @remote 类（有 .spawn）
    if isinstance(actor_class, type) and issubclass(actor_class, Actor):
        return actor_class
    if hasattr(actor_class, "spawn") and callable(getattr(actor_class, "spawn")):
        return actor_class

    raise ValueError(
        f"'{class_name}' is not an Actor subclass nor a @remote class (no .spawn).\n"
        f"Use pulsing.core.Actor or @pulsing.remote."
    )
