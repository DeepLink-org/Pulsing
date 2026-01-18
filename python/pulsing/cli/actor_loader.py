"""Generic Actor loader - dynamically load and instantiate Actor classes"""

import importlib
import json
from typing import Any

from pulsing.actor import Actor


def load_actor_class(class_path: str) -> type[Actor]:
    """Load Actor class from module path

    Args:
        class_path: Full class path, e.g., 'pulsing.actors.worker.TransformersWorker'

    Returns:
        Actor class

    Raises:
        ImportError: If module or class cannot be imported
        ValueError: If class is not an Actor subclass
    """
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path '{class_path}'. Expected format: 'module.path.ClassName'\n"
            f"Example: pulsing.actors.worker.TransformersWorker"
        )

    # Split module path and class name
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid class path '{class_path}'. Expected format: 'module.path.ClassName'"
        )

    module_path, class_name = parts

    try:
        # Import module
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}': {e}\n"
            f"Make sure the module is installed and the path is correct."
        ) from e

    # Get class from module
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}'.\n"
            f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}"
        )

    actor_class = getattr(module, class_name)

    # Verify it's an Actor subclass
    if not isinstance(actor_class, type) or not issubclass(actor_class, Actor):
        raise ValueError(
            f"'{class_name}' is not an Actor subclass.\n"
            f"Expected a class that inherits from pulsing.actor.Actor"
        )

    return actor_class
