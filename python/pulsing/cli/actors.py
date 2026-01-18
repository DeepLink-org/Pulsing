"""Pulsing CLI - Actor commands"""

import uvloop

from .actor_loader import load_actor_class


def start_generic_actor(
    actor_type: str,
    addr: str | None,
    seeds: list[str],
    name: str = "worker",
    extra_kwargs: dict | None = None,
):
    """Start a generic Actor class by full module path

    Args:
        actor_type: Full class path, e.g., 'pulsing.actors.worker.TransformersWorker'
        addr: Actor System bind address
        seeds: List of seed node addresses
        name: Actor name
        extra_kwargs: Additional CLI arguments to pass to Actor constructor
    """
    import inspect
    from pulsing.actor.helpers import spawn_and_run

    print(f"Loading Actor class: {actor_type}")

    # Load Actor class
    try:
        actor_class = load_actor_class(actor_type)
    except (ImportError, ValueError, AttributeError) as e:
        print(f"Error: {e}")
        return

    print(f"  Class: {actor_class.__name__}")
    print(f"  Module: {actor_class.__module__}")

    # Get Actor constructor signature
    try:
        actor_sig = inspect.signature(actor_class.__init__)
    except Exception as e:
        print(f"Error: Cannot inspect Actor constructor: {e}")
        return

    # Parameters that should NOT be passed to Actor constructor
    cli_only_params = {
        "actor_type",
        "addr",
        "seeds",
        "name",
        "self",
    }

    # Get Actor constructor parameter names
    actor_params = set(actor_sig.parameters.keys()) - {"self"}

    # Extract Actor constructor parameters from extra_kwargs
    constructor_kwargs = {}

    # Filter extra_kwargs to only include Actor constructor parameters
    if extra_kwargs:
        for param_name, value in extra_kwargs.items():
            # Convert kebab-case to snake_case for matching
            snake_case_name = param_name.replace("-", "_")

            # Check if this parameter matches Actor constructor
            if (
                snake_case_name in actor_params
                and snake_case_name not in cli_only_params
            ):
                # Get type hint if available for conversion
                param = actor_sig.parameters.get(snake_case_name)
                if param and param.annotation != inspect.Parameter.empty:
                    try:
                        # Type conversion based on annotation
                        annotation = param.annotation
                        if annotation is bool and isinstance(value, str):
                            value = value.lower() in ("true", "1", "yes", "on")
                        elif annotation is int and isinstance(value, str):
                            value = int(value)
                        elif annotation is float and isinstance(value, str):
                            value = float(value)
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

                constructor_kwargs[snake_case_name] = value

    # Show constructor parameters
    if constructor_kwargs:
        print("  Constructor parameters:")
        for key, value in constructor_kwargs.items():
            # Truncate long values for display
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"    {key}: {value_str}")
    else:
        print("  No constructor parameters provided (using defaults)")

    print(f"  Actor name: {name}")
    if seeds:
        print(f"  Seeds: {', '.join(seeds)}")
    if addr:
        print(f"  Address: {addr}")

    async def run():
        try:
            # Instantiate Actor
            actor_instance = actor_class(**constructor_kwargs)
        except TypeError as e:
            print(f"\nError: Failed to instantiate {actor_class.__name__}")
            print(f"  {e}")
            print("\nHint: Check the constructor signature and provided parameters.")
            required_params = [
                name
                for name, param in actor_sig.parameters.items()
                if name != "self" and param.default == inspect.Parameter.empty
            ]
            if required_params:
                print(f"  Required parameters: {', '.join(required_params)}")
            print(f"  All constructor parameters: {', '.join(actor_params)}")
            print(f"  Provided parameters: {', '.join(constructor_kwargs.keys())}")
            return
        except Exception as e:
            print(f"\nError: Failed to create Actor instance: {e}")
            import traceback

            traceback.print_exc()
            return

        # Spawn and run
        await spawn_and_run(
            actor_instance,
            name=name,
            addr=addr,
            seeds=seeds if seeds else None,
            public=True,
        )

    uvloop.run(run())
