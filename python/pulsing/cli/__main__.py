"""Pulsing CLI - Main entry point"""

import hyperparameter as hp

from pulsing.cli.actor_argv import rewrite_actor_argv
from pulsing.cli.help_text import print_top_level_help


@hp.param("actor")
def actor(
    actor_type: str,  # Positional argument: full class path (e.g., 'my_app.Counter')
    addr: str | None = None,
    seeds: str | None = None,
    name: str = "worker",  # Actor name (default: "worker")
    extra_kwargs: (
        dict | None
    ) = None,  # Additional arguments for Actor constructor (--key value from CLI)
):
    r"""
    Start an Actor-based service.

    This command starts actors based on the Pulsing Actor System.

    Actor type must be a full class path:
    - Format: 'module.path.ClassName'
    - Example: 'my_app.Counter'
    - Example: 'my_module.my_actor.MyCustomActor'

    Parameter separation (avoids name collision):
    - Actor-level (process/cluster): --addr, --seeds, --name. Pass before \"--\".
    - Actor constructor args: pass after \"--\" so they never collide with --addr/--seeds/--name.

    Note: To list actors, use 'pulsing inspect actors' instead.

    Args:
        actor_type: Full class path (positional), e.g. my_app.Counter
        addr: Bind address (e.g. 0.0.0.0:8000)
        seeds: Comma-separated seed nodes (e.g. 192.168.1.1:8000,192.168.1.2:8000)
        name: Actor name (default: worker). Use different names for multiple workers.
        extra_kwargs: Constructor arguments. Pass after \"--\" or via -D actor.extra_kwargs='{...}'.

    Examples:
        # Actor-level before \"--\", constructor args after (no collision)
        pulsing actor my_app.Counter --addr 0.0.0.0:8000 --name counter -- --initial 10

        # Multiple workers
        pulsing actor my_app.Counter --name counter-1 --seeds 127.0.0.1:8000
        pulsing actor my_app.Counter --name counter-2 --seeds 127.0.0.1:8000
    """
    from .actors import start_generic_actor

    # Check for deprecated 'list' subcommand
    if actor_type == "list":
        print("Error: 'pulsing actor list' has been removed.")
        print("Use 'pulsing inspect actors' instead:")
        print("  pulsing inspect actors --endpoint 127.0.0.1:8000")
        print("  pulsing inspect actors --seeds 127.0.0.1:8000")
        return

    # Check if actor_type is a valid class path (must contain dots)
    if "." not in actor_type:
        raise ValueError(
            f"Error: Actor type must be a full class path (e.g., 'my_app.Counter').\n"
            f"Received: '{actor_type}'\n"
            f"Example: pulsing actor my_app.Counter --name counter"
        )

    # Parse seeds
    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]

    # extra_kwargs may be str when passed via -D actor.extra_kwargs='{...}'
    import json

    kwargs = extra_kwargs or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs) if kwargs.strip() else {}
    # Start generic Actor class
    start_generic_actor(
        actor_type=actor_type,
        addr=addr,
        seeds=seed_list,
        name=name,
        extra_kwargs=kwargs,
    )


@hp.param("inspect")
def inspect(
    subcommand: str,  # Positional argument: cluster, actors, metrics, watch
    seeds: str | None = None,
    # Common options
    timeout: float = 10.0,
    best_effort: bool = False,
    # cluster subcommand options (no specific options yet)
    # actors subcommand options
    top: int | None = None,
    filter: str | None = None,
    all_actors: bool = False,
    endpoint: str | None = None,  # Single node endpoint (alternative to seeds)
    json: bool = False,  # JSON output format
    detailed: bool = False,  # Show detailed info (class, module) per node
    # metrics subcommand options
    raw: bool = True,
    # watch subcommand options
    interval: float = 1.0,
    kind: str = "all",  # cluster, actors, metrics, all
    max_rounds: int | None = None,
):
    """
    Inspect the actor system state (observer mode via HTTP API).

    This command uses lightweight HTTP observer mode - does NOT join the gossip cluster.

    Args:
        subcommand: Subcommand to run. Options: 'cluster', 'actors', 'metrics', 'watch' (required)
        seeds: Comma-separated list of seed nodes (required)
        timeout: Request timeout in seconds (default: 10.0)
        best_effort: Continue even if some nodes fail (default: False)
        top: (actors only) Show top N actors by instance count
        filter: (actors only) Filter actor names by substring
        all_actors: (actors only) Include internal/system actors
        raw: (metrics only) Output raw metrics (default: True). If False, show summary only
        interval: (watch only) Refresh interval in seconds (default: 1.0)
        kind: (watch only) What to watch: 'cluster', 'actors', 'metrics', 'all' (default: 'all')
        max_rounds: (watch only) Maximum number of refresh rounds (None = infinite)

    Examples:
        # Inspect cluster members
        pulsing inspect cluster --seeds 127.0.0.1:8000

        # Inspect actors distribution
        pulsing inspect actors --seeds 127.0.0.1:8000 --top 10

        # Inspect metrics
        pulsing inspect metrics --seeds 127.0.0.1:8000 --raw False

        # Watch cluster changes
        pulsing inspect watch --seeds 127.0.0.1:8000 --interval 2.0 --kind cluster
    """
    from .inspect import (
        inspect_actors,
        inspect_cluster,
        inspect_metrics,
        inspect_watch,
    )

    seed_list = []
    if seeds:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]

    if subcommand == "cluster":
        if not seed_list:
            print("Error: --seeds is required for 'inspect cluster' command")
            return
        inspect_cluster(seed_list, timeout=timeout, best_effort=best_effort)
    elif subcommand == "actors":
        if endpoint and seed_list:
            print("Error: Cannot specify both --endpoint and --seeds.")
            print("Use --endpoint for single node, --seeds for cluster.")
            return
        inspect_actors(
            seeds=seed_list if not endpoint else None,
            endpoint=endpoint,
            timeout=timeout,
            best_effort=best_effort,
            top=top,
            filter=filter,
            all_actors=all_actors,
            json_output=json,
            detailed=detailed,
        )
    elif subcommand == "metrics":
        inspect_metrics(seed_list, timeout=timeout, best_effort=best_effort, raw=raw)
    elif subcommand == "watch":
        inspect_watch(
            seed_list,
            timeout=timeout,
            best_effort=best_effort,
            interval=interval,
            kind=kind,
            max_rounds=max_rounds,
        )
    else:
        print(f"Error: Unknown subcommand '{subcommand}'")
        print("Supported subcommands: cluster, actors, metrics, watch")


@hp.param("examples")
def examples(name: str | None = None):
    """
    List or view Pulsing built-in examples.

    Lists all available examples when called without arguments;
    shows detailed description, usage, and source path when given a name.

    Args:
        name: Example name (optional). If omitted, lists all examples.

    Examples:
        # List all examples
        pulsing examples

        # View details of a specific example
        pulsing examples counting_game
    """
    from pulsing.examples import get_example_detail, list_examples

    if name is None:
        all_examples = list_examples()
        if not all_examples:
            print("No examples available.")
            return
        print("Available examples:\n")
        max_name_len = max(len(n) for n, _, _ in all_examples)
        for n, summary, filepath in all_examples:
            print(f"  {n:<{max_name_len}}  {summary}")
        print("\nUse 'pulsing examples <name>' for details.")
        return

    detail = get_example_detail(name)
    if detail is None:
        print(f"Unknown example: '{name}'")
        print("Use 'pulsing examples' to see all available examples.")
        return

    summary, docstring, filepath = detail
    print(f"{'=' * 60}")
    print(f"  {summary}")
    print(f"{'=' * 60}\n")
    if docstring:
        print(docstring)
        print()
    print(f"Source path:\n  {filepath}\n")
    print(f"Quick run:\n  python -m pulsing.examples.{name}")


def main():
    import argparse
    import sys

    workspace_cmds = {"init", "history", "checkpoint", "rollback"}
    if len(sys.argv) >= 2 and sys.argv[1] in workspace_cmds:
        parser = argparse.ArgumentParser(prog="pulsing")
        sub = parser.add_subparsers(dest="command", required=True)
        from pulsing.cli.workspace_cmd import register_workspace_commands

        register_workspace_commands(sub)
        args = parser.parse_args(sys.argv[1:])
        args.func(args)
        return

    if len(sys.argv) == 1 or (
        len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help", "help")
    ):
        print_top_level_help()
        return

    if (
        len(sys.argv) >= 3
        and sys.argv[1] == "examples"
        and not sys.argv[2].startswith("-")
    ):
        sys.argv = [sys.argv[0], "examples", "--name", sys.argv[2]] + sys.argv[3:]

    sys.argv = rewrite_actor_argv(sys.argv)

    hp.launch()


if __name__ == "__main__":
    main()
