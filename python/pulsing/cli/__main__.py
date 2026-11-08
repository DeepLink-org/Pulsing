"""Pulsing CLI - Main entry point"""

import argparse
import json
import sys

from pulsing.cli.actor_argv import rewrite_actor_argv
from pulsing.cli.help_text import print_top_level_help


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


def _boolean(value: str) -> bool:
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _bool_option(parser, name: str, default: bool = False):
    flags = [f"--{name}"]
    if "_" in name:
        flags.append(f"--{name.replace('_', '-')}")
    parser.add_argument(*flags, nargs="?", const=True, type=_boolean, default=default)


def main(argv: list[str] | None = None):
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments in (["-h"], ["--help"], ["help"]):
        print_top_level_help()
        return

    if arguments[0] == "--" or (arguments[0].startswith("-") and "--" in arguments):
        from .launch import main as launch_main

        return launch_main(arguments)

    parser = argparse.ArgumentParser(prog="pulsing")
    sub = parser.add_subparsers(dest="command", required=True)
    from pulsing.cli.workspace_cmd import register_workspace_commands

    register_workspace_commands(sub)
    actors = sub.add_parser("actor", help="Start an actor service")
    actors.add_argument("actor_type")
    actors.add_argument("--addr")
    actors.add_argument("--seeds")
    actors.add_argument("--name", default="worker")
    actors.add_argument("--extra_kwargs", "--extra-kwargs", default="{}")
    actors.add_argument("-D", dest="overrides", action="append", default=[])
    actors.set_defaults(func=actor)

    observer = sub.add_parser("inspect", help="Observe a cluster via HTTP")
    observer.add_argument(
        "subcommand", choices=("cluster", "actors", "metrics", "watch")
    )
    observer.add_argument("--seeds")
    observer.add_argument("--endpoint")
    observer.add_argument("--timeout", type=float, default=10.0)
    observer.add_argument("--top", type=int)
    observer.add_argument("--filter")
    observer.add_argument("--interval", type=float, default=1.0)
    observer.add_argument(
        "--kind", choices=("all", "cluster", "actors", "metrics"), default="all"
    )
    observer.add_argument("--max_rounds", "--max-rounds", type=int)
    for flag in ("best_effort", "all_actors", "json", "detailed"):
        _bool_option(observer, flag)
    _bool_option(observer, "raw", default=True)
    observer.set_defaults(func=inspect)

    example = sub.add_parser("examples", help="Browse built-in examples")
    example.add_argument("name", nargs="?")
    example.add_argument("--name", dest="option_name")
    example.set_defaults(func=examples)

    args = parser.parse_args(rewrite_actor_argv(["pulsing", *arguments])[1:])
    if args.command in ("init", "history", "checkpoint", "rollback"):
        args.func(args)
        return
    options = vars(args).copy()
    func = options.pop("func")
    command = options.pop("command")
    if command == "actor":
        overrides = options.pop("overrides")
        try:
            kwargs = json.loads(options["extra_kwargs"])
            if not isinstance(kwargs, dict):
                raise ValueError("actor kwargs must be a JSON object")
            for override in overrides:
                prefix = "actor.extra_kwargs="
                if not override.startswith(prefix):
                    raise ValueError("only -D actor.extra_kwargs=JSON is supported")
                extra = json.loads(override[len(prefix) :])
                if not isinstance(extra, dict):
                    raise ValueError("actor kwargs must be a JSON object")
                kwargs.update(extra)
            options["extra_kwargs"] = kwargs
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
    elif command == "examples":
        options["name"] = options.pop("option_name") or options["name"]
    func(**options)


if __name__ == "__main__":
    main()
