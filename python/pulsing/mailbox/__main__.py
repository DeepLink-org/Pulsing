"""Host actor mailboxes, or expose one registered identity over MCP stdio."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys

import pulsing as pul


async def serve(
    addr: str, state_dir: str, identity: str, managed: bool = False
) -> None:
    from .managed import host_runtime

    async with host_runtime(addr, state_dir, publish_endpoint=managed) as directory:
        roles = await directory.roles()
        if not managed and identity not in {role["identity"] for role in roles}:
            await directory.register(identity, "router")
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        print(f"Mailbox host ready at {pul.get_system().addr}", flush=True)
        await stopped.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("serve", help="run the trusted mailbox host")
    host.add_argument("--addr", default="127.0.0.1:8100")
    host.add_argument("--state-dir", required=True)
    host.add_argument("--identity", default="router")
    host.add_argument("--managed", action="store_true", help=argparse.SUPPRESS)
    mcp = commands.add_parser(
        "mcp", help="bind MCP stdio tools to one registered identity"
    )
    mcp.add_argument("--identity", required=True)
    mcp.add_argument("--seed", action="append", default=[])
    mcp.add_argument(
        "--managed", action="store_true", help="automatically bootstrap or join"
    )
    mcp.add_argument("--state-dir", help="persistent local sidecar state")
    mcp.add_argument("--kind", default="worker")
    mcp.add_argument(
        "--addr",
        default=None,
        help="local actor address; use a reachable private IP for multi-machine use",
    )
    codex = commands.add_parser(
        "codex-plugin", help="generate an identity-bound Codex plugin and marketplace"
    )
    codex.add_argument("--output", required=True, help="new marketplace directory")
    codex.add_argument("--identity", required=True)
    codex.add_argument("--seed", action="append", default=[])
    codex.add_argument("--addr", default=None)
    codex.add_argument("--state-dir", help="persistent local sidecar state")
    codex.add_argument("--kind", default="worker")
    codex.add_argument("--python", default=sys.executable, help="MCP Python executable")
    args = parser.parse_args()
    if args.command == "serve":
        asyncio.run(serve(args.addr, args.state_dir, args.identity, args.managed))
    elif args.command == "codex-plugin":
        from .codex import generate_plugin

        try:
            root = generate_plugin(
                args.output,
                args.identity,
                args.seed,
                args.addr,
                args.python,
                state_dir=args.state_dir,
                kind=args.kind,
            )
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Codex marketplace generated at {root}")
        print(
            f"Identity: {args.identity}; mailbox startup and registration are automatic."
        )
    else:
        try:
            from .mcp import run
        except ModuleNotFoundError as exc:
            if exc.name == "mcp":
                parser.error("MCP support requires: pip install 'pulsing[mailbox]'")
            raise

        if not args.managed and not args.seed and not os.environ.get("SEED"):
            parser.error("mcp requires --seed or --managed")
        run(
            args.identity,
            args.seed,
            args.addr or os.environ.get("PULSING_ADDR") or "127.0.0.1:0",
            managed=args.managed,
            state_dir=args.state_dir,
            kind=args.kind,
        )


if __name__ == "__main__":
    main()
