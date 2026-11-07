"""Host actor mailboxes, or expose one registered identity over MCP stdio."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import signal
from pathlib import Path

import pulsing as pul

from .actors import DIRECTORY_NAME, MailboxDirectory


async def serve(addr: str, state_dir: str, identity: str) -> None:
    root = Path(state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Only one host may own this state directory. A process crash releases flock.
    with (root / ".host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await pul.init(addr=addr)
        try:
            directory = await MailboxDirectory.spawn(str(root), name=DIRECTORY_NAME)
            await directory.restore()
            roles = await directory.roles()
            if identity not in {role["identity"] for role in roles}:
                await directory.register(identity, "router")
            stopped = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stopped.set)
            print(f"Mailbox host ready at {pul.get_system().addr}", flush=True)
            await stopped.wait()
        finally:
            await pul.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("serve", help="run the trusted mailbox host")
    host.add_argument("--addr", default="127.0.0.1:8100")
    host.add_argument("--state-dir", required=True)
    host.add_argument("--identity", default="router")
    mcp = commands.add_parser(
        "mcp", help="bind MCP stdio tools to one registered identity"
    )
    mcp.add_argument("--identity", required=True)
    mcp.add_argument("--seed", required=True, action="append")
    args = parser.parse_args()
    if args.command == "serve":
        asyncio.run(serve(args.addr, args.state_dir, args.identity))
    else:
        try:
            from .mcp import run
        except ModuleNotFoundError as exc:
            if exc.name == "mcp":
                parser.error("MCP support requires: pip install 'pulsing[mailbox]'")
            raise

        run(args.identity, args.seed)


if __name__ == "__main__":
    main()
