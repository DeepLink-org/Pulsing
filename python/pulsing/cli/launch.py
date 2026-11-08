"""``pulsing -- COMMAND``: one companion process for one external agent harness."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import secrets
import signal
import socket
import sys
import uuid
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path

from .network import Network, choose_network


def codex_command(command: list[str], url: str) -> list[str]:
    settings = {
        "mcp_servers.pulsing.url": url,
        "mcp_servers.pulsing.bearer_token_env_var": "PULSING_MCP_TOKEN",
        "mcp_servers.pulsing.required": True,
    }
    overrides = [
        arg
        for key, value in settings.items()
        for arg in ("-c", f"{key}={json.dumps(value)}")
    ]
    return [command[0], "--no-daemon", *overrides, *command[1:]]


@asynccontextmanager
async def mcp_endpoint(server, token: str, audit=None):
    import uvicorn
    from starlette.responses import Response

    class LocalServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            # The launcher, not the HTTP server, supervises the child.
            yield

    server.settings.stateless_http = True
    server.settings.json_response = True
    app = server.streamable_http_app()
    if audit is not None:
        app = audit.wrap(app)

    async def authenticated(scope, receive, send):
        if scope["type"] == "http":
            header = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(header, f"Bearer {token}".encode()):
                await Response(status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        http = LocalServer(
            uvicorn.Config(
                authenticated,
                log_level="error",
                access_log=False,
                timeout_graceful_shutdown=3,
            )
        )
        task = asyncio.create_task(http.serve(sockets=[listener]))
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while not http.started:
                if task.done():
                    await task
                    raise RuntimeError("MCP endpoint exited before readiness")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("MCP endpoint did not become ready")
                await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{listener.getsockname()[1]}/mcp", task
        finally:
            http.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def run_child(
    command: list[str], environment: dict[str, str], server_task
) -> int:
    detached = not sys.stdin.isatty()
    child = await asyncio.create_subprocess_exec(
        *command, env=environment, start_new_session=detached
    )
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    received = []

    def send(sig):
        try:
            if detached:
                os.killpg(child.pid, sig)
            else:
                child.send_signal(sig)
        except ProcessLookupError:
            pass

    # stdin/stdout/stderr and the foreground terminal are inherited unchanged.
    def forward(sig):
        received.append(sig)
        stopped.set()
        if child.returncode is None:
            # A terminal Ctrl-C already reaches both foreground processes.
            if sig != signal.SIGINT or detached:
                send(sig)

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    for sig in signals:
        loop.add_signal_handler(sig, forward, sig)
    wait = asyncio.create_task(child.wait())
    interrupt = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait(
            [wait, interrupt, server_task], return_when=asyncio.FIRST_COMPLETED
        )
        if server_task in done and wait not in done:
            raise RuntimeError("companion MCP endpoint stopped while agent was running")
        if interrupt in done and not wait.done():
            try:
                await asyncio.wait_for(asyncio.shield(wait), timeout=5)
            except TimeoutError:
                send(signal.SIGKILL)
        code = await wait
        return 128 + received[0] if received else (code if code >= 0 else 128 - code)
    finally:
        if child.returncode is None:
            send(signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(wait), timeout=5)
            except TimeoutError:
                send(signal.SIGKILL)
                await wait
        if detached:
            send(signal.SIGTERM)
        interrupt.cancel()
        await asyncio.gather(interrupt, return_exceptions=True)
        for sig, handler in previous.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)


@asynccontextmanager
async def companion_runtime(
    network: Network, state_dir: Path, instance: str, auto_seed: bool
):
    from pulsing.exceptions import PulsingRuntimeError
    from pulsing.mailbox.managed import host_runtime

    async with AsyncExitStack() as stack:

        async def start(selected):
            return await stack.enter_async_context(
                host_runtime(
                    selected.addr, state_dir, seeds=selected.peers, namespace=instance
                )
            )

        try:
            directory = await start(network)
        except (OSError, PulsingRuntimeError):
            if not auto_seed:
                raise
            # Another local launcher may bind the seed between our probe and
            # runtime startup. Re-plan once, before starting any target program.
            selected = choose_network([network.addr, *network.peers])
            if selected == network:
                raise
            network = selected
            directory = await start(network)
        yield directory, network


async def launch(
    command: list[str],
    *,
    network: Network,
    instance: str,
    state_dir: Path,
    adapter: str,
    auto_seed: bool = False,
    audit_log: Path | None = None,
) -> int:
    import pulsing as pul
    from pulsing.mailbox.companion import AgentClients, DIRECTORY_PREFIX
    from pulsing.mailbox.mcp import create_server

    os.environ["PULSING_TRACING_CONSOLE"] = "0"
    rules = Path(__file__).parents[1] / "mailbox" / "COMPANION.md"
    async with companion_runtime(network, state_dir, instance, auto_seed) as (
        directory,
        network,
    ):
        if network.peers and not network.is_seed:
            deadline = asyncio.get_running_loop().time() + 15
            while not any(
                entry["path"].startswith(DIRECTORY_PREFIX)
                and entry["path"] != DIRECTORY_PREFIX + instance
                for entry in await pul.get_system().all_named_actors()
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(
                        "no companion discovered at the configured seeds"
                    )
                await asyncio.sleep(0.05)
        server = create_server(
            instance,
            ["1"] if network.is_seed else network.peers,
            client_provider=AgentClients(directory),
            protocol_text=rules.read_text(encoding="utf-8"),
        )
        token = secrets.token_urlsafe(32)
        audit = None
        if audit_log is not None:
            from pulsing.mailbox.audit import MCPAudit

            audit = MCPAudit(audit_log, instance)
        async with mcp_endpoint(server, token, audit) as (url, server_task):
            environment = os.environ | {
                "PULSING_INSTANCE_ID": instance,
                "PULSING_ADDR": pul.get_system().addr,
                "PULSING_MCP_URL": url,
                "PULSING_MCP_TOKEN": token,
                "PULSING_PROTOCOL": str(rules),
            }
            print(
                f"Pulsing companion {instance} ready at {pul.get_system().addr}",
                file=sys.stderr,
            )
            if adapter == "codex":
                command = codex_command(command, url)
            return await run_child(command, environment, server_task)


def main(arguments: list[str]) -> None:
    from pulsing.exceptions import PulsingRuntimeError
    from pulsing.mailbox.actors import validate_identity
    from pulsing.mailbox.managed import default_state_dir, seed_configuration

    parser = argparse.ArgumentParser(
        prog="pulsing", usage="pulsing [launch options] -- COMMAND [ARGS...]"
    )
    parser.add_argument("--addr", default=os.environ.get("PULSING_ADDR") or None)
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument(
        "--instance",
        default=os.environ.get("PULSING_INSTANCE_ID") or uuid.uuid4().hex[:16],
    )
    parser.add_argument("--state-dir", default=os.environ.get("PULSING_STATE_DIR"))
    parser.add_argument("--audit-log", default=os.environ.get("PULSING_AUDIT_LOG"))
    parser.add_argument(
        "--adapter", choices=("auto", "codex", "generic"), default="auto"
    )
    separator = arguments.index("--")
    options = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1 :]
    if not command:
        parser.error("a command is required after --")
    try:
        validate_identity(options.instance)
        if len(options.instance) > 24 or "." in options.instance:
            raise ValueError(
                "instance identity must be at most 24 letters, digits, _ or -"
            )
        _, peers = seed_configuration(options.seed, os.environ.get("SEED"))
        network = choose_network(peers, options.addr)
        adapter = options.adapter
        if adapter == "auto":
            adapter = (
                "codex"
                if Path(command[0]).name in ("codex", "codex.exe")
                else "generic"
            )
        state = (
            Path(options.state_dir).expanduser()
            if options.state_dir
            else default_state_dir().parent / "agents" / options.instance
        )
        code = asyncio.run(
            launch(
                command,
                network=network,
                instance=options.instance,
                state_dir=state,
                adapter=adapter,
                auto_seed=network.is_seed and options.addr is None and bool(peers),
                audit_log=Path(options.audit_log) if options.audit_log else None,
            )
        )
    except FileNotFoundError as exc:
        print(f"pulsing: {exc}", file=sys.stderr)
        code = 127
    except KeyboardInterrupt:
        code = 130
    except ModuleNotFoundError as exc:
        if exc.name not in ("mcp", "uvicorn"):
            raise
        print("pulsing -- requires: pip install 'pulsing[mailbox]'", file=sys.stderr)
        code = 1
    except (
        ValueError,
        OSError,
        RuntimeError,
        TimeoutError,
        PulsingRuntimeError,
    ) as exc:
        print(f"pulsing: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
