"""Start and reuse the local mailbox sidecar on demand, without a service step."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pulsing as pul

from .actors import DIRECTORY_NAME, MailboxDirectory

ENDPOINT_FILE = "endpoint.json"


def seed_configuration(
    seeds: list[str], env_seed: str | None = None
) -> tuple[bool, list[str]]:
    """Explicit arguments win; SEED=1 hosts here, addresses supply bootstrap peers."""
    values = list(seeds)
    if not values and env_seed:
        values = [value.strip() for value in env_seed.split(",")]
    if any(not value.strip() for value in values):
        raise ValueError("seeds must not contain empty addresses")
    if "1" in values:
        if values != ["1"]:
            raise ValueError("SEED=1 cannot be combined with peer seed addresses")
        return True, []
    return False, values


def default_state_dir() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return base / "pulsing" / "mailbox"


@asynccontextmanager
async def host_runtime(
    addr: str,
    state_dir: str | Path,
    *,
    publish_endpoint: bool = True,
    seeds: list[str] | None = None,
    namespace: str | None = None,
):
    """Own the directory and listen as the seed in THIS process, not a sidecar."""
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".host.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await pul.init(addr=addr, seeds=seeds)
        try:
            name = (
                DIRECTORY_NAME if namespace is None else f"mail/instances/{namespace}"
            )
            directory = await MailboxDirectory.spawn(
                str(root), namespace=namespace, name=name
            )
            await directory.restore()
            if publish_endpoint:
                pending = root / f"{ENDPOINT_FILE}.tmp"
                pending.write_text(
                    json.dumps({"pid": os.getpid(), "addr": pul.get_system().addr}),
                    encoding="utf-8",
                )
                pending.replace(root / ENDPOINT_FILE)
            yield directory
        finally:
            if publish_endpoint:
                (root / ENDPOINT_FILE).unlink(missing_ok=True)
            await pul.shutdown()


def _running(root: Path) -> bool:
    with (root / ".host.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _endpoint(root: Path) -> str | None:
    if not _running(root):
        return None
    try:
        value = json.loads((root / ENDPOINT_FILE).read_text(encoding="utf-8"))
        os.kill(value["pid"], 0)
        return value["addr"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def ensure_host(state_dir: str | Path, addr: str, timeout: float = 20) -> str:
    """Return a live sidecar endpoint, serializing concurrent bootstrap attempts.

    The sidecar deliberately outlives MCP sessions: another local or remote
    consumer may still need it. Only the process we started may be terminated on
    startup failure. A seed-based join never calls this fallback.
    """
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with (root / ".startup.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for mailbox bootstrap lock")
                await asyncio.sleep(0.05)

        endpoint = _endpoint(root)
        if endpoint:
            host, _, port = addr.rpartition(":")
            live_host, _, live_port = endpoint.rpartition(":")
            if host != live_host or port not in ("0", live_port):
                raise ValueError(
                    f"mailbox state already serves {endpoint}, not {addr}; "
                    "use its address or a different state-dir"
                )
            return endpoint
        if _running(root):
            raise RuntimeError(
                f"mailbox state is owned by a host without a ready endpoint: {root}"
            )
        # An unlocked state directory cannot have a live sidecar. Discard stale
        # readiness before launching, including after an unclean host exit.
        (root / ENDPOINT_FILE).unlink(missing_ok=True)
        with (root / "host.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pulsing.mailbox",
                    "serve",
                    "--managed",
                    "--addr",
                    addr,
                    "--state-dir",
                    str(root),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                env=os.environ
                | {
                    "PULSING_TRACING_CONSOLE": "0",
                    "PULSING_PROBING_AUTO_TRACING": "0",
                },
            )
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"mailbox sidecar exited ({process.returncode}); see {root / 'host.log'}"
                    )
                endpoint = _endpoint(root)
                if endpoint:
                    return endpoint
                await asyncio.sleep(0.05)
            raise TimeoutError(
                f"mailbox sidecar did not become ready; see {root / 'host.log'}"
            )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)
            raise
