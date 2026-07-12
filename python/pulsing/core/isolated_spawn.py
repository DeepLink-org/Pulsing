"""Spawn an actor in a child OS process (out-cluster Connect + IPC bridge on parent)."""

from __future__ import annotations

import asyncio
import os
import pickle
import socket
import sys
import tempfile
from typing import Any

from pulsing.cluster_spawn import normalize_seed_for_local_child
from pulsing.core.isolated_bridge import (
    IsolatedBridgeActor,
    IsolatedSpawnHandle,
    wait_child_ready,
)


async def spawn_isolated_actor(
    system: Any,
    actor: Any,
    *,
    name: str | None,
    public: bool,
    restart_policy: str,
    max_restarts: int,
    min_backoff: float,
    max_backoff: float,
) -> IsolatedSpawnHandle:
    from pulsing.core.remote import _WrappedActor

    if restart_policy != "never":
        raise ValueError(
            "isolated spawn (new_process=True with an actor) only supports restart_policy='never' in MVP"
        )

    wrapped = actor if isinstance(actor, _WrappedActor) else _WrappedActor(actor)

    fd, pickle_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(wrapped, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
        raise

    gateway_addr = normalize_seed_for_local_child(system.addr)
    host = "127.0.0.1"
    loop = asyncio.get_running_loop()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proc: asyncio.subprocess.Process | None = None
    try:
        srv.bind((host, 0))
        srv.listen(1)
        srv.setblocking(False)
        port = srv.getsockname()[1]

        env = os.environ.copy()
        env.update(
            {
                "PULSING_ISOLATED_WORKER": "1",
                "PULSING_GATEWAY_ADDR": gateway_addr,
                "PULSING_IPC_HOST": host,
                "PULSING_IPC_PORT": str(port),
                "PULSING_ISOLATED_PICKLE_PATH": pickle_path,
            }
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pulsing.isolated_worker",
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            client_sock, _ = await asyncio.wait_for(
                loop.sock_accept(srv), timeout=120.0
            )
            client_sock.setblocking(False)
            reader, writer = await asyncio.open_connection(sock=client_sock)
            await wait_child_ready(reader)
        except Exception:
            if proc is not None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            raise
        finally:
            srv.close()

        try:
            os.unlink(pickle_path)
        except OSError:
            pass

        bridge = IsolatedBridgeActor(reader, writer, pickle_path=pickle_path)
        try:
            ref = await system.spawn(
                bridge,
                name=name,
                public=public,
                restart_policy="never",
                max_restarts=max_restarts,
                min_backoff=min_backoff,
                max_backoff=max_backoff,
            )
        except Exception:
            if proc is not None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    proc.kill()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            raise

        assert proc is not None
        return IsolatedSpawnHandle(ref=ref, process=proc)
    except Exception:
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
        raise
