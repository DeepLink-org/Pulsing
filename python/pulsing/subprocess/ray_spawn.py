"""Ray-based resource scheduling for ProcessActor.

When resources is specified, a Ray Actor bootstraps Pulsing on a
resource-matching node, spawns ProcessActor there, and the local node
resolves it via system.resolve_named() for transparent interaction.

Requires: pip install 'ray[default]'
"""

from __future__ import annotations

import asyncio
import socket
import uuid
from typing import Any


def _get_public_or_local_ip() -> str:
    """Get public IP if available, otherwise local network IP.

    Returns:
        str: Public IP address or local network IP (e.g., 192.168.x.x)
    """
    # Try to get public IP by checking internet connectivity
    try:
        # Connect to a public DNS server to determine outbound IP
        # This doesn't actually send data, just determines routing
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(1)
            # Google's public DNS
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]

            # Check if this is a public IP (not private ranges)
            # Private IP ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
            parts = local_ip.split(".")
            if len(parts) == 4:
                first = int(parts[0])
                second = int(parts[1])

                # Check if it's NOT a private IP
                if not (
                    first == 10
                    or (first == 172 and 16 <= second <= 31)
                    or (first == 192 and second == 168)
                    or first == 127  # localhost
                ):
                    # This appears to be a public IP
                    return local_ip

            # If we got here, it's a private/local network IP
            return local_ip
    except Exception:
        pass

    # Fallback: get hostname-based IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip and not local_ip.startswith("127."):
            return local_ip
    except Exception:
        pass

    # Last resort: return localhost
    return "127.0.0.1"


def _replace_addr_with_routable_ip(addr: str, *, force_local: bool = False) -> str:
    """Replace the host in addr with public or local network IP.

    Args:
        addr: Address string in format "host:port"
        force_local: If True, keep localhost for local testing scenarios

    Returns:
        str: Address with routable IP "ip:port"
    """
    if ":" not in addr:
        return addr

    host, port = addr.rsplit(":", 1)

    # If host is already a specific IP (not 0.0.0.0 or bind-all), keep it
    # For local testing (127.0.0.1), check force_local flag
    if host in ("localhost", "127.0.0.1"):
        if force_local:
            return addr
        # For single-machine Ray testing, keep localhost
        try:
            import ray

            if ray.is_initialized():
                # Check if Ray cluster is local (single node)
                nodes = ray.nodes()
                alive_nodes = [n for n in nodes if n.get("Alive", False)]
                if len(alive_nodes) <= 1:
                    # Single-node Ray cluster, keep localhost
                    return addr
        except Exception:
            pass

    if host not in ("0.0.0.0", "localhost", "127.0.0.1", "::"):
        # Check if it's already a routable IP
        try:
            parts = host.split(".")
            if len(parts) == 4:
                first = int(parts[0])
                # If it's not a bind-all address, assume it's already correct
                if first != 0:
                    return addr
        except (ValueError, IndexError):
            # Not a valid IPv4, return as-is
            return addr

    # Replace with routable IP
    routable_ip = _get_public_or_local_ip()
    return f"{routable_ip}:{port}"


def _build_pulsing_node_actor_cls():
    """Lazily build the Ray Actor class to avoid import-time ray dependency."""
    import ray

    @ray.remote
    class PulsingNodeActor:
        """Ray Actor that bootstraps Pulsing on a remote node and spawns a ProcessActor."""

        def __init__(
            self,
            seed_addr: str,
            actor_name: str,
            process_args: Any,
            process_kwargs: dict,
        ):
            self._seed_addr = seed_addr
            self._actor_name = actor_name
            self._process_args = process_args
            self._process_kwargs = process_kwargs

        async def start(self) -> dict:
            """Initialize Pulsing, spawn ProcessActor, return discovery info."""
            import ray as _ray

            from pulsing.core import init
            from .process import ProcessActor

            # Determine node IP from Ray runtime context
            ctx = _ray.get_runtime_context()
            node_id = ctx.get_node_id()
            node_ip = None
            for node in _ray.nodes():
                if node["NodeID"] == node_id and node["Alive"]:
                    node_ip = node["NodeManagerAddress"]
                    break
            if node_ip is None:
                raise RuntimeError("Cannot determine Ray node IP")

            # Bootstrap Pulsing on this node (port 0 = OS-assigned)
            system = await init(addr=f"{node_ip}:0", seeds=[self._seed_addr])

            # Spawn ProcessActor locally on this node with public=True
            # to ensure it's globally discoverable via gossip
            await ProcessActor.spawn(
                self._process_args,
                name=self._actor_name,
                placement="local",
                public=True,
                **self._process_kwargs,
            )

            return {
                "actor_name": self._actor_name,
                "node_id": system.node_id.id,
            }

        async def shutdown(self):
            """Shutdown Pulsing on this node."""
            from pulsing.core import shutdown

            await shutdown()

    return PulsingNodeActor


_PulsingNodeActorCls = None


def _get_pulsing_node_actor_cls():
    global _PulsingNodeActorCls
    if _PulsingNodeActorCls is None:
        _PulsingNodeActorCls = _build_pulsing_node_actor_cls()
    return _PulsingNodeActorCls


async def ray_spawn_process_actor(
    system,
    args: Any,
    actor_name: str | None,
    resources: dict,
    **process_kwargs,
):
    """Spawn a ProcessActor on a Ray-scheduled node matching resource requirements.

    Args:
        system: The local ActorSystem (must have a routable address).
        args: Command args for ProcessActor (same as subprocess).
        actor_name: Name for the remote ProcessActor. Auto-generated if None.
        resources: Resource dict passed to Ray Actor .options() (e.g. {"num_cpus": 1}).
        **process_kwargs: Extra kwargs forwarded to ProcessActor (stdin, stdout, etc.).

    Returns:
        ActorProxy pointing to the remote ProcessActor.
    """
    try:
        import ray
    except ImportError:
        raise ImportError(
            "ray_resources requires Ray. Install with: pip install 'ray[default]'"
        )

    if not ray.is_initialized():
        ray.init()

    # Guard: system must have a routable address for cluster communication
    addr = str(system.addr) if system.addr else None
    if not addr:
        raise RuntimeError(
            "resources requires a routable Pulsing address. "
            "Call pulsing.init(addr='0.0.0.0:PORT') before using resources."
        )

    # Replace bind-all addresses (0.0.0.0) with public or local network IP
    addr = _replace_addr_with_routable_ip(addr)

    if actor_name is None:
        actor_name = f"actors/ray_process_{uuid.uuid4().hex[:12]}"
    elif "/" not in actor_name:
        actor_name = f"actors/{actor_name}"

    PulsingNodeActor = _get_pulsing_node_actor_cls()

    # Create Ray Actor with requested resources
    ray_actor = PulsingNodeActor.options(**resources).remote(
        seed_addr=addr,
        actor_name=actor_name,
        process_args=args,
        process_kwargs=process_kwargs,
    )

    # Start Pulsing + ProcessActor on the remote node
    info = await ray_actor.start.remote()

    # Resolve the remote ProcessActor via Pulsing cluster
    # Add retry logic to handle gossip propagation delay
    from .process import ProcessActor

    max_retries = 10
    retry_delay = 0.5
    last_error = None

    proxy = None
    for attempt in range(max_retries):
        try:
            proxy = await ProcessActor.resolve(
                actor_name, node_id=info["node_id"], timeout=5
            )
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Failed to resolve ProcessActor '{actor_name}' after {max_retries} attempts"
                ) from last_error

    if proxy is None:
        raise RuntimeError(f"Failed to resolve ProcessActor '{actor_name}'")

    # Store Ray Actor handle on proxy to prevent GC
    proxy._ray_node_actor = ray_actor  # type: ignore[attr-defined]

    return proxy


async def cleanup_ray_actor(proxy_or_handle) -> None:
    """Gracefully shut down a PulsingNodeActor Ray actor.

    Accepts either a proxy with ``_ray_node_actor`` attribute or a raw Ray
    actor handle.  Always safe to call — never raises.
    """
    try:
        import ray
    except ImportError:
        return

    # Extract handle from proxy or use directly
    if hasattr(proxy_or_handle, "_ray_node_actor"):
        handle = getattr(proxy_or_handle, "_ray_node_actor", None)
        proxy_or_handle._ray_node_actor = None  # prevent double-cleanup
    else:
        handle = proxy_or_handle

    if handle is None:
        return

    # Graceful shutdown with timeout
    try:
        await asyncio.wait_for(handle.shutdown.remote(), timeout=5.0)
    except Exception:
        pass

    # Final cleanup — safe on already-dead actors
    try:
        ray.kill(handle)
    except Exception:
        pass
