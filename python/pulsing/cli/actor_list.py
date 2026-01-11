"""Actor list command implementation

Query actors from remote actor systems via simple HTTP API.

Usage:
    # Query single endpoint
    pulsing actor list --endpoint 127.0.0.1:8000

    # Query cluster
    pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001

This implementation uses direct HTTP/2 requests instead of joining the gossip cluster,
making it a lightweight observer tool.
"""

import asyncio
import json
import subprocess

MAX_NODES_DISPLAY = 64  # Maximum number of nodes to display
HTTP_TIMEOUT = 10  # seconds


def http_get_sync(url: str) -> dict | list | None:
    """Make HTTP/2 GET request using curl (which supports h2c)"""
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",  # Silent
                "--http2-prior-knowledge",  # Use h2c (HTTP/2 over cleartext)
                "-m",
                str(HTTP_TIMEOUT),  # Timeout
                url,
            ],
            capture_output=True,
            text=True,
            timeout=HTTP_TIMEOUT + 2,
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
        return None
    except Exception:
        return None


async def query_single_endpoint(
    endpoint: str, all_actors: bool, output_format: str
) -> bool:
    """Query a single actor system endpoint via HTTP API"""

    # Ensure endpoint has port
    if ":" not in endpoint:
        endpoint = f"{endpoint}:8000"

    # Ensure http:// prefix
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"

    print(f"Connecting to {endpoint}...")

    # Get actors list
    url = f"{endpoint}/actors"
    if all_actors:
        url += "?all=true"

    actors = http_get_sync(url)
    if actors is None:
        print(f"Error: Cannot connect to {endpoint}")
        return False

    print(f"Connected to {endpoint}")
    print()

    _print_output(actors, output_format)
    return True


async def query_cluster(seeds: list[str], all_actors: bool, output_format: str) -> bool:
    """Query all nodes in a cluster via HTTP API"""

    # Normalize seeds
    normalized_seeds = []
    for seed in seeds:
        if ":" not in seed:
            seed = f"{seed}:8000"
        if not seed.startswith("http"):
            seed = f"http://{seed}"
        normalized_seeds.append(seed)

    print(f"Connecting to cluster via seeds: {seeds}...")

    # Get cluster members from first available seed
    members = None
    for seed in normalized_seeds:
        members = http_get_sync(f"{seed}/cluster/members")
        if members:
            break

    if not members:
        print("Error: Cannot connect to any seed node")
        return False

    # Filter alive members
    alive_members = [m for m in members if m.get("status") == "Alive"]
    print(f"Found {len(alive_members)} alive nodes")

    if len(alive_members) > MAX_NODES_DISPLAY:
        print(
            f"Warning: Cluster has {len(alive_members)} nodes, "
            f"showing first {MAX_NODES_DISPLAY}"
        )
        alive_members = alive_members[:MAX_NODES_DISPLAY]

    print()

    # Collect actors from each node
    all_nodes_data = []

    for i, member in enumerate(alive_members):
        addr = member.get("addr")
        node_id = member.get("node_id")

        if not addr:
            continue

        # Ensure http:// prefix
        if not addr.startswith("http"):
            addr = f"http://{addr}"

        if output_format == "table":
            print(f"{'='*80}")
            print(f"[{i+1}/{len(alive_members)}] Node {node_id} ({addr})")
            print(f"{'='*80}")

        # Get actors from this node
        url = f"{addr}/actors"
        if all_actors:
            url += "?all=true"

        actors = http_get_sync(url)
        if actors is None:
            if output_format == "table":
                print("  Error: Cannot connect to this node")
                print()
            continue

        if output_format == "table":
            _print_actors_table(actors)
            print()
        else:
            all_nodes_data.append(
                {
                    "node_id": node_id,
                    "addr": addr,
                    "actors": actors,
                }
            )

    # Print JSON if needed
    if output_format == "json":
        print(json.dumps(all_nodes_data, indent=2))

    # Summary
    if output_format == "table":
        print(f"{'='*80}")
        print(f"Cluster: {len(alive_members)} nodes")
        print(f"{'='*80}")

    return True


def _print_output(actors_data: list[dict], output_format: str):
    """Print actors in specified format"""
    if output_format == "json":
        print(json.dumps(actors_data, indent=2))
    else:
        _print_actors_table(actors_data)


async def list_actors_impl(all_actors: bool = False, output_format: str = "table"):
    """
    List actors in the current (local) system.

    This function is for testing and in-process usage.
    For CLI, use list_actors_command which uses HTTP API.

    Args:
        all_actors: Show all actors including internal system actors
        output_format: Output format ('table' or 'json')
    """
    from pulsing.actor import get_system

    system = get_system()
    all_named = await system.all_named_actors()

    actors_data = []
    for actor_info in all_named:
        path = actor_info.get("path", "")
        name = path[7:] if path.startswith("actors/") else path

        # Skip system/core
        if path == "system/core":
            continue

        # Filter internal actors if needed
        if not all_actors and name.startswith("_"):
            continue

        actor_data = {
            "name": name,
            "type": "system" if name.startswith("_") else "user",
            "uptime": 0,
        }

        # Get detailed instance info
        detailed = actor_info.get("detailed_instances", [])
        if detailed:
            inst = detailed[0]
            actor_data["actor_id"] = inst.get("actor_id", "-")
            actor_data["module"] = inst.get("module", "-")
            actor_data["class"] = inst.get("class", "-")
            actor_data["file"] = inst.get("file", "-")

        actors_data.append(actor_data)

    _print_output(actors_data, output_format)


def _print_actors_table(actors_data: list[dict]):
    """Print actors in table format"""
    if not actors_data:
        print("  No actors found.")
        return

    # Check for errors
    if actors_data and "error" in actors_data[0]:
        print(f"  Error: {actors_data[0]['error']}")
        return

    # Check if we have detailed info (class, module)
    has_details = any(a.get("class") for a in actors_data)

    if has_details:
        print(f"  {'Name':<25} {'Type':<8} {'Class':<25} {'Module':<30}")
        print(f"  {'-'*90}")

        for actor in actors_data:
            name = actor.get("name", "")
            actor_type = actor.get("type", "user")
            cls = actor.get("class", "-")
            module = actor.get("module", "-")
            print(f"  {name:<25} {actor_type:<8} {cls:<25} {module:<30}")
    else:
        print(f"  {'Name':<40} {'Type':<10}")
        print(f"  {'-'*50}")

        for actor in actors_data:
            name = actor.get("name", "")
            actor_type = actor.get("type", "user")
            print(f"  {name:<40} {actor_type:<10}")

    print(f"\n  Total: {len(actors_data)} actor(s)")


def list_actors_command(
    endpoint: str | None = None,
    seeds: str | None = None,
    all_actors: bool = False,
    json_output: bool = False,
):
    """
    List actors from a remote actor system or cluster.

    Uses simple HTTP/2 API calls - does NOT join the gossip cluster.

    Args:
        endpoint: Single actor system endpoint (e.g., '127.0.0.1:8000')
        seeds: Comma-separated cluster seed addresses
        all_actors: Show all actors including internal system actors
        json_output: Output in JSON format

    Examples:
        # Query single endpoint
        pulsing actor list --endpoint 127.0.0.1:8000

        # Query cluster
        pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001

        # Show all actors as JSON
        pulsing actor list --endpoint 127.0.0.1:8000 --all_actors True --json True
    """
    if not endpoint and not seeds:
        print("Error: Either --endpoint or --seeds is required.")
        print()
        print("Usage:")
        print("  pulsing actor list --endpoint 127.0.0.1:8000")
        print("  pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001")
        return

    if endpoint and seeds:
        print("Error: Cannot specify both --endpoint and --seeds.")
        print("Use --endpoint for single node, --seeds for cluster.")
        return

    output_format = "json" if json_output else "table"

    if endpoint:
        asyncio.run(query_single_endpoint(endpoint, all_actors, output_format))
    else:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
        asyncio.run(query_cluster(seed_list, all_actors, output_format))
