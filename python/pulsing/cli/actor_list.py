"""Actor list command implementation

Query actors from a remote actor system or cluster.

Usage:
    # Query single endpoint
    pulsing actor list --endpoint 127.0.0.1:8000

    # Query cluster
    pulsing actor list --seeds 127.0.0.1:8000,127.0.0.1:8001
"""

import asyncio

MAX_NODES_DISPLAY = 64  # Maximum number of nodes to display


async def query_single_endpoint(endpoint: str, all_actors: bool, output_format: str):
    """Query a single actor system endpoint"""
    from pulsing.actor import SystemConfig, create_actor_system

    print(f"Connecting to {endpoint}...")

    # Parse endpoint
    if ":" not in endpoint:
        endpoint = f"{endpoint}:8000"

    # Create temporary system to connect
    host = endpoint.rsplit(":", 1)[0]
    if host in ("127.0.0.1", "localhost"):
        config = SystemConfig.with_addr("127.0.0.1:0").with_seeds([endpoint])
    else:
        config = SystemConfig.standalone().with_seeds([endpoint])

    system = await create_actor_system(config)

    # Wait for connection
    await asyncio.sleep(1.0)

    # Find the target node
    members = await system.members()
    target_node = None
    for m in members:
        addr = m.get("addr", "")
        if addr == endpoint or endpoint in addr:
            target_node = m
            break

    if not target_node:
        print(f"Error: Cannot find node at {endpoint}")
        print(f"Found nodes: {[m.get('addr') for m in members]}")
        await system.shutdown()
        return

    node_id = str(target_node.get("node_id"))
    node_addr = target_node.get("addr")

    print(f"Connected to node {node_id} ({node_addr})")
    print()

    # Get actors on this node using instance lookup
    actors_data = await _get_node_actors_via_instances(system, node_id, all_actors)
    _print_output(actors_data, output_format)

    await system.shutdown()


async def query_cluster(seeds: list[str], all_actors: bool, output_format: str):
    """Query all nodes in a cluster"""
    from pulsing.actor import SystemConfig, create_actor_system

    print(f"Connecting to cluster via seeds: {seeds}...")

    # Create temporary system to join cluster
    if any(s.startswith("127.0.0.1") or s.startswith("localhost") for s in seeds):
        config = SystemConfig.with_addr("127.0.0.1:0").with_seeds(seeds)
    else:
        config = SystemConfig.standalone().with_seeds(seeds)

    system = await create_actor_system(config)

    # Wait for cluster discovery
    await asyncio.sleep(1.5)

    members = await system.members()
    # Filter to only include seed nodes
    seed_ports = {s.split(":")[-1] for s in seeds}
    real_members = [
        m
        for m in members
        if m.get("status") == "Alive" and m.get("addr", "").split(":")[-1] in seed_ports
    ]

    if not real_members:
        # If no seed matches, just use all alive members except ourselves
        my_addr = system.addr
        real_members = [
            m
            for m in members
            if m.get("status") == "Alive" and m.get("addr") != my_addr
        ]

    print(f"Found {len(real_members)} target nodes")

    if len(real_members) > MAX_NODES_DISPLAY:
        print(
            f"Warning: Cluster has {len(real_members)} nodes, showing first {MAX_NODES_DISPLAY}"
        )
        real_members = real_members[:MAX_NODES_DISPLAY]

    print()

    # Collect actors per node
    all_nodes_data = []

    for i, member in enumerate(real_members):
        node_id = str(member.get("node_id"))
        node_addr = member.get("addr")

        if output_format == "table":
            print(f"{'='*70}")
            print(f"[{i+1}/{len(real_members)}] Node ({node_addr})")
            print(f"{'='*70}")

        actors_data = await _get_node_actors_via_instances(system, node_id, all_actors)

        if output_format == "table":
            _print_actors_table(actors_data)
            print()
        else:
            all_nodes_data.append(
                {
                    "node_id": node_id,
                    "addr": node_addr,
                    "actors": actors_data,
                }
            )

    # Print JSON if needed
    if output_format == "json":
        import json

        print(json.dumps(all_nodes_data, indent=2))

    # Summary
    if output_format == "table":
        total = (
            sum(len(d.get("actors", [])) for d in all_nodes_data)
            if all_nodes_data
            else 0
        )
        print(f"{'='*70}")
        print(f"Cluster: {len(real_members)} nodes")
        print(f"{'='*70}")

    await system.shutdown()


async def _get_node_actors_via_instances(
    system, target_node_id: str, all_actors: bool
) -> list[dict]:
    """Get actors on a specific node by looking up instances for each named actor"""
    try:
        all_named = await system.all_named_actors()
    except Exception as e:
        return [{"error": str(e)}]

    # Build node -> actors mapping by checking instances
    node_actors = []

    for actor_info in all_named:
        path = str(actor_info.get("path", ""))
        name = path[7:] if path.startswith("actors/") else path

        # Skip internal actors unless all_actors is True
        if not all_actors and name.startswith("_"):
            continue

        instance_count = actor_info.get("instance_count", 0)
        if instance_count == 0:
            continue

        # Get instances for this actor
        try:
            instances = await system.get_named_instances(name)
            for inst in instances:
                inst_node_id = str(inst.get("node_id", ""))
                if inst_node_id == target_node_id:
                    node_actors.append(
                        {
                            "name": name,
                            "type": "system" if name.startswith("_") else "user",
                            "actor_id": inst.get("actor_id", "-"),
                        }
                    )
        except Exception:
            pass  # Skip on error

    return node_actors


def _print_output(actors_data: list[dict], output_format: str):
    """Print actors in specified format"""
    if output_format == "json":
        import json

        print(json.dumps(actors_data, indent=2))
    else:
        _print_actors_table(actors_data)


def _print_actors_table(actors_data: list[dict]):
    """Print actors in table format"""
    if not actors_data:
        print("  No actors found.")
        return

    # Check for errors
    if actors_data and "error" in actors_data[0]:
        print(f"  Error: {actors_data[0]['error']}")
        return

    print(f"  {'Name':<40} {'Type':<10} {'Actor ID':<20}")
    print(f"  {'-'*70}")

    for actor in actors_data:
        name = actor.get("name", "")
        actor_type = actor.get("type", "user")
        actor_id = actor.get("actor_id", "-")
        if actor_id is None:
            actor_id = "-"
        print(f"  {name:<40} {actor_type:<10} {actor_id:<20}")

    print(f"\n  Total: {len(actors_data)} actor(s)")


async def list_actors_impl(all_actors: bool = False, output_format: str = "table"):
    """
    Implementation for listing actors in the current system (for testing).

    Args:
        all_actors: Show all actors including internal system actors
        output_format: Output format ('table' or 'json')
    """
    from pulsing.actor import get_system

    system = get_system()

    # Get all named actors
    all_named = await system.all_named_actors()

    # Build actors list
    actors_data = []
    for actor_info in all_named:
        path = actor_info.get("path", "")
        name = path[7:] if path.startswith("actors/") else path

        # Filter internal actors if needed
        if not all_actors and name.startswith("_"):
            continue

        # Get detailed instance information
        try:
            instances = await system.get_named_instances(name)
            for inst in instances:
                actors_data.append(
                    {
                        "name": name,
                        "type": "system" if name.startswith("_") else "user",
                        "actor_id": inst.get("actor_id", "-"),
                        "uptime": inst.get("uptime_s", 0),
                    }
                )
        except Exception:
            actors_data.append(
                {
                    "name": name,
                    "type": "system" if name.startswith("_") else "user",
                    "actor_id": "-",
                    "uptime": 0,
                }
            )

    _print_output(actors_data, output_format)


def list_actors_command(
    endpoint: str | None = None,
    seeds: str | None = None,
    all_actors: bool = False,
    json_output: bool = False,
):
    """
    List actors from a remote actor system or cluster.

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
    import uvloop

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
        uvloop.run(query_single_endpoint(endpoint, all_actors, output_format))
    else:
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
        uvloop.run(query_cluster(seed_list, all_actors, output_format))
