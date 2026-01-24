"""Pulsing CLI - System inspection (observer mode via HTTP API)

This module provides lightweight inspection tools that query actor systems
via HTTP/2 endpoints without joining the gossip cluster.
"""

import json
import subprocess
import time
from collections import defaultdict
from typing import Any

# Maximum number of nodes to display
MAX_NODES_DISPLAY = 64


def http_get_sync(url: str, timeout: float = 10.0) -> dict | list | None:
    """Make HTTP/2 GET request using curl (which supports h2c), returns JSON"""
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",  # Silent
                "--http2-prior-knowledge",  # Use h2c (HTTP/2 over cleartext)
                "-m",
                str(int(timeout)),  # Timeout
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
        return None
    except Exception:
        return None


def http_get_text_sync(url: str, timeout: float = 10.0) -> str | None:
    """Make HTTP/2 GET request using curl, returns raw text (for /metrics endpoint)"""
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",  # Silent
                "--http2-prior-knowledge",  # Use h2c (HTTP/2 over cleartext)
                "-m",
                str(int(timeout)),  # Timeout
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 2,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
        return None
    except Exception:
        return None


def normalize_address(addr: str) -> str:
    """Normalize address to http://host:port format"""
    if ":" not in addr:
        addr = f"{addr}:8000"
    if not addr.startswith("http"):
        addr = f"http://{addr}"
    return addr


def get_cluster_members(seeds: list[str], timeout: float) -> list[dict] | None:
    """Get cluster members from first available seed"""
    for seed in seeds:
        normalized = normalize_address(seed)
        members = http_get_sync(f"{normalized}/cluster/members", timeout)
        if members:
            return members
    return None


def get_alive_members(members: list[dict]) -> list[dict]:
    """Filter alive members from cluster members list"""
    return [m for m in members if m.get("status") == "Alive"]


def inspect_cluster(seeds: list[str], timeout: float = 10.0, best_effort: bool = False):
    """Inspect cluster members and status"""
    print(f"Connecting to cluster via seeds: {seeds}...")

    members = get_cluster_members(seeds, timeout)
    if not members:
        print("Error: Cannot connect to any seed node")
        if not best_effort:
            return
        return

    alive_members = get_alive_members(members)

    # Count by status
    status_counts = defaultdict(int)
    for m in members:
        status = m.get("status", "Unknown")
        status_counts[status] += 1

    print(f"\nCluster Status: {len(members)} total nodes ({len(alive_members)} alive)")
    print("=" * 80)

    # Status summary
    print("\nStatus Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # Member details
    if len(alive_members) > MAX_NODES_DISPLAY:
        print(
            f"\nWarning: Cluster has {len(alive_members)} alive nodes, "
            f"showing first {MAX_NODES_DISPLAY}"
        )
        alive_members = alive_members[:MAX_NODES_DISPLAY]

    print(f"\n{'Node ID':<20} {'Address':<30} {'Status':<15}")
    print("-" * 80)

    for member in sorted(alive_members, key=lambda m: m.get("node_id", 0)):
        node_id = str(member.get("node_id", "-"))
        addr = member.get("addr", "-")
        status = member.get("status", "Unknown")
        print(f"{node_id:<20} {addr:<30} {status:<15}")

    # Show non-alive members if any
    non_alive = [m for m in members if m.get("status") != "Alive"]
    if non_alive:
        print(f"\nNon-Alive Members ({len(non_alive)}):")
        for member in non_alive:
            node_id = str(member.get("node_id", "-"))
            addr = member.get("addr", "-")
            status = member.get("status", "Unknown")
            print(f"  {node_id} ({addr}) - {status}")

    print("\n" + "=" * 80)


def inspect_actors(
    seeds: list[str] | None = None,
    endpoint: str | None = None,
    timeout: float = 10.0,
    best_effort: bool = False,
    top: int | None = None,
    filter: str | None = None,
    all_actors: bool = False,
    json_output: bool = False,
    detailed: bool = False,
):
    """
    Inspect actors from a single node or cluster.

    Supports two modes:
    - Single node: Use --endpoint to query one node
    - Cluster: Use --seeds to query all nodes in cluster (aggregated view)
    """
    # Handle single endpoint mode
    if endpoint:
        return _inspect_single_node(
            endpoint, timeout, all_actors, json_output, detailed, filter
        )

    # Cluster mode (original behavior)
    if not seeds:
        print("Error: Either --endpoint or --seeds is required.")
        print()
        print("Usage:")
        print("  pulsing inspect actors --endpoint 127.0.0.1:8000")
        print("  pulsing inspect actors --seeds 127.0.0.1:8000")
        return

    _inspect_cluster_actors(
        seeds, timeout, best_effort, top, filter, all_actors, json_output
    )


def _inspect_single_node(
    endpoint: str,
    timeout: float,
    all_actors: bool,
    json_output: bool,
    detailed: bool,
    filter_str: str | None = None,
):
    """Inspect actors from a single node"""
    # Normalize endpoint
    if ":" not in endpoint:
        endpoint = f"{endpoint}:8000"
    normalized = normalize_address(endpoint)

    print(f"Connecting to {normalized}...")

    url = f"{normalized}/actors"
    if all_actors:
        url += "?all=true"

    actors = http_get_sync(url, timeout)
    if actors is None:
        print(f"Error: Cannot connect to {normalized}")
        return

    print(f"Connected to {normalized}")
    print()

    # Apply filter if specified
    if filter_str:
        actors = [a for a in actors if filter_str.lower() in a.get("name", "").lower()]

    if json_output:
        print(json.dumps(actors, indent=2))
        return

    _print_actors_table(actors, detailed)


def _print_actors_table(actors_data: list[dict], detailed: bool = False):
    """Print actors in table format"""
    if not actors_data:
        print("  No actors found.")
        return

    # Check for errors
    if actors_data and "error" in actors_data[0]:
        print(f"  Error: {actors_data[0]['error']}")
        return

    # Check if we have detailed info (class, module)
    has_details = detailed or any(a.get("class") for a in actors_data)

    if has_details:
        print(
            f"  {'Name':<25} {'Type':<8} {'Actor ID':<20} {'Class':<25} {'Module':<30}"
        )
        print(f"  {'-' * 110}")

        for actor in actors_data:
            name = actor.get("name", "")
            actor_type = actor.get("type", "user")
            actor_id = actor.get("actor_id", "-")
            if isinstance(actor_id, (int, float)):
                actor_id = str(int(actor_id))
            elif not isinstance(actor_id, str):
                actor_id = str(actor_id) if actor_id else "-"
            cls = actor.get("class", "-")
            module = actor.get("module", "-")
            print(f"  {name:<25} {actor_type:<8} {actor_id:<20} {cls:<25} {module:<30}")
    else:
        print(f"  {'Name':<40} {'Type':<10} {'Actor ID':<20}")
        print(f"  {'-' * 72}")

        for actor in actors_data:
            name = actor.get("name", "")
            actor_type = actor.get("type", "user")
            actor_id = actor.get("actor_id", "-")
            if isinstance(actor_id, (int, float)):
                actor_id = str(int(actor_id))
            elif not isinstance(actor_id, str):
                actor_id = str(actor_id) if actor_id else "-"
            print(f"  {name:<40} {actor_type:<10} {actor_id:<20}")

    print(f"\n  Total: {len(actors_data)} actor(s)")


def _inspect_cluster_actors(
    seeds: list[str],
    timeout: float,
    best_effort: bool,
    top: int | None,
    filter: str | None,
    all_actors: bool,
    json_output: bool,
):
    """Inspect actors distribution across cluster (aggregated view)"""
    # Convert top to int if it's a string (hyperparameter may pass as string)
    if top is not None and not isinstance(top, int):
        try:
            top = int(top)
        except (ValueError, TypeError):
            top = None
    print(f"Connecting to cluster via seeds: {seeds}...")

    members = get_cluster_members(seeds, timeout)
    if not members:
        print("Error: Cannot connect to any seed node")
        if not best_effort:
            return
        return

    alive_members = get_alive_members(members)
    print(f"Found {len(alive_members)} alive nodes")

    # Collect actors from each node
    # actor_name -> list of (node_id, actor_id) tuples
    actor_distribution: dict[str, list[tuple[str, str]]] = defaultdict(list)
    failed_nodes = []

    for member in alive_members:
        addr = member.get("addr")
        node_id = str(member.get("node_id", "-"))

        if not addr:
            continue

        normalized = normalize_address(addr)
        url = f"{normalized}/actors"
        if all_actors:
            url += "?all=true"

        actors = http_get_sync(url, timeout)
        if actors is None:
            failed_nodes.append((node_id, addr))
            if not best_effort:
                print(f"Error: Cannot connect to node {node_id} ({addr})")
            continue

        # Extract actor names and IDs
        for actor in actors:
            name = actor.get("name", "")
            if not name:
                continue

            # Apply filter if specified
            if filter and filter.lower() not in name.lower():
                continue

            # Get actor_id if available
            actor_id = actor.get("actor_id", "")
            if isinstance(actor_id, (int, float)):
                actor_id = str(int(actor_id))
            elif not isinstance(actor_id, str):
                actor_id = str(actor_id) if actor_id else "-"

            actor_distribution[name].append((node_id, actor_id))

    if failed_nodes:
        print(f"\nWarning: Failed to query {len(failed_nodes)} node(s):")
        for node_id, addr in failed_nodes:
            print(f"  {node_id} ({addr})")

    if not actor_distribution:
        print("\nNo actors found.")
        return

    # Sort by instance count (descending)
    sorted_actors = sorted(
        actor_distribution.items(), key=lambda x: len(x[1]), reverse=True
    )

    # Apply top filter
    if top is not None and top > 0:
        sorted_actors = sorted_actors[:top]

    print(f"\nActor Distribution ({len(actor_distribution)} unique actors):")
    print("=" * 100)
    print(
        f"{'Actor Name':<30} {'Actor ID (node:local)':<25} {'Instances':<12} {'Nodes':<30}"
    )
    print("-" * 100)

    for actor_name, node_actor_pairs in sorted_actors:
        instance_count = len(node_actor_pairs)
        # Get unique node IDs
        unique_nodes = sorted(set(node_id for node_id, _ in node_actor_pairs))
        nodes_str = ", ".join(unique_nodes[:5])
        if len(unique_nodes) > 5:
            nodes_str += f" ... (+{len(unique_nodes) - 5} more)"

        # Get actor IDs - collect all unique IDs
        actor_ids = [actor_id for _, actor_id in node_actor_pairs if actor_id != "-"]
        if actor_ids:
            unique_actor_ids = sorted(set(actor_ids))
            if len(unique_actor_ids) == 1:
                # All instances have same ID (single instance or same ID across nodes)
                actor_id_str = unique_actor_ids[0]
            else:
                # Multiple different IDs, show first one and indicate more
                actor_id_str = unique_actor_ids[0]
                if len(unique_actor_ids) > 1:
                    actor_id_str += f" (+{len(unique_actor_ids) - 1} more)"
        else:
            actor_id_str = "-"

        print(
            f"{actor_name:<30} {actor_id_str:<20} {instance_count:<12} {nodes_str:<30}"
        )

    # Summary
    total_instances = sum(len(pairs) for pairs in actor_distribution.values())
    unique_nodes = set()
    for pairs in actor_distribution.values():
        unique_nodes.update(node_id for node_id, _ in pairs)
    print("\n" + "=" * 100)
    print(
        f"Total: {len(actor_distribution)} unique actors, {total_instances} instances"
    )
    print(f"Across {len(unique_nodes)} nodes")


def inspect_metrics(
    seeds: list[str], timeout: float = 10.0, best_effort: bool = False, raw: bool = True
):
    """Inspect Prometheus metrics from cluster nodes"""
    print(f"Connecting to cluster via seeds: {seeds}...")

    members = get_cluster_members(seeds, timeout)
    if not members:
        print("Error: Cannot connect to any seed node")
        if not best_effort:
            return
        return

    alive_members = get_alive_members(members)
    print(f"Found {len(alive_members)} alive nodes\n")

    failed_nodes = []

    for i, member in enumerate(alive_members):
        addr = member.get("addr")
        node_id = str(member.get("node_id", "-"))

        if not addr:
            continue

        normalized = normalize_address(addr)
        metrics = http_get_text_sync(f"{normalized}/metrics", timeout)

        if metrics is None:
            failed_nodes.append((node_id, addr))
            if not best_effort:
                print(f"Error: Cannot connect to node {node_id} ({addr})")
            continue

        if raw:
            # Output raw metrics (as text)
            print(f"{'=' * 80}")
            print(f"[{i + 1}/{len(alive_members)}] Node {node_id} ({addr})")
            print(f"{'=' * 80}")
            print(metrics)
            print()
        else:
            # Summary mode: extract key metrics
            lines = metrics.split("\n")
            key_metrics = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Look for important metrics
                if any(
                    keyword in line
                    for keyword in [
                        "pulsing_cluster_members",
                        "pulsing_actor_messages",
                        "pulsing_actor_",
                    ]
                ):
                    key_metrics.append(line)
            print(f"Node {node_id} ({addr}):")
            if key_metrics:
                for metric in key_metrics[:20]:  # Limit to top 20
                    print(f"  {metric}")
            else:
                print("  (no key metrics found)")
            print()

    if failed_nodes:
        print(f"\nWarning: Failed to query {len(failed_nodes)} node(s):")
        for node_id, addr in failed_nodes:
            print(f"  {node_id} ({addr})")

    if not raw:
        print("=" * 80)
        print("(Use --raw True to see full metrics)")


def inspect_watch(
    seeds: list[str],
    timeout: float = 10.0,
    best_effort: bool = False,
    interval: float = 1.0,
    kind: str = "all",
    max_rounds: int | None = None,
):
    """Watch cluster state changes"""
    # Convert max_rounds to int if it's a string (hyperparameter may pass as string)
    if max_rounds is not None and not isinstance(max_rounds, int):
        try:
            max_rounds = int(max_rounds)
        except (ValueError, TypeError):
            max_rounds = None

    # Convert interval to float if it's a string
    if not isinstance(interval, (int, float)):
        try:
            interval = float(interval)
        except (ValueError, TypeError):
            interval = 1.0

    print(f"Watching cluster via seeds: {seeds}...")
    print(f"Refresh interval: {interval}s, Watching: {kind}")
    print("Press Ctrl+C to stop\n")

    previous_state: dict[str, Any] = {}

    round_count = 0

    try:
        while True:
            if max_rounds is not None and round_count >= max_rounds:
                break

            round_count += 1
            current_time = time.strftime("%H:%M:%S")

            # Collect current state
            current_state: dict[str, Any] = {}

            if kind in ("cluster", "all"):
                members = get_cluster_members(seeds, timeout)
                if members:
                    alive = get_alive_members(members)
                    current_state["cluster"] = {
                        "total": len(members),
                        "alive": len(alive),
                        "members": {
                            str(m.get("node_id")): m.get("status") for m in members
                        },
                    }

            if kind in ("actors", "all"):
                members = get_cluster_members(seeds, timeout)
                if members:
                    alive = get_alive_members(members)
                    actor_distribution: dict[str, int] = defaultdict(int)
                    for member in alive:
                        addr = member.get("addr")
                        if not addr:
                            continue
                        normalized = normalize_address(addr)
                        actors = http_get_sync(f"{normalized}/actors", timeout)
                        if actors:
                            for actor in actors:
                                name = actor.get("name", "")
                                if name:
                                    actor_distribution[name] += 1
                    current_state["actors"] = dict(actor_distribution)

            # Detect and print changes
            if previous_state:
                changes = []

                if "cluster" in current_state and "cluster" in previous_state:
                    prev = previous_state["cluster"]
                    curr = current_state["cluster"]

                    if prev["alive"] != curr["alive"]:
                        changes.append(
                            f"Alive nodes: {prev['alive']} -> {curr['alive']}"
                        )

                    # Check member status changes
                    prev_members = prev.get("members", {})
                    curr_members = curr.get("members", {})
                    for node_id, status in curr_members.items():
                        prev_status = prev_members.get(node_id)
                        if prev_status != status:
                            changes.append(f"Node {node_id}: {prev_status} -> {status}")

                if "actors" in current_state and "actors" in previous_state:
                    prev_actors = previous_state["actors"]
                    curr_actors = current_state["actors"]

                    # New actors
                    new_actors = set(curr_actors.keys()) - set(prev_actors.keys())
                    if new_actors:
                        changes.append(
                            f"New actors: {', '.join(sorted(new_actors)[:5])}"
                        )

                    # Removed actors
                    removed_actors = set(prev_actors.keys()) - set(curr_actors.keys())
                    if removed_actors:
                        changes.append(
                            f"Removed actors: {', '.join(sorted(removed_actors)[:5])}"
                        )

                    # Count changes
                    for actor_name in set(curr_actors.keys()) & set(prev_actors.keys()):
                        if prev_actors[actor_name] != curr_actors[actor_name]:
                            changes.append(
                                f"{actor_name}: {prev_actors[actor_name]} -> {curr_actors[actor_name]} instances"
                            )

                if changes:
                    print(f"[{current_time}] Changes detected:")
                    for change in changes:
                        print(f"  • {change}")
                    print()
                else:
                    print(f"[{current_time}] No changes")
            else:
                # First round - just show current state
                if "cluster" in current_state:
                    c = current_state["cluster"]
                    print(
                        f"[{current_time}] Initial state: {c['alive']}/{c['total']} nodes alive"
                    )
                if "actors" in current_state:
                    a = current_state["actors"]
                    print(
                        f"[{current_time}] Initial state: {len(a)} unique actors, {sum(a.values())} total instances"
                    )
                print()

            previous_state = current_state
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nWatch stopped.")
