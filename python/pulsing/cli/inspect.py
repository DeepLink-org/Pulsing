"""Pulsing CLI - System inspection"""

import asyncio

import uvloop


def inspect_system(seeds: list[str]):
    """Inspect the actor system state"""
    from pulsing.actor import SystemConfig, create_actor_system

    async def run():
        if not seeds:
            print("Error: --seeds is required for 'inspect' command")
            return

        print(f"Connecting to cluster via seeds: {seeds}...")

        # If seeds are local, bind to 127.0.0.1 to ensure connectivity
        if any(s.startswith("127.0.0.1") or s.startswith("localhost") for s in seeds):
            config = SystemConfig.with_addr("127.0.0.1:0").with_seeds(seeds)
        else:
            config = SystemConfig.standalone().with_seeds(seeds)

        system = await create_actor_system(config)

        # Give some time for discovery
        await asyncio.sleep(1.5)

        members = await system.members()
        print(f"\nCluster Status: {len(members)} nodes found")
        print("=" * 60)

        # Get all named actors automatically
        all_named_actors = {}
        try:
            for info in await system.all_named_actors():
                path = str(info.get("path", ""))
                name = path[7:] if path.startswith("actors/") else path
                if info.get("instance_count", 0) > 0:
                    try:
                        instances = await system.get_named_instances(name)
                        if instances:
                            all_named_actors[name] = instances
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [Warning] Failed to get all named actors: {e}")

        # Group named actors by node
        node_actors = {}
        for name, instances in all_named_actors.items():
            for inst in instances:
                node_actors.setdefault(str(inst.get("node_id")), []).append(name)

        # Display nodes and their actors
        for member in members:
            node_id = str(member.get("node_id"))
            print(f"\nNode: {node_id} ({member.get('addr')}) [{member.get('status')}]")

            if member.get("status") != "Alive":
                print("  [Node is not alive]")
                continue

            actors = node_actors.get(node_id, [])
            if not actors:
                print("  [No named actors on this node]")
                continue

            # Group by base type
            actor_groups = {}
            for name in actors:
                base = name.rsplit("_", 1)[0] if "_" in name else name
                actor_groups.setdefault(base, []).append(name)

            print(f"  Named Actors ({len(actors)}):")
            for base, names in sorted(actor_groups.items()):
                if len(names) == 1:
                    print(f"    - actors/{names[0]}")
                else:
                    print(f"    - actors/{base}_* ({len(names)} instances)")
                    for name in sorted(names)[:5]:
                        print(f"        • {name}")
                    if len(names) > 5:
                        print(f"        ... and {len(names) - 5} more")

        # Summary
        if all_named_actors:
            total = sum(len(instances) for instances in all_named_actors.values())
            print(
                f"\nTotal Named Actors: {len(all_named_actors)} types, {total} instances"
            )

        print("\n" + "=" * 60)
        await system.shutdown()

    uvloop.run(run())
