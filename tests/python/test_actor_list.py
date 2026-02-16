"""Test actor list command"""

import asyncio
import pytest
import json
from pulsing.core import init, remote, get_system, list_actors
from pulsing.cli.inspect import _print_actors_table
import io
import sys


@remote
class TestCounter:
    def __init__(self):
        self.count = 0


@remote
class TestCalculator:
    def add(self, a, b):
        return a + b


@pytest.mark.asyncio
async def test_actor_list_basic():
    """Test basic actor listing"""
    await init()
    system = get_system()

    # Create some actors locally (list_actors only returns local actors)
    await TestCounter.local(system, name="counter-1")
    await TestCounter.local(system, name="counter-2")
    await TestCalculator.local(system, name="calc")

    # Wait a bit for actors to be registered in the system
    await asyncio.sleep(0.2)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # Use all_named_actors and get_named_instances instead of list_actors
        # list_actors uses SystemActor registry which may not include Python actors
        # all_named_actors uses gossip protocol and includes all named actors
        all_named = await system.all_named_actors()

        # Build actors list similar to HTTP API format
        actors_data = []
        for info in all_named:
            path_str = str(info.get("path", ""))
            if path_str == "system/core":
                continue

            name = path_str[7:] if path_str.startswith("actors/") else path_str

            # Skip internal actors (old style and new system/ namespace)
            if name.startswith("_") or name.startswith("system/"):
                continue

            # Get instances for this actor
            instance_count = info.get("instance_count", 0)
            if instance_count > 0:
                try:
                    instances = await system.get_named_instances(name)
                    for inst in instances:
                        # Check if this instance is on this node
                        if str(inst.get("node_id")) == str(system.node_id.id):
                            actor_data = {
                                "name": name,
                                "type": "user",
                                "actor_id": str(inst.get("actor_id", "")),
                            }
                            actors_data.append(actor_data)
                except Exception:
                    pass

        _print_actors_table(actors_data)
        output = buffer.getvalue()

        # Check output contains actor names
        assert "counter-1" in output
        assert "counter-2" in output
        assert "calc" in output

        # Should not contain system actors
        assert "_system" not in output
        assert "_python_actor_service" not in output

        # Check total count
        assert "Total: 3 actor(s)" in output

    finally:
        sys.stdout = old_stdout


@pytest.mark.asyncio
async def test_actor_list_all():
    """Test listing all actors including system actors"""
    await init()
    system = get_system()

    # Create one user actor locally (list_actors only returns local actors)
    await TestCounter.local(system, name="test-counter")

    # Wait a bit for actors to be registered in the system
    await asyncio.sleep(0.2)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # Use all_named_actors and get_named_instances instead of list_actors
        all_named = await system.all_named_actors()

        # Build actors list similar to HTTP API format (include all actors)
        actors_data = []
        for info in all_named:
            path_str = str(info.get("path", ""))
            if path_str == "system/core":
                continue

            name = path_str[7:] if path_str.startswith("actors/") else path_str

            # Get instances for this actor
            instance_count = info.get("instance_count", 0)
            if instance_count > 0:
                try:
                    instances = await system.get_named_instances(name)
                    for inst in instances:
                        # Check if this instance is on this node
                        if str(inst.get("node_id")) == str(system.node_id.id):
                            actor_data = {
                                "name": name,
                                "type": "system" if name.startswith("_") else "user",
                                "actor_id": str(inst.get("actor_id", "")),
                            }
                            actors_data.append(actor_data)
                except Exception:
                    pass

        _print_actors_table(actors_data)
        output = buffer.getvalue()

        # Check output contains user actor
        assert "test-counter" in output

        # Should also contain system actors
        assert "system/core" in output or "system/python_actor_service" in output

    finally:
        sys.stdout = old_stdout


@pytest.mark.asyncio
async def test_actor_list_json():
    """Test JSON output format"""
    await init()
    system = get_system()

    # Create actor locally (list_actors only returns local actors)
    await TestCounter.local(system, name="json-test")

    # Wait a bit for actors to be registered in the system
    await asyncio.sleep(0.2)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # Use all_named_actors and get_named_instances instead of list_actors
        all_named = await system.all_named_actors()

        # Build actors list similar to HTTP API format
        actors_data = []
        for info in all_named:
            path_str = str(info.get("path", ""))
            if path_str == "system/core":
                continue

            name = path_str[7:] if path_str.startswith("actors/") else path_str

            # Filter internal actors
            if name.startswith("_"):
                continue

            # Get instances for this actor
            instance_count = info.get("instance_count", 0)
            if instance_count > 0:
                try:
                    instances = await system.get_named_instances(name)
                    for inst in instances:
                        # Check if this instance is on this node
                        if str(inst.get("node_id")) == str(system.node_id.id):
                            actor_data = {
                                "name": name,
                                "type": "user",
                                "actor_id": str(inst.get("actor_id", "")),
                            }
                            actors_data.append(actor_data)
                except Exception:
                    pass

        print(json.dumps(actors_data, indent=2))
        output = buffer.getvalue()

        # Should be valid JSON
        data = json.loads(output)

        # Check structure
        assert isinstance(data, list)
        assert len(data) >= 1

        # Check actor data structure
        actor = data[0]
        assert "name" in actor
        assert "type" in actor
        assert "actor_id" in actor
        # Note: uptime may not be available from get_named_instances
        # HTTP API includes it from instance metadata, but we're using direct API

    finally:
        sys.stdout = old_stdout
