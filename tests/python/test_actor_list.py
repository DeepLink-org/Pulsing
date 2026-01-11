"""Test actor list command"""

import asyncio
import pytest
from pulsing.actor import init, remote, get_system
from pulsing.cli.actor_list import list_actors_impl
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

    # Create some actors
    await TestCounter.remote(system, name="counter-1")
    await TestCounter.remote(system, name="counter-2")
    await TestCalculator.remote(system, name="calc")

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # List user actors only
        await list_actors_impl(all_actors=False, output_format="table")
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

    # Create one user actor
    await TestCounter.remote(system, name="test-counter")

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # List all actors
        await list_actors_impl(all_actors=True, output_format="table")
        output = buffer.getvalue()

        # Check output contains user actor
        assert "test-counter" in output

        # Should also contain system actors
        assert "_system_internal" in output or "_python_actor_service" in output

    finally:
        sys.stdout = old_stdout


@pytest.mark.asyncio
async def test_actor_list_json():
    """Test JSON output format"""
    await init()
    system = get_system()

    await TestCounter.remote(system, name="json-test")

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        await list_actors_impl(all_actors=False, output_format="json")
        output = buffer.getvalue()

        # Should be valid JSON
        import json

        data = json.loads(output)

        # Check structure
        assert isinstance(data, list)
        assert len(data) >= 1

        # Check actor data structure
        actor = data[0]
        assert "name" in actor
        assert "type" in actor
        assert "uptime" in actor

    finally:
        sys.stdout = old_stdout
