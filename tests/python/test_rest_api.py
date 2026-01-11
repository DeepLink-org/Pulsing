"""Test REST API endpoints for actor list and cluster info"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock
import json
import subprocess

from pulsing.cli.actor_list import (
    query_single_endpoint,
    query_cluster,
    _print_output,
    _print_actors_table,
    http_get_sync,
)


class TestRestApiHelpers:
    """Test helper functions for REST API"""

    def test_print_actors_table_user_only(self, capsys):
        """Test printing user actors table"""
        actors = [
            {
                "name": "counter-1",
                "type": "user",
                "class": "Counter",
                "module": "__main__",
            },
            {
                "name": "calculator",
                "type": "user",
                "class": "Calculator",
                "module": "app",
            },
        ]
        _print_actors_table(actors)
        captured = capsys.readouterr()
        assert "counter-1" in captured.out
        assert "calculator" in captured.out
        assert "Counter" in captured.out
        assert "Calculator" in captured.out
        assert "Total: 2 actor(s)" in captured.out

    def test_print_actors_table_empty(self, capsys):
        """Test printing empty actors table"""
        _print_actors_table([])
        captured = capsys.readouterr()
        assert "No actors found" in captured.out

    def test_print_output_table(self, capsys):
        """Test _print_output with table format"""
        actors = [{"name": "test", "type": "user", "class": "Test", "module": "app"}]
        _print_output(actors, "table")
        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_print_output_json(self, capsys):
        """Test _print_output with JSON format"""
        actors = [{"name": "test", "type": "user", "class": "Test", "module": "app"}]
        _print_output(actors, "json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "test"


class TestHttpGetSync:
    """Test http_get_sync function"""

    def test_http_get_sync_success(self):
        """Test successful HTTP GET request"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"status": "ok"})

        with patch("subprocess.run", return_value=mock_result):
            result = http_get_sync("http://127.0.0.1:8000/health")
            assert result == {"status": "ok"}

    def test_http_get_sync_failure(self):
        """Test HTTP GET request failure"""
        mock_result = MagicMock()
        mock_result.returncode = 7  # Connection refused

        with patch("subprocess.run", return_value=mock_result):
            result = http_get_sync("http://127.0.0.1:8000/health")
            assert result is None

    def test_http_get_sync_invalid_json(self):
        """Test HTTP GET with invalid JSON response"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json"

        with patch("subprocess.run", return_value=mock_result):
            result = http_get_sync("http://127.0.0.1:8000/health")
            assert result is None


class TestQuerySingleEndpoint:
    """Test query_single_endpoint function"""

    @pytest.mark.asyncio
    async def test_query_single_endpoint_success(self):
        """Test successful query to single endpoint"""
        # /actors endpoint returns a flat array of actors
        mock_actors_response = [
            {
                "name": "counter",
                "type": "user",
                "actor_id": "123:1",
                "class": "Counter",
                "module": "__main__",
                "file": "/app/main.py",
            }
        ]

        with patch("pulsing.cli.actor_list.http_get_sync") as mock_http:
            mock_http.return_value = mock_actors_response
            result = await query_single_endpoint(
                "127.0.0.1:8000", all_actors=False, output_format="table"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_query_single_endpoint_connection_error(self):
        """Test query with connection error"""
        with patch("pulsing.cli.actor_list.http_get_sync") as mock_http:
            mock_http.return_value = None
            result = await query_single_endpoint(
                "127.0.0.1:8000", all_actors=False, output_format="table"
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_query_single_endpoint_json_format(self, capsys):
        """Test query with JSON output format"""
        # /actors endpoint returns a flat array
        mock_response = [
            {
                "name": "test",
                "type": "user",
                "actor_id": "123:1",
                "class": "Test",
                "module": "app",
            }
        ]

        with patch("pulsing.cli.actor_list.http_get_sync") as mock_http:
            mock_http.return_value = mock_response
            result = await query_single_endpoint(
                "127.0.0.1:8000", all_actors=False, output_format="json"
            )
            assert result is True
            captured = capsys.readouterr()
            # Extract JSON from output (skip connection info lines)
            lines = captured.out.strip().split("\n")
            json_start = next(
                i for i, line in enumerate(lines) if line.strip().startswith("[")
            )
            json_str = "\n".join(lines[json_start:])
            data = json.loads(json_str)
            assert len(data) == 1


class TestQueryCluster:
    """Test query_cluster function"""

    @pytest.mark.asyncio
    async def test_query_cluster_success(self):
        """Test successful cluster query"""
        members_response = [
            {"node_id": "123", "addr": "127.0.0.1:8001", "status": "Alive"},
            {"node_id": "456", "addr": "127.0.0.1:8002", "status": "Alive"},
        ]

        # /actors returns a flat array of actors
        actors_response = [
            {
                "name": "counter",
                "type": "user",
                "actor_id": "123:1",
                "class": "Counter",
                "module": "__main__",
            }
        ]

        def mock_http_get(url):
            if "/cluster/members" in url:
                return members_response
            elif "/actors" in url:
                return actors_response
            return None

        with patch("pulsing.cli.actor_list.http_get_sync", side_effect=mock_http_get):
            result = await query_cluster(
                ["127.0.0.1:8000"], all_actors=False, output_format="table"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_query_cluster_no_members(self):
        """Test cluster query with no members found"""
        with patch("pulsing.cli.actor_list.http_get_sync") as mock_http:
            mock_http.return_value = None
            result = await query_cluster(
                ["127.0.0.1:8000"], all_actors=False, output_format="table"
            )
            assert result is False


class TestActorMetadataParsing:
    """Test actor metadata parsing from API responses"""

    def test_parse_actor_with_full_metadata(self):
        """Test parsing actor with complete metadata"""
        actor_data = {
            "path": "actors/my-counter",
            "detailed_instances": [
                {
                    "node_id": 12345,
                    "actor_id": "12345:42",
                    "class": "MyCounter",
                    "module": "myapp.counters",
                    "file": "/app/myapp/counters.py",
                }
            ],
        }

        path = actor_data.get("path", "")
        name = path[7:] if path.startswith("actors/") else path
        assert name == "my-counter"

        instances = actor_data.get("detailed_instances", [])
        assert len(instances) == 1

        inst = instances[0]
        assert inst.get("actor_id") == "12345:42"
        assert inst.get("class") == "MyCounter"
        assert inst.get("module") == "myapp.counters"
        assert inst.get("file") == "/app/myapp/counters.py"

    def test_parse_actor_with_minimal_metadata(self):
        """Test parsing actor with minimal metadata"""
        actor_data = {
            "path": "actors/simple",
            "detailed_instances": [{"node_id": 12345}],
        }

        instances = actor_data.get("detailed_instances", [])
        inst = instances[0]

        # Missing fields should return None or default
        assert inst.get("actor_id") is None
        assert inst.get("class") is None
        assert inst.get("module") is None

    def test_parse_system_actor(self):
        """Test parsing system/internal actor"""
        actor_data = {
            "path": "actors/_python_actor_service",
            "detailed_instances": [
                {
                    "node_id": 12345,
                    "actor_id": "12345:0",
                    "class": "PythonActorService",
                    "module": "pulsing.actor.remote",
                }
            ],
        }

        path = actor_data.get("path", "")
        name = path[7:] if path.startswith("actors/") else path

        # Internal actors start with _
        assert name.startswith("_")
        assert name == "_python_actor_service"

    def test_filter_internal_actors(self):
        """Test filtering internal actors"""
        actors = [
            {"path": "actors/counter", "type": "user"},
            {"path": "actors/_internal", "type": "system"},
            {"path": "actors/calc", "type": "user"},
        ]

        user_only = [a for a in actors if not a["path"].split("/")[-1].startswith("_")]
        assert len(user_only) == 2

        all_actors = actors
        assert len(all_actors) == 3
