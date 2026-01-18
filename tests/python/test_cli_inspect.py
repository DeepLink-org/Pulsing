"""Test CLI inspect command

Tests for the inspect CLI subcommands using function-level unit tests.
Since all CLI commands are functions (via hp.param), we can test them directly.
"""

import io
import json
import sys
from unittest.mock import patch, MagicMock

import pytest

from pulsing.cli.__main__ import inspect as inspect_cli
from pulsing.cli.inspect import (
    inspect_cluster,
    inspect_actors,
    inspect_metrics,
    normalize_address,
    get_cluster_members,
    get_alive_members,
)


class TestHelperFunctions:
    """Test helper functions in inspect module"""

    def test_normalize_address(self):
        """Test address normalization"""
        assert normalize_address("127.0.0.1:8000") == "http://127.0.0.1:8000"
        assert normalize_address("127.0.0.1") == "http://127.0.0.1:8000"
        assert normalize_address("http://127.0.0.1:8000") == "http://127.0.0.1:8000"

    def test_get_alive_members(self):
        """Test filtering alive members"""
        members = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
            {"node_id": 2, "addr": "127.0.0.1:8001", "status": "Suspect"},
            {"node_id": 3, "addr": "127.0.0.1:8002", "status": "Alive"},
            {"node_id": 4, "addr": "127.0.0.1:8003", "status": "Failed"},
        ]
        alive = get_alive_members(members)
        assert len(alive) == 2
        assert alive[0]["node_id"] == 1
        assert alive[1]["node_id"] == 3

    @patch("pulsing.cli.inspect.http_get_sync")
    def test_get_cluster_members(self, mock_http_get):
        """Test getting cluster members from seeds"""
        # First seed fails, second succeeds
        mock_http_get.side_effect = [None, [{"node_id": 1, "status": "Alive"}]]

        members = get_cluster_members(["127.0.0.1:8000", "127.0.0.1:8001"], timeout=5.0)
        assert members is not None
        assert len(members) == 1
        assert members[0]["node_id"] == 1

        # All seeds fail
        mock_http_get.side_effect = [None, None]
        members = get_cluster_members(["127.0.0.1:8000", "127.0.0.1:8001"], timeout=5.0)
        assert members is None


class TestInspectCluster:
    """Test inspect cluster subcommand"""

    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_cluster_success(self, mock_get_members):
        """Test successful cluster inspection"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
            {"node_id": 2, "addr": "127.0.0.1:8001", "status": "Alive"},
            {"node_id": 3, "addr": "127.0.0.1:8002", "status": "Suspect"},
        ]

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_cluster(["127.0.0.1:8000"], timeout=5.0)
            output = buffer.getvalue()

            assert "Connecting to cluster via seeds" in output
            assert "Cluster Status: 3 total nodes (2 alive)" in output
            assert "Status Summary" in output
            assert "Alive: 2" in output
            assert "Suspect: 1" in output
            assert "Node ID" in output
            assert "127.0.0.1:8000" in output
            assert "127.0.0.1:8001" in output
        finally:
            sys.stdout = old_stdout

    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_cluster_no_connection(self, mock_get_members):
        """Test cluster inspection when no connection available"""
        mock_get_members.return_value = None

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_cluster(["127.0.0.1:8000"], timeout=5.0, best_effort=False)
            output = buffer.getvalue()

            assert "Error: Cannot connect to any seed node" in output
        finally:
            sys.stdout = old_stdout

    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_cluster_best_effort(self, mock_get_members):
        """Test cluster inspection with best_effort=True"""
        mock_get_members.return_value = None

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            # Should not raise, just return silently
            inspect_cluster(["127.0.0.1:8000"], timeout=5.0, best_effort=True)
            output = buffer.getvalue()

            assert "Error: Cannot connect to any seed node" in output
        finally:
            sys.stdout = old_stdout


class TestInspectActors:
    """Test inspect actors subcommand"""

    @patch("pulsing.cli.inspect.http_get_sync")
    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_actors_success(self, mock_get_members, mock_http_get):
        """Test successful actors inspection"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
            {"node_id": 2, "addr": "127.0.0.1:8001", "status": "Alive"},
        ]

        # Mock actors from each node
        mock_http_get.side_effect = [
            [{"name": "worker-1"}, {"name": "worker-2"}],  # Node 1
            [{"name": "worker-1"}, {"name": "router"}],  # Node 2
        ]

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_actors(["127.0.0.1:8000"], timeout=5.0)
            output = buffer.getvalue()

            assert "Found 2 alive nodes" in output
            assert "Actor Distribution" in output
            assert "worker-1" in output
            assert "worker-2" in output
            assert "router" in output
            assert "Total:" in output
        finally:
            sys.stdout = old_stdout

    @patch("pulsing.cli.inspect.http_get_sync")
    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_actors_with_filter(self, mock_get_members, mock_http_get):
        """Test actors inspection with filter"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
        ]

        mock_http_get.side_effect = [
            [{"name": "worker-1"}, {"name": "router-1"}, {"name": "other"}],
        ]

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_actors(["127.0.0.1:8000"], timeout=5.0, filter="worker")
            output = buffer.getvalue()

            assert "worker-1" in output
            assert "router-1" not in output
            assert "other" not in output
        finally:
            sys.stdout = old_stdout

    @patch("pulsing.cli.inspect.http_get_sync")
    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_actors_with_top(self, mock_get_members, mock_http_get):
        """Test actors inspection with top limit"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
        ]

        # Create many actors
        actors = [{"name": f"actor-{i}"} for i in range(20)]
        mock_http_get.side_effect = [actors]

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_actors(["127.0.0.1:8000"], timeout=5.0, top=5)
            output = buffer.getvalue()

            # Should only show top 5
            assert "actor-0" in output
            assert "actor-4" in output
            # Should not show all 20
            lines = output.split("\n")
            actor_lines = [
                line for line in lines if "actor-" in line and "Actor Name" not in line
            ]
            assert len(actor_lines) <= 5
        finally:
            sys.stdout = old_stdout

    @patch("pulsing.cli.inspect.http_get_sync")
    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_actors_partial_failure(self, mock_get_members, mock_http_get):
        """Test actors inspection with some node failures"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
            {"node_id": 2, "addr": "127.0.0.1:8001", "status": "Alive"},
        ]

        # First node succeeds, second fails
        mock_http_get.side_effect = [
            [{"name": "worker-1"}],  # Node 1
            None,  # Node 2 fails
        ]

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_actors(["127.0.0.1:8000"], timeout=5.0, best_effort=True)
            output = buffer.getvalue()

            assert "worker-1" in output
            assert (
                "Warning: Failed to query" in output
                or "Error: Cannot connect" in output
            )
        finally:
            sys.stdout = old_stdout


class TestInspectMetrics:
    """Test inspect metrics subcommand"""

    @patch("pulsing.cli.inspect.http_get_text_sync")
    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_metrics_raw(self, mock_get_members, mock_http_get_text):
        """Test metrics inspection with raw output"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
        ]

        mock_http_get_text.return_value = (
            "# HELP test_metric Test metric\ntest_metric 1.0\n"
        )

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_metrics(["127.0.0.1:8000"], timeout=5.0, raw=True)
            output = buffer.getvalue()

            assert "Node 1" in output
            assert "test_metric" in output
            assert "# HELP" in output
        finally:
            sys.stdout = old_stdout

    @patch("pulsing.cli.inspect.http_get_text_sync")
    @patch("pulsing.cli.inspect.get_cluster_members")
    def test_inspect_metrics_summary(self, mock_get_members, mock_http_get_text):
        """Test metrics inspection with summary output"""
        mock_get_members.return_value = [
            {"node_id": 1, "addr": "127.0.0.1:8000", "status": "Alive"},
        ]

        metrics_text = """# HELP pulsing_cluster_members Cluster members
pulsing_cluster_members{status="Alive"} 2.0
# HELP pulsing_actor_messages Actor messages
pulsing_actor_messages_total 100.0
# HELP other_metric Other metric
other_metric 5.0
"""

        mock_http_get_text.return_value = metrics_text

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_metrics(["127.0.0.1:8000"], timeout=5.0, raw=False)
            output = buffer.getvalue()

            assert "Node 1" in output
            assert "pulsing_cluster_members" in output
            assert "pulsing_actor_messages" in output
            # Should not show other_metric (not a key metric)
            assert "other_metric" not in output or "(no key metrics found)" in output
        finally:
            sys.stdout = old_stdout


class TestCLIEntryPoint:
    """Test CLI entry point function"""

    @patch("pulsing.cli.inspect.inspect_cluster")
    def test_inspect_cli_cluster_subcommand(self, mock_inspect_cluster):
        """Test CLI entry point with cluster subcommand"""
        inspect_cli(
            subcommand="cluster",
            seeds="127.0.0.1:8000",
            timeout=5.0,
            best_effort=False,
        )

        mock_inspect_cluster.assert_called_once_with(
            ["127.0.0.1:8000"], timeout=5.0, best_effort=False
        )

    @patch("pulsing.cli.inspect.inspect_actors")
    def test_inspect_cli_actors_subcommand(self, mock_inspect_actors):
        """Test CLI entry point with actors subcommand"""
        inspect_cli(
            subcommand="actors",
            seeds="127.0.0.1:8000,127.0.0.1:8001",
            timeout=5.0,
            best_effort=False,
            top=10,
            filter="worker",
            all_actors=False,
        )

        mock_inspect_actors.assert_called_once_with(
            seeds=["127.0.0.1:8000", "127.0.0.1:8001"],
            endpoint=None,
            timeout=5.0,
            best_effort=False,
            top=10,
            filter="worker",
            all_actors=False,
            json_output=False,
            detailed=False,
        )

    @patch("pulsing.cli.inspect.inspect_metrics")
    def test_inspect_cli_metrics_subcommand(self, mock_inspect_metrics):
        """Test CLI entry point with metrics subcommand"""
        inspect_cli(
            subcommand="metrics",
            seeds="127.0.0.1:8000",
            timeout=5.0,
            best_effort=False,
            raw=False,
        )

        mock_inspect_metrics.assert_called_once_with(
            ["127.0.0.1:8000"], timeout=5.0, best_effort=False, raw=False
        )

    @patch("pulsing.cli.inspect.inspect_watch")
    def test_inspect_cli_watch_subcommand(self, mock_inspect_watch):
        """Test CLI entry point with watch subcommand"""
        inspect_cli(
            subcommand="watch",
            seeds="127.0.0.1:8000",
            timeout=5.0,
            best_effort=False,
            interval=2.0,
            kind="cluster",
            max_rounds=5,
        )

        mock_inspect_watch.assert_called_once_with(
            ["127.0.0.1:8000"],
            timeout=5.0,
            best_effort=False,
            interval=2.0,
            kind="cluster",
            max_rounds=5,
        )

    def test_inspect_cli_no_seeds(self):
        """Test CLI entry point without seeds (should show error)"""
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_cli(subcommand="cluster", seeds=None)
            output = buffer.getvalue()

            assert "Error: --seeds is required" in output
        finally:
            sys.stdout = old_stdout

    def test_inspect_cli_unknown_subcommand(self):
        """Test CLI entry point with unknown subcommand"""
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            inspect_cli(subcommand="unknown", seeds="127.0.0.1:8000")
            output = buffer.getvalue()

            assert "Unknown subcommand" in output or "Error" in output
        finally:
            sys.stdout = old_stdout

    def test_inspect_cli_seeds_parsing(self):
        """Test that seeds are properly parsed (comma-separated, trimmed)"""
        with patch("pulsing.cli.inspect.inspect_cluster") as mock_inspect:
            inspect_cli(
                subcommand="cluster",
                seeds="127.0.0.1:8000, 127.0.0.1:8001 , 127.0.0.1:8002",
            )

            # Should parse and trim seeds
            mock_inspect.assert_called_once()
            call_args = mock_inspect.call_args[0][
                0
            ]  # First positional arg (seeds list)
            assert len(call_args) == 3
            assert "127.0.0.1:8000" in call_args
            assert "127.0.0.1:8001" in call_args
            assert "127.0.0.1:8002" in call_args
            # Should be trimmed (no spaces)
            assert all(" " not in seed for seed in call_args)
