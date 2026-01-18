"""Test REST API endpoints for actor list and cluster info

Note: Most functions have been moved to pulsing.cli.inspect.
This file is kept for backward compatibility testing of _print_actors_table.
"""

import pytest
from pulsing.cli.inspect import _print_actors_table


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

    # Note: Tests for removed functions (http_get_sync, query_single_endpoint, query_cluster, _print_output)
    # have been removed. These functions are now in pulsing.cli.inspect.
    # Use 'pulsing inspect actors' for CLI usage.
