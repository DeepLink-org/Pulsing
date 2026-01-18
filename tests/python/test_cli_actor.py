"""Test CLI actor command

Tests for the actor CLI command using function-level unit tests.
Since all CLI commands are functions (via hp.param), we can test them directly.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from pulsing.cli.__main__ import actor as actor_cli


class TestActorCLI:
    """Test actor CLI command"""

    def test_actor_list_deprecated(self, capsys):
        """Test that 'list' subcommand shows deprecation message"""
        actor_cli(
            actor_type="list",
            seeds="127.0.0.1:8000",
        )
        captured = capsys.readouterr()
        assert "pulsing actor list" in captured.out
        assert "pulsing inspect actors" in captured.out

    def test_actor_invalid_class_path(self):
        """Test error for invalid class path (no dots)"""
        with pytest.raises(ValueError, match="must be a full class path"):
            actor_cli(actor_type="router")

    def test_actor_invalid_class_path_message(self):
        """Test error message shows correct format"""
        with pytest.raises(ValueError) as exc_info:
            actor_cli(actor_type="router")
        assert "full class path" in str(exc_info.value)
        assert "pulsing.actors.worker.TransformersWorker" in str(exc_info.value)
