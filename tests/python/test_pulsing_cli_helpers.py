# SPDX-License-Identifier: Apache-2.0
"""Tests for pulsing top-level CLI helpers."""

from __future__ import annotations

from pulsing.cli.actor_argv import rewrite_actor_argv


def test_rewrite_actor_argv_injects_extra_kwargs() -> None:
    argv = [
        "pulsing",
        "actor",
        "pkg.Cls",
        "--addr",
        "0.0.0.0:8000",
        "--",
        "--model_name",
        "gpt2",
    ]
    out = rewrite_actor_argv(argv)
    assert out[1] == "actor"
    assert "-D" in out
    assert "actor.extra_kwargs" in out[-1]
    assert "model_name" in out[-1]


def test_rewrite_actor_argv_unchanged_without_dash_dash() -> None:
    argv = ["pulsing", "inspect", "cluster", "--seeds", "127.0.0.1:8000"]
    assert rewrite_actor_argv(argv) == argv
