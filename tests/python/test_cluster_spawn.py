"""Unit tests for cluster child spawn env helpers."""

from pulsing.cluster_spawn import (
    build_cluster_child_env,
    normalize_seed_for_local_child,
)


def test_normalize_seed_loopback() -> None:
    assert normalize_seed_for_local_child("0.0.0.0:9123") == "127.0.0.1:9123"
    assert normalize_seed_for_local_child("127.0.0.1:8000") == "127.0.0.1:8000"


def test_build_cluster_child_env() -> None:
    env = build_cluster_child_env(
        child_addr="127.0.0.1:0",
        seed_addrs=["127.0.0.1:9000", "127.0.0.1:9001"],
        passphrase="secret",
        extra={"FOO": "bar"},
    )
    assert env["PULSING_NODE_ADDR"] == "127.0.0.1:0"
    assert env["PULSING_SEEDS"] == "127.0.0.1:9000,127.0.0.1:9001"
    assert env["PULSING_PASSPHRASE"] == "secret"
    assert env["FOO"] == "bar"
