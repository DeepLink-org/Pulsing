"""Shared SEED configuration elects its owner and selects joiner interfaces."""

from __future__ import annotations

import errno
import socket
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest

from pulsing.cli import network


def test_local_seed_is_bound_without_self_peering(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: ip == "10.0.0.10")
    monkeypatch.setattr(network, "_port_free", lambda ip, port: True)
    chosen = network.choose_network(["10.0.0.10:8100"])
    assert chosen == network.Network("10.0.0.10:8100", [], True)


def test_remote_seed_uses_routed_interface_and_ephemeral_port(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: False)
    routed = Mock(return_value="10.0.0.11")
    monkeypatch.setattr(network, "_route_address", routed)
    chosen = network.choose_network(["10.0.0.10:8100"])
    assert chosen == network.Network("10.0.0.11:0", ["10.0.0.10:8100"], False)
    routed.assert_called_once_with("10.0.0.10", 8100)


def test_existing_local_seed_is_joined_instead_of_rebound(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: True)
    monkeypatch.setattr(network, "_port_free", lambda ip, port: False)
    monkeypatch.setattr(network, "_route_address", lambda ip, port: ip)
    chosen = network.choose_network(["127.0.0.1:8100"])
    assert chosen == network.Network("127.0.0.1:0", ["127.0.0.1:8100"], False)


def test_explicit_address_is_preserved(monkeypatch):
    local = Mock(
        side_effect=AssertionError("explicit join address must not be guessed")
    )
    monkeypatch.setattr(network, "_is_local", local)
    chosen = network.choose_network(["10.0.0.10:8100"], "10.0.0.11:8200")
    assert chosen == network.Network("10.0.0.11:8200", ["10.0.0.10:8100"], False)
    assert network.choose_network([], "127.0.0.1:8200").is_seed


def test_multiple_seed_addresses_do_not_require_remote_seed_online(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: ip == "10.0.0.10")
    monkeypatch.setattr(network, "_port_free", lambda ip, port: True)
    chosen = network.choose_network(["10.0.0.20:8100", "10.0.0.10:8100"])
    assert chosen == network.Network("10.0.0.10:8100", ["10.0.0.20:8100"], True)


def test_unreachable_route_tries_another_seed(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: False)

    def route(ip, port):
        if ip == "2001:db8::1":
            raise OSError(errno.ENETUNREACH, "IPv6 not routed")
        return "10.0.0.11"

    monkeypatch.setattr(network, "_route_address", route)
    peers = ["[2001:db8::1]:8100", "10.0.0.10:8100"]
    assert network.choose_network(peers) == network.Network("10.0.0.11:0", peers, False)


def test_ipv6_source_is_formatted_as_socket_address(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: False)
    monkeypatch.setattr(network, "_route_address", lambda ip, port: "2001:db8::2")
    chosen = network.choose_network(["[2001:db8::1]:8100"])
    assert chosen.addr == "[2001:db8::2]:0"


@pytest.mark.parametrize(
    "seed",
    [
        "localhost:8100",
        "10.0.0.10",
        "10.0.0.10:0",
        "10.0.0.10:65536",
        "0.0.0.0:8100",
        "[::]:8100",
        "224.0.0.1:8100",
        "[10.0.0.10:8100",
    ],
)
def test_invalid_seed_is_rejected_before_network_probes(seed, monkeypatch):
    monkeypatch.setattr(
        network, "_is_local", Mock(side_effect=AssertionError("must validate"))
    )
    with pytest.raises(ValueError, match="seed must"):
        network.choose_network([seed])


def test_no_seed_remains_local_only():
    assert network.choose_network([]) == network.Network("127.0.0.1:0", [], True)


def test_real_local_detection_and_port_occupancy():
    assert network._is_local("127.0.0.1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        seed = f"127.0.0.1:{listener.getsockname()[1]}"
        chosen = network.choose_network([seed])
        assert not chosen.is_seed
        assert chosen.addr == "127.0.0.1:0"
        assert chosen.peers == [seed]


async def test_losing_seed_bind_race_joins_before_starting_agent(monkeypatch, tmp_path):
    from pulsing.cli import launch
    from pulsing.mailbox import managed

    attempts = []
    joined = network.Network("10.0.0.10:0", ["10.0.0.10:8100"], False)

    @asynccontextmanager
    async def start(addr, state_dir, *, seeds, namespace):
        attempts.append((addr, seeds, namespace))
        if len(attempts) == 1:
            raise OSError(errno.EADDRINUSE, "another launcher won the seed port")
        yield "directory"

    monkeypatch.setattr(managed, "host_runtime", start)
    monkeypatch.setattr(launch, "choose_network", lambda seeds: joined)
    owner = network.Network("10.0.0.10:8100", [], True)
    async with launch.companion_runtime(owner, tmp_path, "agent", True) as result:
        assert result == ("directory", joined)
    assert attempts == [
        ("10.0.0.10:8100", [], "agent"),
        ("10.0.0.10:0", ["10.0.0.10:8100"], "agent"),
    ]


async def test_explicit_seed_bind_failure_is_not_silently_relocated(
    monkeypatch, tmp_path
):
    from pulsing.cli import launch
    from pulsing.mailbox import managed

    @asynccontextmanager
    async def start(*args, **kwargs):
        raise OSError(errno.EADDRINUSE, "explicit port is busy")
        yield

    monkeypatch.setattr(managed, "host_runtime", start)
    monkeypatch.setattr(
        launch, "choose_network", Mock(side_effect=AssertionError("must not retry"))
    )
    owner = network.Network("10.0.0.10:8100", [], True)
    with pytest.raises(OSError, match="explicit port is busy"):
        async with launch.companion_runtime(owner, tmp_path, "agent", False):
            pytest.fail("runtime should not start")


def test_missing_route_does_not_create_loopback_network(monkeypatch):
    monkeypatch.setattr(network, "_is_local", lambda ip: False)
    monkeypatch.setattr(
        network,
        "_route_address",
        Mock(side_effect=OSError(errno.ENETUNREACH, "no route")),
    )
    with pytest.raises(OSError, match="no route to configured seeds"):
        network.choose_network(["10.0.0.10:8100"])
