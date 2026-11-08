"""Choose the companion's listening address from a shared seed configuration."""

from __future__ import annotations

import errno
import ipaddress
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class Network:
    addr: str
    peers: list[str]
    is_seed: bool


def _parse_seed(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator:
        raise ValueError(f"seed must be an IP:port address: {value}")
    if host.startswith("[") != host.endswith("]"):
        raise ValueError(f"seed must be an IP:port address: {value}")
    host = host.removeprefix("[").removesuffix("]")
    try:
        ip = ipaddress.ip_address(host)
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"seed must be an IP:port address: {value}") from exc
    if not 1 <= port <= 65535 or ip.is_unspecified or ip.is_multicast:
        raise ValueError(f"seed must have a concrete IP and port 1–65535: {value}")
    return str(ip), port


def _address(ip: str, port: int) -> str:
    return f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"


def _family(ip: str):
    return socket.AF_INET6 if ":" in ip else socket.AF_INET


def _is_local(ip: str) -> bool:
    # Binding a UDP socket to an ephemeral port checks all configured interfaces,
    # including aliases/VPNs, without hostname guesses or sending any traffic.
    try:
        with socket.socket(_family(ip), socket.SOCK_DGRAM) as probe:
            probe.bind((ip, 0))
    except OSError as exc:
        if exc.errno in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, errno.ENODEV):
            return False
        raise
    return True


def _port_free(ip: str, port: int) -> bool:
    with socket.socket(_family(ip), socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((ip, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def _route_address(ip: str, port: int) -> str:
    # UDP connect consults the routing table; it sends no application packet.
    # The source address is the interface the OS would use to reach this seed.
    with socket.socket(_family(ip), socket.SOCK_DGRAM) as probe:
        probe.connect((ip, port))
        return probe.getsockname()[0]


def choose_network(seeds: list[str], addr: str | None = None) -> Network:
    """An available local seed address hosts; all other nodes join via routing."""
    endpoints = [_parse_seed(seed) for seed in seeds]
    canonical = [_address(ip, port) for ip, port in endpoints]
    for (ip, port), seed in zip(endpoints, canonical):
        if addr is not None and addr != seed:
            continue
        if _is_local(ip) and _port_free(ip, port):
            return Network(seed, [peer for peer in canonical if peer != seed], True)

    if addr is not None:
        return Network(addr, canonical, not canonical)
    if not canonical:
        return Network("127.0.0.1:0", [], True)

    errors = []
    for ip, port in endpoints:
        try:
            return Network(_address(_route_address(ip, port), 0), canonical, False)
        except OSError as exc:
            errors.append(str(exc))
    raise OSError(f"no route to configured seeds: {'; '.join(errors)}")
