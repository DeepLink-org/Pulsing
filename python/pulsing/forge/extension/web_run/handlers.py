# SPDX-License-Identifier: Apache-2.0
"""``web.run`` — client web search / open / find (Codex ext/web-search).

Codex ships no open-source implementation of this tool: the ChatGPT client
browses/searches via a hosted product service. This Forge implementation only
covers the ``open`` operation (direct HTTPS fetch), gated behind an explicit
host allowlist. ``search``/``find`` remain unimplemented and return a clear
error rather than a silent no-op.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pulsing.forge.context import ToolCallContext
from pulsing.forge.result import ToolResult

_FETCH_CAP = 512 * 1024
_TIMEOUT_SECONDS = 30


def _allowed_hosts() -> set[str]:
    raw = os.environ.get(
        "FORGE_WEB_ALLOW", os.environ.get("PULSING_CRAFT_FETCH_ALLOW", "")
    ).strip()
    return {h.strip().lower().rstrip(".") for h in raw.split(",") if h.strip()}


def _host_matches_allowlist(host: str, allowed: set[str]) -> bool:
    """Match Codex network-proxy domain patterns: exact, *.apex, **.apex."""
    host = host.lower().rstrip(".")
    for pattern in allowed:
        if pattern.startswith("**."):
            apex = pattern[3:]
            if host == apex or host.endswith(f".{apex}"):
                return True
        elif pattern.startswith("*."):
            apex = pattern[2:]
            if host != apex and host.endswith(f".{apex}"):
                return True
        elif host == pattern:
            return True
    return False


def _resolve_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        raw_addr = info[4][0].split("%", 1)[0]  # strip IPv6 zone id
        try:
            ips.append(ipaddress.ip_address(raw_addr))
        except ValueError:
            continue
    return ips


def _ipv4_in_cidr(
    ip: ipaddress.IPv4Address, base: tuple[int, int, int, int], prefix: int
) -> bool:
    ip_int = int(ip)
    base_int = (base[0] << 24) | (base[1] << 16) | (base[2] << 8) | base[3]
    mask = 0 if prefix == 0 else (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF
    return (ip_int & mask) == (base_int & mask)


def _is_non_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Align with codex-rs/network-proxy policy.rs (CGNAT, TEST-NET, etc.).
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            return _is_non_public(mapped)
        return (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_reserved
        )
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
        or ip == ipaddress.IPv4Address("255.255.255.255")
        or _ipv4_in_cidr(ip, (0, 0, 0, 0), 8)
        or _ipv4_in_cidr(ip, (100, 64, 0, 0), 10)
        or _ipv4_in_cidr(ip, (192, 0, 0, 0), 24)
        or _ipv4_in_cidr(ip, (192, 0, 2, 0), 24)
        or _ipv4_in_cidr(ip, (198, 18, 0, 0), 15)
        or _ipv4_in_cidr(ip, (198, 51, 100, 0), 24)
        or _ipv4_in_cidr(ip, (203, 0, 113, 0), 24)
        or _ipv4_in_cidr(ip, (240, 0, 0, 0), 4)
    )


_BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata", "metadata.google.internal"})


def _reject_ssrf_target(host: str) -> str | None:
    """Return an error message if ``host`` resolves to a non-public address, else None.

    Blocks direct IP-literal access and DNS names that resolve to loopback,
    private, link-local (e.g. the ``169.254.169.254`` cloud metadata service),
    or other non-routable ranges. This runs both before the initial request
    and before following any redirect.
    """
    host = host.rstrip(".").lower()
    if (
        host in _BLOCKED_HOSTNAMES
        or host.endswith(".localhost")
        or host.endswith(".local")
    ):
        return f"host {host!r} is blocked to prevent SSRF"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if _is_non_public(ip):
            return (
                f"host {host!r} is a non-public address ({ip}); blocked to prevent SSRF"
            )
        return None
    ips = _resolve_ips(host)
    if not ips:
        return f"could not resolve host {host!r}"
    blocked = [ip for ip in ips if _is_non_public(ip)]
    if blocked:
        return f"host {host!r} resolves to a non-public address ({blocked[0]}); blocked to prevent SSRF"
    return None


def _check_url_allowed(url: str, allowed: set[str]) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "web.run open: only http/https URLs are supported"
    if parsed.username or parsed.password:
        return "web.run open: URLs with embedded credentials are not allowed"
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return "web.run open: URL has no host"
    if not _host_matches_allowlist(host, allowed):
        return f"web.run open: host {host!r} is not in FORGE_WEB_ALLOW"
    return _reject_ssrf_target(host)


class _AllowlistRedirectHandler(HTTPRedirectHandler):
    """Re-validates every redirect hop against the same allowlist/SSRF checks.

    ``urllib`` follows redirects transparently by default, which would let an
    allowlisted host bounce the request to an internal address (e.g. cloud
    metadata) after the initial check already passed. Rejecting the redirect
    here means the request to the disallowed target is never made.
    """

    def __init__(self, allowed: set[str]) -> None:
        self._allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802 (stdlib override)
        error = _check_url_allowed(newurl, self._allowed)
        if error is not None:
            raise HTTPError(newurl, code, f"redirect blocked: {error}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_url(url: str) -> ToolResult:
    allowed = _allowed_hosts()
    if not allowed:
        return ToolResult(
            content=(
                "web.run open disabled: set FORGE_WEB_ALLOW (comma-separated hostnames) "
                "or configure standalone search via FORGE_WEB_SEARCH=1 and provider credentials."
            ),
            is_error=True,
        )
    url = url.strip()
    if not url:
        return ToolResult(content="web.run open: URL is empty", is_error=True)
    error = _check_url_allowed(url, allowed)
    if error is not None:
        return ToolResult(content=error, is_error=True)

    opener = build_opener(_AllowlistRedirectHandler(allowed))
    req = Request(url, headers={"User-Agent": "pulsing-forge-web-run/1.0"})
    try:
        with opener.open(req, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            max_bytes = _FETCH_CAP
            data = resp.read(max_bytes + 1)
    except HTTPError as exc:
        reason = str(exc.reason or "")
        if reason.startswith("redirect blocked:"):
            return ToolResult(content=f"web.run open: {reason}", is_error=True)
        return ToolResult(
            content=f"web.run open: HTTP {exc.code} fetching {url}: {reason}",
            is_error=True,
        )
    except TimeoutError:
        return ToolResult(
            content=f"web.run open: timed out after {_TIMEOUT_SECONDS}s fetching {url}",
            is_error=True,
        )
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return ToolResult(
                content=f"web.run open: timed out after {_TIMEOUT_SECONDS}s fetching {url}",
                is_error=True,
            )
        return ToolResult(
            content=f"web.run open: failed to fetch {url}: {exc.reason}",
            is_error=True,
        )
    except OSError as exc:
        return ToolResult(
            content=f"web.run open: failed to fetch {url}: {exc}", is_error=True
        )
    if len(data) > max_bytes:
        return ToolResult(
            content=f"web.run open: response from {url} exceeds {max_bytes} byte cap",
            is_error=True,
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return ToolResult(content=text)


def handle_web_run(*, ctx: ToolCallContext, **kwargs: Any) -> ToolResult:
    # Accept Codex SearchCommands-shaped payload (flat or nested under "commands").
    payload = dict(kwargs)
    if "commands" in payload and isinstance(payload["commands"], dict):
        payload = dict(payload["commands"])

    flat_url = payload.get("url")
    if (
        flat_url
        and payload.get("open") is None
        and not payload.get("search_query")
        and not payload.get("image_query")
        and not (payload.get("find") or [])
    ):
        payload = {**payload, "open": [{"url": str(flat_url)}]}

    if payload.get("search_query") or payload.get("image_query"):
        if os.environ.get("FORGE_WEB_SEARCH", "").strip() not in ("1", "true", "yes"):
            return ToolResult(
                content=(
                    "web.run search requires standalone search (Codex alpha/search). "
                    "Set FORGE_WEB_SEARCH=1 and wire provider auth in Craft, "
                    "or use hosted web_search on the model provider."
                ),
                is_error=True,
            )
        return ToolResult(
            content="standalone web search client not configured in Forge MVP",
            is_error=True,
        )

    open_ops = payload.get("open")
    if open_ops is not None:
        if not isinstance(open_ops, list):
            return ToolResult(
                content="web.run open: expected a list of open commands", is_error=True
            )
        if not open_ops:
            return ToolResult(
                content="web.run open: command list is empty", is_error=True
            )
        first = open_ops[0]
        if not isinstance(first, dict):
            return ToolResult(
                content="web.run open: each command must be an object with 'url' or 'ref_id'",
                is_error=True,
            )
        ref = str(first.get("ref_id") or first.get("url") or "").strip()
        if not ref:
            return ToolResult(
                content="web.run open: missing 'url' or 'ref_id'", is_error=True
            )
        if "://" in ref and not (
            ref.startswith("http://") or ref.startswith("https://")
        ):
            return ToolResult(
                content="web.run open: only http/https URLs are supported",
                is_error=True,
            )
        if ref.startswith("http://") or ref.startswith("https://"):
            return _fetch_url(ref)
        return ToolResult(
            content=(
                f"web.run open: ref_id {ref!r} is not a literal URL "
                "(turn refs require a search backend)"
            ),
            is_error=True,
        )

    find_ops = payload.get("find") or []
    if isinstance(find_ops, list) and find_ops:
        op = find_ops[0]
        if not isinstance(op, dict):
            return ToolResult(
                content="find op must be an object with 'ref_id' and 'pattern'",
                is_error=True,
            )
        return ToolResult(
            content=json.dumps(
                {
                    "status": "unsupported_in_mvp",
                    "ref_id": op.get("ref_id"),
                    "pattern": op.get("pattern"),
                    "hint": "find-in-page requires prior search results; use open with https URL",
                },
                indent=2,
            ),
            is_error=True,
        )

    if not payload:
        return ToolResult(content="empty web.run command", is_error=True)
    return ToolResult(
        content=json.dumps({"received": payload}, indent=2),
        is_error=True,
    )
