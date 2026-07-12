# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``web.run`` Forge tool (open/find/search + SSRF guards).

Codex ships no open-source implementation of ``web.run`` to compare against
(it's a hosted ChatGPT product feature); these tests instead pin down Forge's
own contract: allowlisted ``open`` fetches, and no request is ever made to a
non-public address regardless of configuration.
"""

from __future__ import annotations

from typing import Any

import ipaddress

import pytest

from pulsing.forge.extension.web_run import handlers as web_run

pytestmark = pytest.mark.forge

_PUBLIC_IP = ipaddress.ip_address(
    "93.184.216.34"
)  # example.com A record (IANA reserved doc range)


@pytest.fixture
def mock_public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub DNS so fetch tests never hit the network or local resolver quirks."""

    def _public_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return [_PUBLIC_IP]
        return [ip]

    monkeypatch.setattr(web_run, "_resolve_ips", _public_ips)


class _FakeResponse:
    def __init__(self, body: bytes, url: str = "") -> None:
        self._body = body
        self.url = url

    def read(self, n: int) -> bytes:
        return self._body[:n]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeOpener:
    def __init__(self, body: bytes = b"", exc: Exception | None = None) -> None:
        self._body = body
        self._exc = exc

    def open(self, req: Any, timeout: float | None = None) -> _FakeResponse:
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._body)


@pytest.fixture(autouse=True)
def _clean_web_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FORGE_WEB_ALLOW", "PULSING_CRAFT_FETCH_ALLOW", "FORGE_WEB_SEARCH"):
        monkeypatch.delenv(var, raising=False)


def test_open_fetches_allowlisted_host(
    monkeypatch: pytest.MonkeyPatch, local_forge, mock_public_dns
) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    monkeypatch.setattr(
        web_run, "build_opener", lambda *_: _FakeOpener(body=b"hello world")
    )

    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://example.com/page"}]}
    )

    assert not out.is_error
    assert out.content == "hello world"


def test_open_without_allowlist_is_disabled_by_default(local_forge) -> None:
    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://example.com/page"}]}
    )

    assert out.is_error
    assert "FORGE_WEB_ALLOW" in out.content


def test_open_rejects_host_not_in_allowlist(
    monkeypatch: pytest.MonkeyPatch, local_forge
) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")

    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://evil.example.org/x"}]}
    )

    assert out.is_error
    assert "evil.example.org" in out.content
    assert "is not in FORGE_WEB_ALLOW" in out.content


def test_allowlist_wildcard_subdomain(monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = {"*.example.com"}
    assert web_run._host_matches_allowlist("api.example.com", allowed)
    assert not web_run._host_matches_allowlist("example.com", allowed)
    assert not web_run._host_matches_allowlist("evil.example.org", allowed)


def test_allowlist_wildcard_apex_and_subdomains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = {"**.example.com"}
    assert web_run._host_matches_allowlist("example.com", allowed)
    assert web_run._host_matches_allowlist("api.example.com", allowed)


def test_open_blocks_cgnat_ip_even_if_allowlisted() -> None:
    error = web_run._check_url_allowed("http://100.64.0.1/", {"100.64.0.1"})
    assert error is not None
    assert "non-public" in error or "SSRF" in error


def test_open_accepts_flat_url_arg(
    monkeypatch: pytest.MonkeyPatch, local_forge, mock_public_dns
) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    monkeypatch.setattr(
        web_run, "build_opener", lambda *_: _FakeOpener(body=b"flat ok")
    )

    out = local_forge.call_tool("web.run", {"url": "https://example.com/flat"})

    assert not out.is_error
    assert out.content == "flat ok"


def test_redirect_blocked_reports_clearly(
    monkeypatch: pytest.MonkeyPatch, local_forge, mock_public_dns
) -> None:
    from urllib.error import HTTPError

    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    err = HTTPError(
        "http://169.254.169.254/",
        302,
        "redirect blocked: host '169.254.169.254' is a non-public address",
        {},
        None,
    )
    monkeypatch.setattr(web_run, "build_opener", lambda *_: _FakeOpener(exc=err))

    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://example.com/redirect"}]}
    )

    assert out.is_error
    assert "redirect blocked" in out.content
    assert "non-public" in out.content


def test_open_rejects_non_http_scheme(local_forge) -> None:
    out = local_forge.call_tool("web.run", {"open": [{"url": "file:///etc/passwd"}]})

    assert out.is_error
    assert "only http/https" in out.content


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://127.0.0.1:8080/",
        "http://0.0.0.0/",
        "http://[::1]/",
    ],
)
def test_open_blocks_ssrf_to_non_public_ip_even_if_allowlisted(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    host = web_run.urlparse(url).hostname or ""
    monkeypatch.setenv("FORGE_WEB_ALLOW", host)

    error = web_run._check_url_allowed(url, {host})

    assert error is not None
    assert "non-public" in error or "SSRF" in error


def test_check_url_allowed_accepts_public_looking_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 8.8.8.8 is a real public IP (Google DNS); used only to exercise the
    # "not blocked" branch of the IP classifier without a real network call.
    error = web_run._check_url_allowed("http://8.8.8.8/", {"8.8.8.8"})
    assert error is None


def test_redirect_to_disallowed_host_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_run, "_resolve_ips", lambda host: [])
    handler = web_run._AllowlistRedirectHandler({"example.com"})

    with pytest.raises(web_run.HTTPError):
        handler.redirect_request(
            None, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data/"
        )


def test_open_reports_timeout_clearly(
    monkeypatch: pytest.MonkeyPatch, local_forge, mock_public_dns
) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    monkeypatch.setattr(
        web_run, "build_opener", lambda *_: _FakeOpener(exc=TimeoutError())
    )

    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://example.com/slow"}]}
    )

    assert out.is_error
    assert "timed out" in out.content


def test_open_reports_http_error_clearly(
    monkeypatch: pytest.MonkeyPatch, local_forge, mock_public_dns
) -> None:
    from urllib.error import HTTPError

    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    err = HTTPError("https://example.com/404", 404, "Not Found", {}, None)
    monkeypatch.setattr(web_run, "build_opener", lambda *_: _FakeOpener(exc=err))

    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://example.com/404"}]}
    )

    assert out.is_error
    assert "404" in out.content


def test_open_response_over_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch, local_forge, mock_public_dns
) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    monkeypatch.setattr(web_run, "_FETCH_CAP", 4)
    monkeypatch.setattr(
        web_run, "build_opener", lambda *_: _FakeOpener(body=b"way too long")
    )

    out = local_forge.call_tool(
        "web.run", {"open": [{"url": "https://example.com/big"}]}
    )

    assert out.is_error
    assert "cap" in out.content


def test_search_query_disabled_by_default(local_forge) -> None:
    out = local_forge.call_tool("web.run", {"search_query": "pulsing forge"})

    assert out.is_error
    assert "FORGE_WEB_SEARCH" in out.content


def test_find_op_reports_unsupported(local_forge) -> None:
    out = local_forge.call_tool(
        "web.run", {"find": [{"ref_id": "turn0", "pattern": "x"}]}
    )

    assert out.is_error
    assert "unsupported_in_mvp" in out.content


def test_open_op_must_be_object(local_forge) -> None:
    out = local_forge.call_tool("web.run", {"open": ["not-an-object"]})

    assert out.is_error
    assert "object" in out.content


def test_open_rejects_localhost_hostname() -> None:
    error = web_run._check_url_allowed("http://localhost/", {"localhost"})
    assert error is not None
    assert "SSRF" in error


def test_open_rejects_url_with_credentials() -> None:
    error = web_run._check_url_allowed("http://user:pass@example.com/", {"example.com"})
    assert error is not None
    assert "credentials" in error


def test_open_rejects_empty_url(monkeypatch: pytest.MonkeyPatch, local_forge) -> None:
    monkeypatch.setenv("FORGE_WEB_ALLOW", "example.com")
    out = local_forge.call_tool("web.run", {"open": [{"url": "   "}]})
    assert out.is_error
    assert "missing" in out.content


def test_empty_command_is_rejected(local_forge) -> None:
    out = local_forge.call_tool("web.run", {})

    assert out.is_error
