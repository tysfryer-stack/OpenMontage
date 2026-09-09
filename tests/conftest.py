"""Session-wide test safety net.

**No test may open a network connection.** Provider tools bill per call, so a
test that reaches a real endpoint costs the developer money — silently, and
every time CI runs. This blocks outbound sockets for the whole test session.

The guard is at the socket layer on purpose. Patching `requests` only covers
tools that use `requests`; the fleet also talks to vendor SDKs (google-cloud,
openai, boto3), `httpx`, and raw `urllib`. Everything bottoms out in
`socket.connect`, so that is where the wall goes.

Loopback is still allowed — local servers, ffmpeg RPC, and Backlot fixtures need it.

**Proxies are the hole in a socket-layer guard.** When `HTTPS_PROXY` (or any
sibling) is set, an HTTP client does not connect to the provider at all: it
connects to the proxy and passes the real host in a CONNECT request. The socket
guard only ever sees the proxy's address, so a proxy on loopback — this is the
normal shape for corporate egress proxies, mitmproxy, and sandboxed CI — is
waved straight through to the paid endpoint. The guard reports itself healthy
while billing the developer.

Two defences, because either alone leaves a gap:

1. Proxy environment variables are unset for the duration of the session, so
   clients resolve and connect to the true destination host and meet the wall.
2. Connections to a configured proxy endpoint are refused even though it is on
   loopback, which catches a client that was handed an explicit `proxies=`
   argument or that cached the environment at import time.

To write a test that genuinely hits a live API:

    @pytest.mark.live_api
    def test_real_call():
        ...

Marked tests are **skipped by default** and only run with the env flag set:

    OPENMONTAGE_ALLOW_NETWORK=1 pytest -m live_api

In that mode the proxy variables are left intact — a live call may well need
the proxy to leave the network at all.

Limitations: this guards the pytest process. A test that shells out to a
subprocess (node, ffmpeg, npx) is outside its reach — don't call paid APIs
from a subprocess in tests. A client pointed at a loopback proxy that appears
in no environment variable is likewise invisible to defence 2; defence 1 is
what makes that case unlikely rather than impossible.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

_ALLOW_ENV_FLAG = "OPENMONTAGE_ALLOW_NETWORK"

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}

# Unset for the session so HTTP clients address the real host, not a proxy.
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "ftp_proxy",
)

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection


class NetworkCallInTestError(RuntimeError):
    """Raised when a test tries to open a non-loopback connection."""


def _network_allowed() -> bool:
    return os.environ.get(_ALLOW_ENV_FLAG, "").strip().lower() in {"1", "true", "yes"}


def _canonical_host(host: str) -> str:
    """Collapse every spelling of loopback to one token so hosts compare equal."""
    host = host.strip("[]").lower()
    if host in _LOOPBACK_HOSTS or host.startswith("127."):
        return "\0loopback"
    return host


def _split_address(address) -> tuple[str, int | None] | None:
    """Return (canonical_host, port) for a TCP/UDP target, or None if not one."""
    if isinstance(address, (str, bytes)):
        return None  # AF_UNIX / abstract socket — local by definition
    if not isinstance(address, (tuple, list)) or not address:
        return None  # unrecognised shape; let the real call decide
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    if not isinstance(host, str):
        return ("\0unknown", None)
    port = address[1] if len(address) > 1 and isinstance(address[1], int) else None
    return (_canonical_host(host), port)


def _proxy_endpoints() -> set[tuple[str, int | None]]:
    """Every (host, port) named by a proxy environment variable.

    A variable without an explicit port yields port None, which matches any
    port on that host — blocking too much here is the safe direction.
    """
    endpoints: set[tuple[str, int | None]] = set()
    for var in _PROXY_ENV_VARS:
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        if not parsed.hostname:
            continue
        try:
            port = parsed.port
        except ValueError:  # malformed port — treat as any-port
            port = None
        endpoints.add((_canonical_host(parsed.hostname), port))
    return endpoints


def _is_loopback(address) -> bool:
    """True for loopback TCP/UDP targets and for AF_UNIX socket paths."""
    split = _split_address(address)
    if split is None:
        return True
    return split[0] == "\0loopback"


def _matches_proxy(address, proxies: set[tuple[str, int | None]]) -> bool:
    split = _split_address(address)
    if split is None or not proxies:
        return False
    host, port = split
    return (host, port) in proxies or (host, None) in proxies


def _blocked(address) -> NetworkCallInTestError:
    return NetworkCallInTestError(
        f"Blocked a network connection to {address!r} during a test.\n"
        f"\n"
        f"Tests must not call real endpoints — provider APIs bill per request.\n"
        f"Mock the transport instead (see tests/tools/test_atlas_video.py for the\n"
        f"fake-`requests` pattern), or mark the test @pytest.mark.live_api and run\n"
        f"it deliberately with {_ALLOW_ENV_FLAG}=1."
    )


def _blocked_proxy(address) -> NetworkCallInTestError:
    return NetworkCallInTestError(
        f"Blocked a network connection to {address!r} during a test.\n"
        f"\n"
        f"That address is a configured HTTP proxy. Reaching a provider through a\n"
        f"proxy still bills for the request — the proxy only hides the real host\n"
        f"from this guard, it does not make the call free.\n"
        f"\n"
        f"The client was most likely handed an explicit `proxies=` argument, since\n"
        f"the proxy environment variables are unset for the test session. Mock the\n"
        f"transport instead, or mark the test @pytest.mark.live_api and run it\n"
        f"deliberately with {_ALLOW_ENV_FLAG}=1."
    )


@pytest.fixture(scope="session", autouse=True)
def _block_network():
    """Refuse non-loopback sockets, and proxy hops, for the entire session."""
    if _network_allowed():
        yield
        return

    proxies = _proxy_endpoints()
    saved_env = {var: os.environ[var] for var in _PROXY_ENV_VARS if var in os.environ}
    for var in saved_env:
        del os.environ[var]

    def check(address) -> None:
        if _matches_proxy(address, proxies):
            raise _blocked_proxy(address)
        if not _is_loopback(address):
            raise _blocked(address)

    def guarded_connect(self, address, *args, **kwargs):
        check(address)
        return _real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        check(address)
        return _real_connect_ex(self, address, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        check(address)
        return _real_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    try:
        yield
    finally:
        socket.socket.connect = _real_connect
        socket.socket.connect_ex = _real_connect_ex
        socket.create_connection = _real_create_connection
        os.environ.update(saved_env)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live_api: test performs a real, billable API call. Skipped unless "
        f"{_ALLOW_ENV_FLAG}=1 is set.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip live_api tests unless the operator explicitly opted in."""
    if _network_allowed():
        return
    skip = pytest.mark.skip(
        reason=f"live API test — costs money; set {_ALLOW_ENV_FLAG}=1 to run"
    )
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip)
