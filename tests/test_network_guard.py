"""Meta-tests: prove the session network guard actually blocks paid calls.

If these fail, every other test in the suite is one bug away from spending money.
"""

from __future__ import annotations

import os
import socket

import pytest

from tests import conftest

from tools.graphics.atlas_image import AtlasImage
from tools.video.atlas_video import AtlasVideo


class TestGuardBlocksOutbound:

    def test_raw_socket_connect_is_blocked(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(Exception) as exc:
            sock.connect(("api.atlascloud.ai", 443))
        assert "Blocked a network connection" in str(exc.value)

    def test_create_connection_is_blocked(self):
        with pytest.raises(Exception) as exc:
            socket.create_connection(("api.atlascloud.ai", 443), timeout=5)
        assert "Blocked a network connection" in str(exc.value)

    def test_requests_cannot_reach_a_provider(self):
        requests = pytest.importorskip("requests")
        with pytest.raises(Exception):
            requests.get("https://api.atlascloud.ai/api/v1/model/prediction/x", timeout=5)

    def test_loopback_still_permitted(self):
        """Local servers and fixtures must keep working."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            client.close()
        finally:
            server.close()


class TestPaidToolsCannotSpend:
    """The guard must hold even with a real key present in the environment."""

    def test_atlas_image_fails_instead_of_billing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "sk-looks-real-but-must-not-be-used")
        result = AtlasImage().execute({
            "prompt": "this must never reach the API",
            "output_path": str(tmp_path / "nope.png"),
        })
        assert result.success is False
        assert result.cost_usd == 0.0

    def test_atlas_video_fails_instead_of_billing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "sk-looks-real-but-must-not-be-used")
        result = AtlasVideo().execute({
            "prompt": "this must never reach the API",
            "output_path": str(tmp_path / "nope.mp4"),
        })
        assert result.success is False
        assert result.cost_usd == 0.0


class TestLiveApiMarkerIsSkipped:

    @pytest.mark.live_api
    def test_this_should_never_run_by_default(self):
        raise AssertionError(
            "A @live_api test executed without OPENMONTAGE_ALLOW_NETWORK=1 — "
            "the opt-in gate is broken and real spending is possible."
        )


class TestProxyBypassIsClosed:
    """A socket-layer guard is blind to proxies unless it defends against them.

    With HTTPS_PROXY set, an HTTP client connects to the proxy and names the
    real host inside a CONNECT request. A loopback proxy therefore looked like
    permitted local traffic, and paid calls sailed through a guard that still
    reported itself healthy.
    """

    def test_proxy_env_is_unset_during_the_session(self):
        """Defence 1: clients must resolve the true host, so they hit the wall."""
        for var in conftest._PROXY_ENV_VARS:
            assert var not in os.environ, (
                f"{var} is set during a guarded test session; HTTP clients would "
                f"connect to the proxy instead of the provider and slip past the "
                f"socket guard."
            )

    def test_configured_loopback_proxy_is_blocked(self, monkeypatch):
        """Defence 2: an explicitly-passed proxy is refused despite being local."""
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:33611")
        proxies = conftest._proxy_endpoints()
        assert conftest._matches_proxy(("127.0.0.1", 33611), proxies)
        # Same endpoint by another spelling of loopback.
        assert conftest._matches_proxy(("localhost", 33611), proxies)
        # A different local port is ordinary loopback traffic and stays allowed.
        assert not conftest._matches_proxy(("127.0.0.1", 8000), proxies)

    def test_proxy_without_explicit_port_matches_any_port(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "proxy.internal")
        proxies = conftest._proxy_endpoints()
        assert conftest._matches_proxy(("proxy.internal", 3128), proxies)
        assert conftest._matches_proxy(("proxy.internal", 8080), proxies)

    def test_no_proxy_configured_blocks_nothing_extra(self, monkeypatch):
        for var in conftest._PROXY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert conftest._proxy_endpoints() == set()
        assert not conftest._matches_proxy(("127.0.0.1", 33611), set())

    def test_malformed_proxy_url_does_not_crash_collection(self, monkeypatch):
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:not-a-port")
        conftest._proxy_endpoints()  # must not raise
