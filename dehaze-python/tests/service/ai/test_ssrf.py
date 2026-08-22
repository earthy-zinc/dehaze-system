import asyncio
import socket

import pytest
from fastapi import HTTPException

from app.service.ai import a2a_client as a2a_client_mod
from app.service.ai.a2a_client import A2AClient, A2AClientError
from app.utils import ssrf


@pytest.fixture
def patched_loop(monkeypatch):
    def apply(fn):
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", fn)

    return apply


def _infos(*ips):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ips]


@pytest.mark.asyncio
class TestIsSafeUrl:
    async def test_non_https_rejected(self):
        assert not await ssrf.is_safe_url("http://example.com")

    async def test_localhost_rejected(self):
        assert not await ssrf.is_safe_url("https://localhost:8080/a")

    async def test_loopback_ip_rejected(self):
        assert not await ssrf.is_safe_url("https://127.0.0.1/api")

    async def test_private_ip_rejected(self, monkeypatch):
        monkeypatch.setattr(ssrf, "_is_hostname", lambda h: False)
        assert not await ssrf.is_safe_url("https://192.168.1.10/x")
        assert not await ssrf.is_safe_url("https://10.0.0.5/x")
        assert not await ssrf.is_safe_url("https://172.16.3.4/x")

    async def test_link_local_ip_rejected(self, monkeypatch):
        monkeypatch.setattr(ssrf, "_is_hostname", lambda h: False)
        assert not await ssrf.is_safe_url("https://169.254.169.254/x")

    async def test_public_ip_allowed(self, monkeypatch):
        monkeypatch.setattr(ssrf, "_is_hostname", lambda h: False)
        assert await ssrf.is_safe_url("https://8.8.8.8/x")

    async def test_domain_resolving_to_internal_rejected(self, patched_loop):
        ssrf._RESOLVED_CACHE.clear()

        async def fake(host, port, **kw):
            return _infos("192.168.50.10")

        patched_loop(fake)
        assert not await ssrf.is_safe_url("https://evil.example.com/x")

    async def test_domain_resolving_to_public_allowed(self, patched_loop):
        ssrf._RESOLVED_CACHE.clear()

        async def fake(host, port, **kw):
            return _infos("93.184.216.34")

        patched_loop(fake)
        assert await ssrf.is_safe_url("https://example.com/x")

    async def test_mixed_resolution_rejected_if_any_internal(self, patched_loop):
        ssrf._RESOLVED_CACHE.clear()

        async def fake(host, port, **kw):
            return _infos("93.184.216.34", "10.1.2.3")

        patched_loop(fake)
        assert not await ssrf.is_safe_url("https://mixed.example.com/x")

    async def test_dns_failure_conservatively_rejected(self, patched_loop):
        ssrf._RESOLVED_CACHE.clear()

        async def boom(host, port, **kw):
            raise socket.gaierror("nxdomain")

        patched_loop(boom)
        assert not await ssrf.is_safe_url("https://nx.example.com/x")

    async def test_validate_https_url_raises_on_unsafe(self):
        with pytest.raises(HTTPException):
            await ssrf.validate_https_url("http://192.168.1.1")

    async def test_domain_ttl_cache(self, patched_loop):
        ssrf._RESOLVED_CACHE.clear()
        count = {"n": 0}

        async def fake(host, port, **kw):
            count["n"] += 1
            return _infos("93.184.216.34")

        patched_loop(fake)
        assert await ssrf.is_safe_url("https://cached.example.com/x")
        assert await ssrf.is_safe_url("https://cached.example.com/y")
        assert count["n"] == 1


class _FakeEndpoint:
    def __init__(self, base_url, credential="", auth_type=""):
        self.base_url = base_url
        self.credential = credential
        self.auth_type = auth_type
        self.id = 1


async def test_a2a_runtime_rejects_unsafe_url(monkeypatch):
    endpoint = _FakeEndpoint("http://192.168.1.1/a2a")

    async def unsafe(x):
        return False

    monkeypatch.setattr(a2a_client_mod, "is_safe_url", unsafe)
    client = A2AClient()
    with pytest.raises(A2AClientError):
        await client._rpc(endpoint, "message/send", {"messages": []})
    with pytest.raises(A2AClientError):
        await client.message_send(endpoint, [])
