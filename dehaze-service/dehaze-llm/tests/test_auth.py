"""鉴权契约：全部端点校验 Authorization: Bearer，缺失/错误一律 401。"""

import pytest

from app.config import settings

PROTECTED = [
    ("GET", "/health"),
    ("GET", "/ready"),
    ("GET", "/v1/models"),
    ("GET", "/metrics"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/rerank"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
async def test_missing_header_rejected(raw_client, method, path):
    resp = await raw_client.request(method, path, json={})
    assert resp.status_code == 401
    assert resp.json() == {"code": "UNAUTHORIZED", "message": "未授权：内部服务密钥缺失或错误"}


@pytest.mark.parametrize(("method", "path"), PROTECTED)
async def test_wrong_key_rejected(raw_client, method, path):
    resp = await raw_client.request(
        method, path, json={}, headers={"Authorization": "Bearer wrong-key"}
    )
    assert resp.status_code == 401


async def test_custom_header_no_longer_accepted(raw_client):
    """自定义请求头不再是鉴权载体（旧 X-MCP-Key 式用法必须被拒）。"""
    resp = await raw_client.get("/health", headers={"X-Internal-Key": "test-internal-key"})
    assert resp.status_code == 401
    resp = await raw_client.get(
        "/health", headers={"Authorization": "test-internal-key"}
    )  # 缺 Bearer 前缀
    assert resp.status_code == 401


async def test_correct_key_passes(raw_client, chat_llm):
    resp = await raw_client.get("/health", headers={"Authorization": "Bearer test-internal-key"})
    assert resp.status_code == 200


async def test_empty_configured_key_fails_closed(raw_client, monkeypatch):
    """密钥未配置时全部拒绝（fail closed），不因空密钥退化为无鉴权。"""
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_KEY", "")
    resp = await raw_client.get("/health", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401
