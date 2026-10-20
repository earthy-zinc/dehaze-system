"""鉴权：所有端点校验 Authorization: Bearer ${INTERNAL_SERVICE_KEY}。"""

import pytest

from app.config import settings

_UNAUTHORIZED = {"code": "UNAUTHORIZED", "message": "未授权：内部服务密钥缺失或错误"}


def test_missing_bearer_rejected(client):
    response = client.get("/health")

    assert response.status_code == 401
    assert response.json() == _UNAUTHORIZED


def test_wrong_bearer_rejected(client):
    response = client.get("/health", headers={"Authorization": "Bearer wrong-key"})

    assert response.status_code == 401
    assert response.json() == _UNAUTHORIZED


def test_non_bearer_scheme_rejected(client):
    response = client.get("/health", headers={"Authorization": "Basic bG9jYWw="})

    assert response.status_code == 401
    assert response.json() == _UNAUTHORIZED


def test_valid_bearer_allows_request(client, auth_headers):
    response = client.get("/health", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": settings.APP_NAME}


@pytest.mark.parametrize("path", ["/ready", "/metrics", "/v1/models"])
def test_all_endpoints_require_bearer(client, path):
    assert client.get(path).status_code == 401


def test_unconfigured_key_fails_closed(client, monkeypatch):
    """密钥未配置（空串）时全部 401：不因空密钥退化为无鉴权"""
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_KEY", "")

    blank_token = {"Authorization": "Bearer "}
    configured_key = {"Authorization": "Bearer local-no-key"}

    assert client.get("/health").status_code == 401
    assert client.get("/health", headers=blank_token).status_code == 401
    assert client.get("/health", headers=configured_key).status_code == 401
