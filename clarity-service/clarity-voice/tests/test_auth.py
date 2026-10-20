"""鉴权：全部端点（含探针与 /metrics）强制 Bearer，未携带或密钥错误的请求 401。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import settings


def _speech_payload() -> dict:
    return {
        "model": "huayan",
        "input": "你好",
        "voice": "huayan",
        "response_format": "wav",
        "speed": 1.0,
        "sample_rate": 22050,
    }


def _requests(make_wav) -> list[tuple[str, str, dict]]:
    """覆盖全部端点（含请求体，确保失败原因是鉴权而非参数校验）"""
    return [
        ("GET", "/health", {}),
        ("GET", "/ready", {}),
        ("GET", "/metrics", {}),
        ("GET", "/v1/models", {}),
        ("GET", "/v1/status", {}),
        (
            "POST",
            "/v1/audio/transcriptions",
            {
                "files": {"file": ("a.wav", make_wav(), "audio/wav")},
                "data": {"model": "sensevoice", "hotwords": json.dumps(["去雾"])},
            },
        ),
        ("POST", "/v1/audio/speech", {"json": _speech_payload()}),
    ]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/metrics"),
        ("GET", "/v1/models"),
        ("GET", "/v1/status"),
        ("POST", "/v1/audio/transcriptions"),
        ("POST", "/v1/audio/speech"),
    ],
)
def test_all_endpoints_require_bearer(client: TestClient, make_wav, method, path) -> None:
    kwargs = next(kw for m, p, kw in _requests(make_wav) if (m, p) == (method, path))
    response = client.request(method, path, **kwargs)
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_wrong_key_rejected(client: TestClient, make_wav) -> None:
    for method, path, kwargs in _requests(make_wav):
        response = client.request(
            method, path, headers={"Authorization": "Bearer wrong-key"}, **kwargs
        )
        assert response.status_code == 401
        assert response.json()["code"] == "UNAUTHORIZED"


def test_valid_key_accepted(client: TestClient, auth_headers) -> None:
    for path in ("/health", "/ready", "/metrics", "/v1/models", "/v1/status"):
        assert client.get(path, headers=auth_headers).status_code == 200


def test_websocket_handshake_requires_bearer(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect("/v1/asr/stream?model_id=iic/SenseVoiceSmall") as ws,
    ):
        ws.receive_json()
    assert exc.value.code == 4401


def test_empty_key_fails_closed(client: TestClient, make_wav, monkeypatch) -> None:
    """密钥未配置（空串）时一律 401：空 token 与任意 token 都不得通过"""
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_KEY", "")
    for method, path, kwargs in _requests(make_wav):
        for headers in (
            {},
            {"Authorization": "Bearer "},
            {"Authorization": "Bearer local-no-key"},
        ):
            response = client.request(method, path, headers=headers, **kwargs)
            assert response.status_code == 401, path
            assert response.json()["code"] == "UNAUTHORIZED", path


def test_empty_key_rejects_websocket(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_KEY", "")
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect(
            "/v1/asr/stream?model_id=iic/SenseVoiceSmall", headers={"Authorization": "Bearer "}
        ) as ws,
    ):
        ws.receive_json()
    assert exc.value.code == 4401


def test_key_defaults_to_local_no_key() -> None:
    # 开发零摩擦：默认密钥保证单机直连可用
    assert settings.INTERNAL_SERVICE_KEY == "local-no-key"
