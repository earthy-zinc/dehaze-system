"""能力发现、引擎状态、探针与指标端点。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.engines import asr as asr_engine
from app.engines import tts as tts_engine


def test_models_reports_engine_capabilities(client: TestClient, auth_headers) -> None:
    entries = client.get("/v1/models", headers=auth_headers).json()["data"]
    assert [entry["type"] for entry in entries] == ["asr", "tts", "catalog"]
    # 模型/音色事实源在服务侧：/v1/models 暴露目录，编排侧不再下发运行时工件
    assert entries[2]["catalog"] == {
        "tts": {"huayan": ["huayan"]},
        "asr": ["paraformer", "sensevoice"],
    }

    asr_entry = entries[0]
    assert asr_entry["id"] == "asr"
    assert asr_entry["available"] is True
    assert asr_entry["downloadable"] is True
    assert asr_entry["loaded"] == []
    assert asr_entry["capabilities"]["streaming"] is True
    assert asr_entry["capabilities"]["hotwords"] is True
    assert asr_entry["capabilities"]["formats"] == ["wav", "pcm"]

    tts_entry = entries[1]
    assert tts_entry["id"] == "tts"
    assert tts_entry["capabilities"]["formats"] == ["mp3", "wav", "pcm"]
    assert 24000 in tts_entry["capabilities"]["sample_rates"]
    assert tts_entry["limits"]["speed_range"] == [0.5, 2.0]


def test_models_reports_cached_voices(
    client: TestClient, auth_headers, monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "huayan.onnx").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    monkeypatch.setattr(settings, "TTS_MODEL_DIR", str(tmp_path))

    entries = client.get("/v1/models", headers=auth_headers).json()["data"]
    tts_entry = next(entry for entry in entries if entry["type"] == "tts")
    assert tts_entry["cached"] == ["huayan"]
    assert tts_entry["capabilities"]["voices"] == ["huayan"]


def test_models_marks_unavailable_engine(client: TestClient, auth_headers, monkeypatch) -> None:
    monkeypatch.setattr(asr_engine, "dependency_error", lambda: "funasr 不可用: 未安装")
    asr_entry = client.get("/v1/models", headers=auth_headers).json()["data"][0]
    assert asr_entry["available"] is False
    assert asr_entry["unavailable_reason"] == "funasr 不可用: 未安装"


def test_status_reports_loaded_models(client: TestClient, auth_headers, monkeypatch) -> None:
    payload = client.get("/v1/status", headers=auth_headers).json()
    assert payload == {
        "asr": {"loaded": False, "model_id": None, "device": payload["asr"]["device"]},
        "tts": {"loaded": False, "voice_id": None, "device": payload["tts"]["device"]},
    }

    monkeypatch.setitem(asr_engine._models, "iic/SenseVoiceSmall", object())
    monkeypatch.setitem(tts_engine._models, "/tmp/huayan.onnx", object())
    monkeypatch.setattr(tts_engine, "_last_voice", "huayan")

    loaded = client.get("/v1/status", headers=auth_headers).json()
    assert loaded["asr"]["loaded"] is True
    assert loaded["asr"]["model_id"] == "iic/SenseVoiceSmall"
    assert loaded["tts"] == {
        "loaded": True,
        "voice_id": "huayan",
        "device": loaded["tts"]["device"],
    }


def test_health_ignores_engine_availability(client: TestClient, auth_headers, monkeypatch) -> None:
    """liveness 只看进程存活：引擎依赖缺失也返回 200，且不触发加载"""
    monkeypatch.setattr(asr_engine, "dependency_error", lambda: "funasr 不可用")
    monkeypatch.setattr(tts_engine, "dependency_error", lambda: "piper 不可用")
    response = client.get("/health", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_requires_engine_dependencies(client: TestClient, auth_headers, monkeypatch) -> None:
    assert client.get("/ready", headers=auth_headers).status_code == 200

    monkeypatch.setattr(tts_engine, "dependency_error", lambda: "piper-tts 不可用: 未安装")
    response = client.get("/ready", headers=auth_headers)
    assert response.status_code == 503
    assert response.json() == {"code": "NOT_READY", "message": "piper-tts 不可用: 未安装"}


def test_metrics_exposes_voice_series(client: TestClient, auth_headers, asr_spy, make_wav) -> None:
    client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("a.wav", make_wav(), "audio/wav")},
        data={"model": "sensevoice"},
    )
    body = client.get("/metrics", headers=auth_headers).text
    assert 'dehaze_http_requests_total{endpoint="/v1/audio/transcriptions",status="200"}' in body
    assert 'dehaze_http_request_duration_seconds_count{endpoint="/v1/audio/transcriptions"}' in body
    assert 'dehaze_inference_duration_seconds_count{engine="asr"}' in body
