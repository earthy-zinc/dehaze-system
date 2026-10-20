"""流式识别（WS /v1/asr/stream）：config 消息、分段下行、EOS 收尾、幂等与兜底超时。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from app.engines import asr as asr_engine
from app.errors import ModelUnavailableError

_URL = "/v1/asr/stream"


def _pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(16000 * seconds)


def _connect(client: TestClient, headers: dict, model_id: str = "sensevoice"):
    return client.websocket_connect(f"{_URL}?model_id={model_id}", headers=headers)


def test_stream_partials_then_final_on_eos(client: TestClient, auth_headers, asr_spy) -> None:
    with _connect(client, auth_headers, "paraformer") as ws:
        ws.send_json({"type": "config", "hotwords": ["去雾", "图像"], "session_id": "s-1"})
        ws.send_bytes(_pcm(1))
        assert ws.receive_json() == {"text": "识别1", "is_final": False}
        ws.send_bytes(_pcm(1))
        assert ws.receive_json() == {"text": "识别1识别2", "is_final": False}
        ws.send_text("EOS")
        assert ws.receive_json() == {"text": "识别1识别2", "is_final": True}

    assert asr_spy.preload == ["iic/speech_seaco_paraformer_large_asr_nat-zh"]
    assert {call["model_id"] for call in asr_spy.transcribe} == {
        "iic/speech_seaco_paraformer_large_asr_nat-zh"
    }
    assert all(call["hotwords"] == ["去雾", "图像"] for call in asr_spy.transcribe)


def test_stream_transcribes_residual_audio_on_eos(
    client: TestClient, auth_headers, asr_spy
) -> None:
    """不足一个分段的尾音在 EOS 时补转一次，不重复转写已识别音频"""
    with _connect(client, auth_headers) as ws:
        ws.send_json({"type": "config", "hotwords": []})
        ws.send_bytes(_pcm(0.5))
        ws.send_text("EOS")
        assert ws.receive_json() == {"text": "识别1", "is_final": True}
    assert len(asr_spy.transcribe) == 1
    assert len(asr_spy.transcribe[0]["audio"]) == 16000


def test_stream_hotwords_isolated_per_session(client: TestClient, auth_headers, asr_spy) -> None:
    """多会话并发时热词互不串扰（引擎侧无全局热词表）"""
    with (
        _connect(client, auth_headers, "sensevoice") as first,
        _connect(client, auth_headers, "paraformer") as second,
    ):
        first.send_json({"type": "config", "hotwords": ["甲"]})
        second.send_json({"type": "config", "hotwords": ["乙"]})
        first.send_bytes(_pcm(1))
        second.send_bytes(_pcm(1))
        assert first.receive_json()["is_final"] is False
        assert second.receive_json()["is_final"] is False

    # model_id 已由服务端目录解析为 ModelScope 标识
    assert {(call["model_id"], tuple(call["hotwords"])) for call in asr_spy.transcribe} == {
        ("iic/SenseVoiceSmall", ("甲",)),
        ("iic/speech_seaco_paraformer_large_asr_nat-zh", ("乙",)),
    }


def test_stream_duplicate_session_id_rejected(client: TestClient, auth_headers, asr_spy) -> None:
    with _connect(client, auth_headers) as first:
        first.send_json({"type": "config", "hotwords": [], "session_id": "dup"})
        with _connect(client, auth_headers) as second:
            second.send_json({"type": "config", "hotwords": [], "session_id": "dup"})
            assert second.receive_json() == {
                "code": "SESSION_CONFLICT",
                "message": "会话已在进行中: dup",
            }
            with pytest.raises(WebSocketDisconnect) as exc:
                second.receive_json()
            assert exc.value.code == 4409


def test_stream_requires_model_id(client: TestClient, auth_headers, asr_spy) -> None:
    with client.websocket_connect(_URL, headers=auth_headers) as ws:
        assert ws.receive_json()["code"] == "PARAM_ERROR"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4400


def test_stream_rejects_model_outside_catalog(client: TestClient, auth_headers, asr_spy) -> None:
    """未登记的 model_id 不做静默透传：PARAM_ERROR + 4400"""
    with _connect(client, auth_headers, "iic/SenseVoiceSmall") as ws:
        assert ws.receive_json()["code"] == "PARAM_ERROR"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4400


def test_stream_requires_config_first(client: TestClient, auth_headers, asr_spy) -> None:
    with _connect(client, auth_headers) as ws:
        ws.send_bytes(_pcm(1))
        assert ws.receive_json()["code"] == "PARAM_ERROR"


def test_stream_rejects_invalid_config(client: TestClient, auth_headers, asr_spy) -> None:
    with _connect(client, auth_headers) as ws:
        ws.send_json({"type": "config", "hotwords": "去雾"})
        assert ws.receive_json()["code"] == "PARAM_ERROR"


def test_stream_model_unavailable_closes_with_4503(
    client: TestClient, auth_headers, asr_spy, monkeypatch
) -> None:
    async def fake_preload(model_id: str) -> None:
        raise ModelUnavailableError(f"加载 FunASR 模型失败: {model_id}")

    monkeypatch.setattr(asr_engine, "preload", fake_preload)
    with _connect(client, auth_headers) as ws:
        ws.send_json({"type": "config", "hotwords": []})
        assert ws.receive_json()["code"] == "MODEL_UNAVAILABLE"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4503


def test_stream_idle_timeout_finishes_session(
    client: TestClient, auth_headers, asr_spy, monkeypatch
) -> None:
    """引擎侧仅"无输入自动收尾"兜底：停止上行后由引擎收尾并给出最终结果"""
    monkeypatch.setattr(settings, "ASR_STREAM_IDLE_TIMEOUT", 0.2)
    with _connect(client, auth_headers) as ws:
        ws.send_json({"type": "config", "hotwords": []})
        ws.send_bytes(_pcm(1))
        assert ws.receive_json()["is_final"] is False
        assert ws.receive_json() == {"text": "识别1", "is_final": True}
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
