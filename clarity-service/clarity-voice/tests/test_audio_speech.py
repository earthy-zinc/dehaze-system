"""语音合成（`POST /v1/audio/speech`）：OpenAI 字段、顶层扩展、编码/重采样、
服务端目录解析与参数校验。"""

from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.engines import tts as tts_engine

_URL = "/v1/audio/speech"


def _payload(**overrides) -> dict:
    payload = {
        "model": "huayan",
        "input": "你好，世界",
        "voice": "huayan",
        "response_format": "mp3",
        "speed": 1.0,
        "sample_rate": 22050,
    }
    payload.update(overrides)
    return payload


def test_speech_resolves_catalog_and_returns_mp3(client: TestClient, auth_headers, tts_spy) -> None:
    """不传 voice_params：权重来源由服务端目录按 (model, voice) 解析"""
    pytest.importorskip("lameenc")
    response = client.post(_URL, json=_payload(), headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    # MP3 帧同步字（0xFFE0 掩码）或 ID3 头
    assert response.content[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa", b"ID3")
    assert tts_spy.loads == [{"voice_id": "huayan", "onnx": "zh_CN-huayan-medium.onnx"}]


def test_speech_voice_params_overrides_catalog(client: TestClient, auth_headers, tts_spy) -> None:
    """顶层 voice_params 为调试覆盖：给了就用它，不走目录"""
    pytest.importorskip("lameenc")
    response = client.post(
        _URL,
        json=_payload(voice_params={"onnx": "custom.onnx", "urls": ["https://x/{filename}"]}),
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert tts_spy.loads[0]["onnx"] == "custom.onnx"


def test_speech_wav_resamples_to_requested_rate(
    client: TestClient, auth_headers, tts_spy, monkeypatch
) -> None:
    resampled: list[tuple[int, int]] = []
    monkeypatch.setattr(
        tts_engine,
        "_resample",
        lambda pcm, src_rate, dst_rate: resampled.append((src_rate, dst_rate)) or pcm,
    )
    response = client.post(
        _URL, json=_payload(response_format="wav", sample_rate=16000), headers=auth_headers
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert resampled == [(22050, 16000)]
    with wave.open(io.BytesIO(response.content)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)


def test_speech_pcm_output_is_raw_samples(
    client: TestClient, auth_headers, tts_spy, monkeypatch
) -> None:
    monkeypatch.setattr(
        tts_engine, "_resample", lambda pcm, src_rate, dst_rate: pytest.fail("原生采样率不应重采样")
    )
    response = client.post(
        _URL, json=_payload(response_format="pcm", sample_rate=22050), headers=auth_headers
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/L16"
    assert response.content == b"\x00\x10" * 800


def test_speech_maps_speed_to_length_scale(client: TestClient, auth_headers, tts_spy) -> None:
    client.post(_URL, json=_payload(speed=2.0), headers=auth_headers)
    assert tts_spy.configs[0].length_scale == 0.5


def test_speech_real_resample_keeps_duration_ratio() -> None:
    pytest.importorskip("torchaudio")
    pcm = b"\x00\x10" * 22050
    resampled = tts_engine._resample(pcm, 22050, 16000)
    assert abs(len(resampled) // 2 - 16000) <= 8


def test_internal_error_does_not_leak_detail(
    client: TestClient, auth_headers, tts_spy, monkeypatch
) -> None:
    """500 只回固定 message，异常原文仅进日志"""

    def boom(*args, **kwargs):
        raise RuntimeError("secret-trace /srv/weights/huayan.onnx 崩溃")

    monkeypatch.setattr(tts_engine, "run_synthesize", boom)
    response = client.post(_URL, json=_payload(), headers=auth_headers)
    assert response.status_code == 500
    assert response.json() == {"code": "INTERNAL_ERROR", "message": "内部错误"}
    assert "secret-trace" not in response.text


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"response_format": "ogg"}, "格式不支持"),
        ({"response_format": "opus"}, "OpenAI opus 不在本服务子集内"),
        ({"sample_rate": 12345}, "采样率不支持"),
        ({"speed": 3.0}, "倍速越界"),
        ({"speed": 0.1}, "倍速越界"),
        ({"input": "   "}, "空文本"),
        ({"input": "雾" * 10001}, "文本超长"),
        ({"voice_params": {}}, "voice_params 覆盖缺少 onnx"),
        ({"model": "not-a-model"}, "模型不在目录"),
        ({"voice": "not-a-voice"}, "音色不在目录"),
    ],
)
def test_speech_rejects_invalid_params(
    client: TestClient, auth_headers, tts_spy, overrides: dict, reason: str
) -> None:
    response = client.post(_URL, json=_payload(**overrides), headers=auth_headers)
    assert response.status_code == 400, reason
    assert response.json()["code"] == "PARAM_ERROR", reason
    assert tts_spy.loads == [], reason
