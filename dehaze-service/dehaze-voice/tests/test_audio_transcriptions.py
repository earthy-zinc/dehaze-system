"""离线识别（`POST /v1/audio/transcriptions`）：OpenAI multipart 形态、模型目录解析、
热词与参数校验。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.engines import asr as asr_engine
from app.errors import AudioInvalidError

_URL = "/v1/audio/transcriptions"


def _post(client: TestClient, headers: dict, audio: bytes, **data):
    return client.post(
        _URL,
        headers=headers,
        files={"file": ("a.wav", audio, "audio/wav")},
        data=data,
    )


def test_transcriptions_resolves_model_and_passes_request_params(
    client: TestClient, auth_headers, asr_spy, make_wav
) -> None:
    """统一 model_id → ModelScope 标识由服务端目录解析后传给引擎"""
    audio = make_wav()
    response = _post(
        client,
        auth_headers,
        audio,
        model="paraformer",
        hotwords=json.dumps(["去雾", "增强"]),
    )
    assert response.status_code == 200
    assert response.json() == {"text": "识别1"}
    assert asr_spy.transcribe == [
        {
            "audio": audio,
            "model_id": "iic/speech_seaco_paraformer_large_asr_nat-zh",
            "hotwords": ["去雾", "增强"],
        }
    ]


def test_transcriptions_without_hotwords(client: TestClient, auth_headers, asr_spy, make_wav):
    assert _post(client, auth_headers, make_wav(), model="sensevoice").status_code == 200
    assert asr_spy.transcribe[0]["hotwords"] == []
    assert asr_spy.transcribe[0]["model_id"] == "iic/SenseVoiceSmall"


def test_transcriptions_rejects_missing_model(client: TestClient, auth_headers, make_wav):
    response = _post(client, auth_headers, make_wav())
    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


def test_transcriptions_rejects_unknown_model(
    client: TestClient, auth_headers, asr_spy, make_wav
) -> None:
    """未登记模型不做静默回退：直接 400，避免把任意字符串透传给引擎"""
    response = _post(client, auth_headers, make_wav(), model="not-a-model")
    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"
    assert asr_spy.transcribe == []


def test_transcriptions_rejects_empty_audio(client: TestClient, auth_headers, asr_spy):
    response = _post(client, auth_headers, b"", model="sensevoice")
    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


@pytest.mark.parametrize("hotwords", ["not-json", json.dumps({"a": 1}), json.dumps([1, 2])])
def test_transcriptions_rejects_invalid_hotwords(
    client: TestClient, auth_headers, asr_spy, make_wav, hotwords
):
    response = _post(client, auth_headers, make_wav(), model="sensevoice", hotwords=hotwords)
    assert response.status_code == 400
    assert response.json()["code"] == "PARAM_ERROR"


def test_transcriptions_rejects_unsupported_wav_format(
    client: TestClient, auth_headers, make_wav, monkeypatch
):
    """8kHz WAV 超出引擎输入规格：422（编排侧不重试）"""
    monkeypatch.setattr(asr_engine, "_load_model", lambda model_id: _fake_model([]))
    response = _post(client, auth_headers, make_wav(sample_rate=8000), model="sensevoice")
    assert response.status_code == 422
    assert response.json()["code"] == "AUDIO_INVALID"


def test_decode_audio_reads_16k_wav(make_wav) -> None:
    samples = asr_engine._decode_audio(make_wav(seconds=0.1))
    assert samples.shape == (1600,)
    assert samples.dtype.name == "float32"


def test_decode_audio_rejects_stereo_wav(make_wav) -> None:
    with pytest.raises(AudioInvalidError):
        asr_engine._decode_audio(make_wav(channels=2))


def test_transcribe_applies_per_request_hotwords_and_model_family_params(
    make_wav, monkeypatch
) -> None:
    """热词与模型族参数按请求生效：SeACo 传热词、SenseVoice 传语言与逆文本规范化"""
    seen: list[dict] = []
    monkeypatch.setattr(asr_engine, "_load_model", lambda model_id: _fake_model(seen))
    audio = make_wav(seconds=0.1)

    text = asr_engine.transcribe(audio, "iic/speech_seaco_paraformer_large", ["乙", "甲", "甲"])
    assert text == "Hello 世界"
    assert seen[0]["hotword"] == "乙 甲"  # 去重后按码点排序，保证同一热词集合的推理输入稳定

    asr_engine.transcribe(audio, "iic/SenseVoiceSmall", [])
    assert "hotword" not in seen[1]
    assert seen[1]["language"] == "zh"
    assert seen[1]["use_itn"] is True


def test_engine_keeps_no_global_hotword_state() -> None:
    """热词随请求传入，引擎进程内不得保存全局热词状态"""
    assert not [name for name in vars(asr_engine) if "hotword" in name.lower()]


def _fake_model(seen: list[dict]):
    class _FakeAutoModel:
        def generate(self, **kwargs):
            seen.append(kwargs)
            return [{"text": " Hello <|zh|><|NEUTRAL|>世界 "}]

    return _FakeAutoModel()
