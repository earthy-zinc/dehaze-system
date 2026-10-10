"""测试夹具：引擎全部桩化，断言只落在对外契约与引擎实收参数上。"""

from __future__ import annotations

import sys
import types
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.engines import asr as asr_engine
from app.engines import tts as tts_engine

# 测试不落文件日志（文件日志处理器由 test_logging 用 tmp_path 单独验证）
settings.LOG_ENABLE_FILE = False


@pytest.fixture(scope="session")
def client() -> TestClient:
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.INTERNAL_SERVICE_KEY}"}


@pytest.fixture
def make_wav():
    """构造 16bit WAV 测试音频（默认 16kHz/单声道/1 秒）"""

    def _make(sample_rate: int = 16000, channels: int = 1, seconds: float = 1.0) -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00\x10" * int(sample_rate * seconds) * channels)
        return buffer.getvalue()

    return _make


@pytest.fixture
def asr_spy(monkeypatch) -> SimpleNamespace:
    """桩 ASR 引擎：记录每次转写收到的音频/模型/热词，避免真实模型加载"""
    spy = SimpleNamespace(transcribe=[], preload=[])

    def fake_transcribe(audio: bytes, model_id: str, hotwords: list[str] | None = None) -> str:
        spy.transcribe.append(
            {"audio": audio, "model_id": model_id, "hotwords": list(hotwords or [])}
        )
        return f"识别{len(spy.transcribe)}"

    async def fake_preload(model_id: str) -> None:
        spy.preload.append(model_id)

    monkeypatch.setattr(asr_engine, "transcribe", fake_transcribe)
    monkeypatch.setattr(asr_engine, "preload", fake_preload)
    return spy


@pytest.fixture
def tts_spy(monkeypatch) -> SimpleNamespace:
    """桩 TTS 引擎：假 piper 模块 + 假 PiperVoice，避免真实权重下载与加载"""
    fake_piper = types.ModuleType("piper")
    fake_piper.SynthesisConfig = (  # type: ignore[attr-defined]  # 桩模块动态挂属性，静态检查无法感知
        lambda length_scale=1.0: SimpleNamespace(length_scale=length_scale)
    )
    monkeypatch.setitem(sys.modules, "piper", fake_piper)

    spy = SimpleNamespace(loads=[], configs=[])

    class _FakeModel:
        # 桩音色原生采样率：与请求采样率不同以覆盖重采样分支
        native_rate = 22050

        def __init__(self) -> None:
            self.config = SimpleNamespace(sample_rate=self.native_rate)

        def synthesize(self, text: str, config) -> list:
            spy.configs.append(config)
            return [SimpleNamespace(audio_int16_bytes=b"\x00\x10" * 400)] * 2

    def fake_load_model(voice_id: str, params) -> _FakeModel:
        spy.loads.append({"voice_id": voice_id, "onnx": params.onnx})
        return _FakeModel()

    monkeypatch.setattr(tts_engine, "_load_model", fake_load_model)
    return spy
