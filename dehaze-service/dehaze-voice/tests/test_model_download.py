"""TTS 权重解析、下载（断点续传 + 字节校验）与文件锁路径。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import settings
from app.engines import tts as tts_engine
from app.engines.tts import VoiceParams
from app.errors import ModelUnavailableError, ParamError


@pytest.fixture
def model_dir(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(settings, "TTS_MODEL_DIR", str(tmp_path))
    return tmp_path


class _FakeStream:
    """httpx.stream 的桩响应"""

    def __init__(self, payload: bytes, declared_size: int | None = None) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers = {
            "content-length": str(len(payload) if declared_size is None else declared_size)
        }

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, chunk_size: int):
        yield self._payload

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_weights_already_present_skip_download(model_dir: Path, monkeypatch) -> None:
    onnx = model_dir / "huayan.onnx"
    onnx.write_bytes(b"weights")
    Path(f"{onnx}.json").write_text("{}")
    monkeypatch.setattr(
        tts_engine, "_download", lambda *args: pytest.fail("权重已就绪不应触发下载")
    )
    assert tts_engine._ensure_weights(VoiceParams(onnx="huayan.onnx", size=7)) == onnx


def test_missing_weights_without_urls(model_dir: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="未提供下载地址"):
        tts_engine._ensure_weights(VoiceParams(onnx="huayan.onnx"))


def test_weights_downloaded_via_url_templates(model_dir: Path, monkeypatch) -> None:
    requested: list[str] = []

    def fake_download(url: str, path: Path, size: int | None) -> None:
        requested.append(url)
        path.write_bytes(b"w" * (size or 2))

    monkeypatch.setattr(tts_engine, "_download", fake_download)
    onnx = tts_engine._ensure_weights(
        VoiceParams(onnx="huayan.onnx", size=9, urls=["https://mirror/{filename}"])
    )
    assert requested == ["https://mirror/huayan.onnx", "https://mirror/huayan.onnx.json"]
    assert onnx.read_bytes() == b"w" * 9


def test_weights_from_absolute_path(model_dir: Path, tmp_path: Path) -> None:
    external = tmp_path / "external.onnx"
    external.write_bytes(b"weights")
    Path(f"{external}.json").write_text("{}")
    assert tts_engine._ensure_weights(VoiceParams(onnx=str(external))) == external


def test_url_onnx_lands_in_model_dir(model_dir: Path) -> None:
    assert (
        tts_engine.resolve_onnx_path(VoiceParams(onnx="https://mirror/voices/huayan.onnx"))
        == model_dir / "huayan.onnx"
    )


def test_url_onnx_without_filename_rejected() -> None:
    with pytest.raises(ParamError):
        tts_engine.resolve_onnx_path(VoiceParams(onnx="https://mirror/voices/"))


def test_download_writes_file_and_validates_size(model_dir: Path, monkeypatch) -> None:
    payload = b"onnx-bytes"
    monkeypatch.setattr(tts_engine.httpx, "stream", lambda *args, **kwargs: _FakeStream(payload))
    target = model_dir / "x.onnx"
    tts_engine._download("https://mirror/x.onnx", target, len(payload))
    assert target.read_bytes() == payload


def test_download_rejects_incomplete_file(model_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(tts_engine.httpx, "stream", lambda *args, **kwargs: _FakeStream(b"12345"))
    with pytest.raises(ModelUnavailableError, match="与预期"):
        tts_engine._download("https://mirror/x.onnx", model_dir / "x.onnx", 99)


def test_download_rejects_truncated_body(model_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        tts_engine.httpx,
        "stream",
        lambda *args, **kwargs: _FakeStream(b"12345", declared_size=99),
    )
    with pytest.raises(ModelUnavailableError, match="下载不完整"):
        tts_engine._download("https://mirror/x.onnx", model_dir / "x.onnx", 99)
