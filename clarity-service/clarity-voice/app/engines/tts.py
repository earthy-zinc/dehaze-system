"""Piper 语音合成引擎：懒加载单例 + 权重按需下载 + 输出重采样/编码。

引擎只做"引擎参数进、音频字节出"：
- 音色权重来源（onnx 路径或下载地址、字节数）由编排侧按统一模型注册表（sys_ai_model）
  解析后随请求下发（voice_params），引擎不查库、不做业务缓存
- 权重缓存于 TTS_MODEL_DIR（默认 ${MODEL_CACHE_DIR}/piper），跨进程互斥下载 + 断点续传
- 合成阻塞 CPU/GPU，统一提交本引擎线程池；输出按请求的 format/sample_rate 编码
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from app.config import settings
from app.errors import InferenceError, ModelUnavailableError, ParamError

try:
    import fcntl
except ImportError:  # Windows 无 fcntl：退化为无跨进程锁（单机自愈）
    fcntl = None

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("mp3", "wav", "pcm")
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

_DOWNLOAD_CHUNK = 1024 * 1024

_executor = ThreadPoolExecutor(
    max_workers=settings.TTS_INFERENCE_THREADS, thread_name_prefix="piper-tts"
)

# 已加载音色模型单例 {onnx 绝对路径: PiperVoice}
_models: dict[str, Any] = {}
_models_lock = threading.Lock()
_last_voice: str | None = None


class VoiceParams(BaseModel):
    """音色权重来源（编排侧按统一模型注册表 sys_ai_model 解析后下发）"""

    onnx: str
    size: int | None = Field(default=None, gt=0)
    urls: list[str] = []


def model_dir() -> Path:
    configured = settings.TTS_MODEL_DIR.strip()
    return Path(configured) if configured else Path(settings.MODEL_CACHE_DIR) / "piper"


def dependency_error() -> str | None:
    """引擎依赖是否可加载（/ready 只读判定，不触发权重下载或加载）"""
    try:
        import piper  # noqa: F401
    except ImportError as e:
        return f"piper-tts 不可用: {e}"
    return None


def device() -> str:
    """合成设备：onnxruntime 带 CUDAExecutionProvider 时用 GPU（需装 onnxruntime-gpu）"""
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
        return "cpu"
    except ImportError:
        logger.warning("onnxruntime 不可用，TTS 回退 CPU 合成", exc_info=True)
        return "cpu"


def resolve_onnx_path(params: VoiceParams) -> Path:
    """onnx 为 URL 时落 TTS_MODEL_DIR/<文件名>；为本地路径时相对 TTS_MODEL_DIR 解析"""
    if params.onnx.startswith(("http://", "https://")):
        url_path = urlsplit(params.onnx).path
        if not url_path or url_path.endswith("/"):
            raise ParamError(f"voice_params.onnx 地址缺少文件名: {params.onnx}")
        return model_dir() / Path(url_path).name
    path = Path(params.onnx)
    return path if path.is_absolute() else model_dir() / path


def _ready(path: Path, size: int | None) -> bool:
    return path.exists() and (size is None or path.stat().st_size == size)


def _download(url: str, path: Path, size: int | None) -> None:
    """断点续传下载并校验字节数。

    续传临时文件按进程隔离：共享同一 .part 时无锁平台上的并发追加写会互相穿插，
    产出大于远端大小的残包，其后续传偏移越界（HTTP 416）。
    """
    part = Path(f"{path}.{os.getpid()}.part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.stream(
        "GET",
        url,
        headers=headers,
        timeout=httpx.Timeout(connect=15, read=120, write=30, pool=15),
        follow_redirects=True,
    ) as response:
        if offset and response.status_code != 206:
            offset = 0  # 服务端不支持续传（返回 200 全量）→ 重头下载
        response.raise_for_status()
        total = offset + int(response.headers.get("content-length", 0))
        if size and total != size:
            raise ModelUnavailableError(f"远端文件大小 {total} 与预期 {size} 不一致")
        with part.open("ab" if offset else "wb") as file:
            for chunk in response.iter_bytes(_DOWNLOAD_CHUNK):
                file.write(chunk)
    written = part.stat().st_size
    if size and written != size:
        raise ModelUnavailableError(f"权重下载不完整：{written} != {size}")
    if _ready(path, size):
        # 并发下载同一权重时他方已落地：丢弃本次结果，避免覆盖可能已被推理占用的文件
        part.unlink(missing_ok=True)
        return
    part.replace(path)


def _download_with_fallback(filename: str, path: Path, size: int | None, urls: list[str]) -> None:
    if not urls:
        raise ModelUnavailableError(
            f"TTS 权重 {filename} 缺失且未提供下载地址（voice_params.urls）: {path}"
        )
    last_error: Exception | None = None
    for template in urls:
        try:
            _download(template.format(filename=filename), path, size)
            logger.info("TTS 权重就绪: %s", path)
            return
        except Exception as e:
            last_error = e
            logger.warning("TTS 权重下载失败（%s）: %s，尝试下一个镜像", template, e)
    raise ModelUnavailableError(f"TTS 权重下载失败（{filename}）: {last_error}") from last_error


@contextmanager
def _weights_lock(onnx_path: Path):
    """权重下载的跨进程互斥锁：Windows 无 fcntl 时退化为无锁（续传临时文件本就按进程隔离）"""
    if fcntl is None:
        yield
        return
    with Path(f"{onnx_path}.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _ensure_weights(params: VoiceParams) -> Path:
    """确保 onnx 与其配套 .json 就绪（Piper 需要 onnx + onnx.json 两个文件）"""
    onnx_path = resolve_onnx_path(params)
    config_path = Path(f"{onnx_path}.json")
    if _ready(onnx_path, params.size) and config_path.exists():
        return onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with _weights_lock(onnx_path):
        if not _ready(onnx_path, params.size):
            _download_with_fallback(onnx_path.name, onnx_path, params.size, params.urls)
        if not config_path.exists():
            _download_with_fallback(config_path.name, config_path, None, params.urls)
    return onnx_path


def _load_model(voice_id: str, params: VoiceParams) -> Any:
    """加载（或复用）PiperVoice，线程安全懒加载；权重就绪在锁外完成，避免持锁下载"""
    global _last_voice
    onnx_path = str(_ensure_weights(params))
    with _models_lock:
        if onnx_path in _models:
            return _models[onnx_path]
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise ModelUnavailableError("piper-tts 未安装，语音合成不可用") from e
        use_cuda = device() == "cuda"
        logger.info(
            "加载 TTS 权重 onnx=%s voice_id=%s device=%s",
            onnx_path,
            voice_id,
            "cuda" if use_cuda else "cpu",
        )
        try:
            _models[onnx_path] = PiperVoice.load(onnx_path, use_cuda=use_cuda)
        except Exception as e:
            raise ModelUnavailableError(f"加载 TTS 权重失败: {onnx_path} error={e}") from e
        _last_voice = voice_id
        return _models[onnx_path]


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """16bit 单声道 PCM 重采样（sinc 插值，避免线性插值的混叠失真）"""
    import torch
    from torchaudio.functional import resample

    samples = torch.frombuffer(bytearray(pcm), dtype=torch.int16).float().div_(32768.0)
    resampled = resample(samples, orig_freq=src_rate, new_freq=dst_rate)
    return (resampled * 32768.0).clamp(-32768, 32767).to(torch.int16).numpy().tobytes()


def _encode_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _encode_mp3(pcm: bytes, sample_rate: int) -> bytes:
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())


def synthesize(
    text: str,
    voice_id: str,
    params: VoiceParams,
    speed: float,
    format_: str,
    sample_rate: int,
) -> bytes:
    """合成并编码（阻塞调用，须经 run_synthesize 提交线程池）"""
    model = _load_model(voice_id, params)
    try:
        from piper import SynthesisConfig

        pcm = bytearray()
        for chunk in model.synthesize(text, SynthesisConfig(length_scale=1.0 / speed)):
            pcm.extend(chunk.audio_int16_bytes)
    except Exception as e:
        raise InferenceError(f"语音合成推理失败: {e}") from e
    if not pcm:
        raise InferenceError("合成结果为空音频（文本无可发音内容）")

    data = bytes(pcm)
    native_rate = model.config.sample_rate
    if sample_rate != native_rate:
        data = _resample(data, native_rate, sample_rate)
    if format_ == "pcm":
        return data
    if format_ == "wav":
        return _encode_wav(data, sample_rate)
    return _encode_mp3(data, sample_rate)


async def run_synthesize(
    text: str,
    voice_id: str,
    params: VoiceParams,
    speed: float,
    format_: str,
    sample_rate: int,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, synthesize, text, voice_id, params, speed, format_, sample_rate
    )


def loaded_voice_id() -> str | None:
    return _last_voice if _models else None


def cached_voices() -> list[str]:
    """已落盘的音色（TTS_MODEL_DIR 下的 onnx 文件名）。

    权威音色目录在编排侧统一模型注册表（sys_ai_model）。
    """
    directory = model_dir()
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.onnx"))


def capability() -> dict[str, Any]:
    """能力发现条目"""
    error = dependency_error()
    voice_id = loaded_voice_id()
    entry: dict[str, Any] = {
        "id": "tts",
        "type": "tts",
        "available": error is None,
        "device": device(),
        "loaded": [voice_id] if voice_id else [],
        "cached": cached_voices(),
        "downloadable": True,
        "capabilities": {
            "voices": cached_voices(),
            "formats": list(SUPPORTED_FORMATS),
            "sample_rates": list(SUPPORTED_SAMPLE_RATES),
        },
        "limits": {"max_text_chars": settings.TTS_MAX_TEXT_CHARS, "speed_range": [0.5, 2.0]},
    }
    if error:
        entry["unavailable_reason"] = error
    return entry
