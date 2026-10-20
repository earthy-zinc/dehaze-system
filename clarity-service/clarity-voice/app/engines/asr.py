"""FunASR 语音识别引擎：懒加载单例 + 专用推理线程池。

引擎只做"引擎参数进、识别结果出"：
- 模型由请求携带的 ModelScope model_id 决定（编排侧按统一模型注册表 sys_ai_model
  解析后下发），引擎不查库
- 热词按请求生效，进程内不保存任何跨请求的热词状态（消除多用户并发互相覆盖）
- 权重经 ModelScope 下载并缓存于 ${MODEL_CACHE_DIR}/modelscope，重启不重复下载
- 推理为 CPU/GPU 密集的阻塞调用，统一提交到本引擎的线程池，不阻塞事件循环
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import numpy as np

from app.config import settings
from app.errors import AudioInvalidError, InferenceError, ModelUnavailableError

logger = logging.getLogger(__name__)

# 引擎输入规格：16kHz/16bit/单声道（与编排侧前端上行契约一致，服务侧不做转码）
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2

# SenseVoice 输出中的语言/情感/事件标签（如 <|zh|><|NEUTRAL|>），转写前剥离
_SENSEVOICE_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")

_executor = ThreadPoolExecutor(
    max_workers=settings.ASR_INFERENCE_THREADS, thread_name_prefix="funasr-infer"
)

# 已加载模型单例 {model_id: AutoModel}：多实例部署时每实例一份（替代原先每 Worker 一份）
_models: dict[str, Any] = {}
_models_lock = threading.Lock()


def dependency_error() -> str | None:
    """引擎依赖是否可加载（/ready 只读判定，不触发模型加载或下载）"""
    try:
        import funasr  # noqa: F401
    except ImportError as e:
        return f"funasr 不可用: {e}"
    return None


def device() -> str:
    """推理设备：ASR_DEVICE=auto 时按 torch CUDA 可用性选择"""
    if settings.ASR_DEVICE != "auto":
        return settings.ASR_DEVICE
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except (ImportError, RuntimeError) as e:
        # 探测失败回退 CPU 是设计内降级，但必须可见，否则 GPU 环境故障会被静默吞掉
        logger.warning("CUDA 探测失败，ASR 回退 CPU 推理: %s", e)
        return "cpu"


def _load_model(model_id: str) -> Any:
    """按 ModelScope 模型 ID 加载（或复用）FunASR 模型，线程安全懒加载"""
    with _models_lock:
        if model_id in _models:
            return _models[model_id]
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ModelUnavailableError("funasr 未安装，语音识别不可用") from e
        dev = device()
        logger.info("加载 FunASR 模型 model_id=%s device=%s", model_id, dev)
        try:
            model = AutoModel(model=model_id, disable_update=True, device=dev)
        except Exception as e:
            raise ModelUnavailableError(f"加载 FunASR 模型失败: {model_id} error={e}") from e
        _models[model_id] = model
        return model


def _decode_audio(audio: bytes) -> np.ndarray:
    """音频字节解码为 16kHz 单声道归一化采样序列。

    WAV（RIFF 头）校验声道/位深/采样率后取帧，规格不符即拒绝；裸 PCM 按同规格解释。
    """
    if audio[:4] == b"RIFF":
        try:
            with wave.open(BytesIO(audio)) as wav:
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getframerate() != SAMPLE_RATE
                ):
                    raise AudioInvalidError("仅支持 16kHz/16bit/单声道 WAV/PCM 音频")
                frames = wav.readframes(wav.getnframes())
        except wave.Error as e:
            raise AudioInvalidError(f"WAV 音频解析失败: {e}") from e
    else:
        frames = audio
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio: bytes, model_id: str, hotwords: list[str] | None = None) -> str:
    """音频（完整音频或流式分段）转写为文本，阻塞调用，须经 run_transcribe 提交线程池。

    模型族差异化参数：SenseVoice 定向中文并启用逆文本规范化（"八十"→"80"）；
    热词加权仅 SeACo-Paraformer 系支持，空热词不传参。
    """
    model = _load_model(model_id)
    samples = _decode_audio(audio)
    if samples.size == 0:
        return ""
    kwargs: dict[str, Any] = {"input": samples}
    if model_id.startswith("iic/SenseVoice"):
        kwargs.update(language="zh", use_itn=True)
    if hotwords and model_id.startswith("iic/speech_seaco_paraformer"):
        kwargs["hotword"] = " ".join(sorted(set(hotwords)))
    try:
        result = model.generate(**kwargs)
    except Exception as e:
        raise InferenceError(f"FunASR 推理失败: {model_id} error={e}") from e
    text = result[0].get("text", "") if result else ""
    return _SENSEVOICE_TAG_PATTERN.sub("", str(text)).strip()


async def run_transcribe(audio: bytes, model_id: str, hotwords: list[str] | None = None) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, transcribe, audio, model_id, hotwords)


async def preload(model_id: str) -> None:
    """会话建立前预加载模型：模型不可用时立即失败，避免收完音频才发现"""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _load_model, model_id)


class StreamSession:
    """流式识别会话（伪流式分段增量）：累计满 1 秒音频触发一次分段转写，文本逐段累加。

    分段结果按"当前累计文本"覆盖式下行，与前端既有展示契约一致；会话状态只存在于
    本会话对象内，不同会话（不同用户）互不串扰。
    """

    def __init__(self, model_id: str, hotwords: list[str]) -> None:
        self._model_id = model_id
        self._hotwords = hotwords
        self._buffer = bytearray()
        self._transcribed = 0
        self._text = ""

    @property
    def audio_bytes(self) -> int:
        return len(self._buffer)

    @property
    def text(self) -> str:
        return self._text

    def feed(self, chunk: bytes) -> bool:
        """追加音频块；返回是否已积满一个分段（True 时调用方应触发一次增量转写）"""
        self._buffer.extend(chunk)
        return len(self._buffer) - self._transcribed >= BYTES_PER_SECOND

    async def transcribe_pending(self) -> str:
        """转写尚未识别的音频段并返回当前累计文本（无新增音频时直接返回）"""
        segment = bytes(self._buffer[self._transcribed :])
        self._transcribed = len(self._buffer)
        if segment:
            text = await run_transcribe(segment, self._model_id, self._hotwords)
            self._text += text
        return self._text


def loaded_models() -> list[str]:
    return list(_models)


def loaded_model_id() -> str | None:
    """最近加载的模型（dict 保序，取最后一个）"""
    return next(reversed(_models), None)


def cached_models() -> list[str]:
    """已落盘的模型（ModelScope 缓存目录 hub/{命名空间}/{模型名}）"""
    hub = settings.MODELSCOPE_CACHE_DIR / "hub"
    if not hub.is_dir():
        return []
    return sorted(
        f"{ns.name}/{model.name}" for ns in hub.iterdir() if ns.is_dir() for model in ns.iterdir()
    )


def capability() -> dict[str, Any]:
    """能力发现条目。

    引擎不自持模型目录：`cached` 为本机已缓存权重，`downloadable` 表示任意 ModelScope
    model_id 均可按需下载；权威模型目录在编排侧统一模型注册表（sys_ai_model）。
    """
    error = dependency_error()
    entry: dict[str, Any] = {
        "id": "asr",
        "type": "asr",
        "available": error is None,
        "device": device(),
        "loaded": loaded_models(),
        "cached": cached_models(),
        "downloadable": True,
        "capabilities": {
            "streaming": True,
            "hotwords": True,
            "formats": ["wav", "pcm"],
            "sample_rate": SAMPLE_RATE,
        },
        "limits": {"stream_idle_timeout_seconds": settings.ASR_STREAM_IDLE_TIMEOUT},
    }
    if error:
        entry["unavailable_reason"] = error
    return entry
