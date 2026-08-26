"""FunASR 本地引擎（进程内部署，CPU 推理）

语音识别引擎直接运行在 dehaze-python 进程内（对齐后端实现 §3.1 进程内部署形态）：
- 模型懒加载：首个 ASR 请求时才经 funasr AutoModel 加载，未使用语音功能时不占内存、不拖慢启动
- 模型进程内单例：流式（SenseVoice-Small）/ 离线（SeACo-Paraformer-Large，热词增强版）各一份；
  多 Worker 部署时每 Worker 各持一份（内存成本见后端实现 §9.1）
- 推理在专用线程池执行（CPU 密集，不阻塞事件循环）
- 热词为进程内内存表，注册后对流式/离线推理即时生效（SeACo-Paraformer 原生热词加权）

funasr/modelscope 为主依赖（延迟导入控制启动成本），此处延迟导入
仅作为环境异常的兜底报错，不影响应用其余功能启动。
"""

import asyncio
import logging
import re
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ASR 推理线程池（CPU 密集，与 GPU 图像推理线程池隔离）
_executor = ThreadPoolExecutor(
    max_workers=settings.VOICE_ASR_INFERENCE_THREADS, thread_name_prefix="funasr-infer"
)

# 逻辑模型名 → funasr 模型 ID（ModelScope）
_MODEL_IDS = {
    # 流式：SenseVoice-Small，分段增量识别（伪流式，延迟约 1 秒档）
    "sensevoice": "iic/SenseVoiceSmall",
    # 离线：SeACo-Paraformer-Large（Paraformer-Large 热词增强版，原生支持热词加权）
    "paraformer": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
}

# SenseVoice 输出中的语言/情感/事件标签（如 <|zh|><|NEUTRAL|>），转写前剥离
_SENSEVOICE_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")

# 已加载模型缓存 {模型ID: AutoModel}，进程内单例
_models: dict[str, Any] = {}
_models_lock = threading.Lock()

# 进程内热词表
_hotwords: set[str] = set()
_hotwords_lock = threading.Lock()


class FunASREngineError(Exception):
    """本地引擎调用失败（依赖缺失/模型加载失败/音频格式不合规格）"""


def resolve_model_id(model: str | None, default_logical: str) -> str:
    """解析逻辑模型名为 funasr 模型 ID（白名单校验，未知模型直接报错）"""
    name = model or default_logical
    if name not in _MODEL_IDS:
        raise FunASREngineError(f"不支持的 ASR 模型: {name}（可选: {'/'.join(_MODEL_IDS)}）")
    return _MODEL_IDS[name]


def ensure_model(model: str | None, default_logical: str) -> None:
    """预加载模型（会话建立前调用，依赖缺失/加载失败抛 FunASREngineError）"""
    _load_model(resolve_model_id(model, default_logical))


def _device() -> str:
    """FunASR 推理设备：torch CUDA 可用时用 GPU，否则 CPU"""
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 检测失败按 CPU 处理
        return "cpu"


def _load_model(model_id: str) -> Any:
    """加载（或复用）funasr AutoModel，线程安全懒加载"""
    with _models_lock:
        if model_id in _models:
            return _models[model_id]
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise FunASREngineError(
                "funasr 未安装，语音识别不可用，请执行 uv sync 修复依赖"
            ) from e
        device = _device()
        logger.info("加载 FunASR 模型: %s（%s）", model_id, device)
        try:
            _models[model_id] = AutoModel(model=model_id, disable_update=True, device=device)
        except Exception as e:
            raise FunASREngineError(f"加载 FunASR 模型失败: {model_id} error={e}") from e
        return _models[model_id]


def _decode_audio(audio: bytes) -> list[float]:
    """音频字节解码为 16kHz 单声道归一化采样序列

    WAV（RIFF 头）校验声道/位深/采样率后取帧；裸 PCM 按 16kHz/16bit/mono 解释。
    规格不符（非单声道/非 16bit/非 16kHz）直接拒绝。
    """
    if audio[:4] == b"RIFF":
        try:
            with wave.open(BytesIO(audio)) as wav:
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or wav.getframerate() != 16000
                ):
                    raise FunASREngineError("仅支持 16kHz/16bit/单声道 WAV/PCM 音频")
                frames = wav.readframes(wav.getnframes())
        except wave.Error as e:
            raise FunASREngineError(f"WAV 音频解析失败: {e}") from e
    else:
        frames = audio
    import numpy as np

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _postprocess_text(text: str) -> str:
    """转写文本后处理：剥离 SenseVoice 标签，规整空白"""
    return _SENSEVOICE_TAG_PATTERN.sub("", text).strip()


def register_hotwords(words: list[str]) -> None:
    """注册热词（进程内内存表，替换式更新），对流式/离线推理即时生效"""
    with _hotwords_lock:
        _hotwords.clear()
        _hotwords.update(words)
    logger.info("FunASR 热词已注册: %s 个", len(words))


def _hotword_param() -> str:
    """当前热词表的 generate 参数（空表返回空串表示不启用）"""
    with _hotwords_lock:
        return " ".join(sorted(_hotwords))


def engine_status() -> dict[str, Any]:
    """查询引擎状态（不抛异常）。

    返回：engine_status("online"/"offline")、stream_model_loaded、
    offline_model_loaded，基于模型是否已加载到进程内单例判定。
    """
    if not _models:
        return {
            "engine_status": "offline",
            "stream_model_loaded": False,
            "offline_model_loaded": False,
        }
    return {
        "engine_status": "online",
        "stream_model_loaded": resolve_model_id(None, settings.VOICE_ASR_STREAM_MODEL)
        in _models,
        "offline_model_loaded": resolve_model_id(None, settings.VOICE_ASR_OFFLINE_MODEL)
        in _models,
    }


def transcribe(audio: bytes, model: str | None, default_logical: str) -> str:
    """完整音频转写（阻塞调用，须经线程池执行）

    model 为逻辑模型名（sensevoice/paraformer）或 funasr 模型 ID。
    """
    model_id = resolve_model_id(model, default_logical)
    funasr_model = _load_model(model_id)
    samples = _decode_audio(audio)
    if len(samples) == 0:
        return ""
    kwargs: dict[str, Any] = {"input": samples}
    if model_id.startswith("iic/SenseVoice"):
        # 中文定向 + 逆文本规范化（数字规范化："八十"→"80"，需求 §3.5）
        kwargs.update(language="zh", use_itn=True)
    if model_id.startswith("iic/speech_seaco_paraformer"):
        hotword = _hotword_param()
        if hotword:
            kwargs["hotword"] = hotword
    try:
        result = funasr_model.generate(**kwargs)
    except Exception as e:
        raise FunASREngineError(f"FunASR 推理失败: {model_id} error={e}") from e
    text = result[0].get("text", "") if result else ""
    return _postprocess_text(str(text))


def run_in_executor(func, *args):
    """将阻塞的引擎调用提交到推理线程池，返回 awaitable"""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_executor, func, *args)
