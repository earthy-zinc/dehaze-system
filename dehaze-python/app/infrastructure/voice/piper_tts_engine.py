"""本地 TTS 引擎（Piper，进程内部署，CPU 推理）

对齐 FunASR 引擎的部署形态（后端实现 §3.3）：
- 模型懒加载：首次合成请求时才加载 ONNX 模型，未使用语音功能时不占内存
- 模型自动下载：文件缺失时经 hf-mirror 下载（断点续传 + fcntl 跨进程锁，
  对齐 local_llm_model 的零手工部署策略；Windows 无 fcntl 退化为无锁直下），
  回退 huggingface.co
- 推理在专用线程池执行（CPU 密集，不阻塞事件循环；espeak 音素化由 Piper
  内部全局锁保护，onnxruntime 会话线程安全）
- 输出编码：PCM 重采样（torchaudio sinc 插值）→ WAV 封装 / MP3 编码（lameenc）/ 裸 PCM
"""

import asyncio
import io
import logging
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

try:
    import fcntl
except ImportError:  # Windows 无 fcntl
    fcntl = None

from app.config import settings

logger = logging.getLogger(__name__)

# TTS 推理线程池（CPU 密集，与 ASR / GPU 图像推理线程池隔离）
_executor = ThreadPoolExecutor(
    max_workers=settings.VOICE_TTS_INFERENCE_THREADS, thread_name_prefix="piper-tts"
)

# 音色 → piper-voices 模型文件名（zh_CN-huayan-medium，CC-BY-4.0）
_VOICE_MODEL_FILES = {"huayan": "zh_CN-huayan-medium.onnx"}
# 模型文件字节数（远端实际大小，下载完整性校验用）
_VOICE_MODEL_SIZES = {"huayan": 63201294}

# 下载镜像（hf-mirror 优先，回退 huggingface.co）
_DOWNLOAD_URLS = [
    "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{filename}",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{filename}",
]

_CHUNK = 1024 * 1024

# 已加载模型缓存 {音色: PiperVoice}，进程内单例
_models: dict[str, Any] = {}
_models_lock = threading.Lock()


class LocalTtsError(Exception):
    """本地 TTS 引擎调用失败（依赖缺失/模型加载或下载失败/合成失败/音色不支持）"""


def _model_path(voice: str) -> str:
    """音色模型 onnx 路径（默认仓库根 models/piper/，可经 VOICE_TTS_MODEL_PATH 覆盖默认音色）"""
    configured = settings.VOICE_TTS_MODEL_PATH.strip()
    if configured:
        return configured
    return os.path.join(settings.MODEL_CACHE_DIR, "piper", _VOICE_MODEL_FILES[voice])


def _file_ready(path: str, expected_size: int | None) -> bool:
    if not os.path.exists(path):
        return False
    return expected_size is None or os.path.getsize(path) == expected_size


def _download_file(url: str, path: str, expected_size: int | None) -> None:
    """下载单个文件（断点续传），expected_size 非空时校验最终字节数"""
    part = path + ".part"
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.Client(
        timeout=httpx.Timeout(connect=15, read=120, write=30, pool=15), follow_redirects=True
    ) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if offset and resp.status_code != 206:
                offset = 0  # 服务端不支持续传（返回 200 全量）→ 重头下载
            resp.raise_for_status()
            total = offset + int(resp.headers.get("content-length", 0))
            if expected_size and total and total != expected_size:
                raise RuntimeError(f"远端文件大小 {total} 与预期 {expected_size} 不一致")
            mode = "ab" if offset else "wb"
            downloaded = offset
            with open(part, mode) as f:
                for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                    f.write(chunk)
                    downloaded += len(chunk)
            if expected_size and downloaded != expected_size:
                raise RuntimeError(f"下载数不完整：{downloaded} != {expected_size}")
    os.replace(part, path)


def _ensure_downloaded(voice: str) -> str:
    """确保音色模型文件（onnx + json 配置）就绪，跨进程互斥下载"""
    onnx_path = _model_path(voice)
    json_path = onnx_path + ".json"
    if _file_ready(onnx_path, _VOICE_MODEL_SIZES.get(voice)) and _file_ready(json_path, None):
        return onnx_path
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    lock_file = open(onnx_path + ".lock", "w") if fcntl else None  # Windows：无锁直下
    try:
        if lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not _file_ready(onnx_path, _VOICE_MODEL_SIZES.get(voice)):
            _download_with_fallback(_VOICE_MODEL_FILES[voice], onnx_path,
                                    _VOICE_MODEL_SIZES.get(voice))
        if not _file_ready(json_path, None):
            _download_with_fallback(_VOICE_MODEL_FILES[voice] + ".json", json_path, None)
        return onnx_path
    finally:
        if lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


def _download_with_fallback(filename: str, path: str, expected_size: int | None) -> None:
    last_error: Exception | None = None
    for url_tpl in _DOWNLOAD_URLS:
        try:
            _download_file(url_tpl.format(filename=filename), path, expected_size)
            logger.info("TTS 模型文件就绪：%s（%dMB）", path, os.path.getsize(path) // 1048576)
            return
        except Exception as exc:  # noqa: BLE001 逐镜像尝试，全部失败才抛出
            last_error = exc
            logger.warning("TTS 模型下载失败（%s）：%s，尝试下一个镜像", url_tpl, exc)
    raise LocalTtsError(
        f"TTS 模型自动下载失败（{filename}）：{last_error}。"
        "请检查网络可达 hf-mirror.com / huggingface.co，或手动下载放置于 models/piper/ 目录"
    ) from last_error


def _load_model(voice: str) -> Any:
    """加载（或复用）PiperVoice，线程安全懒加载"""
    with _models_lock:
        if voice in _models:
            return _models[voice]
        try:
            from piper import PiperVoice
        except ImportError as e:
            raise LocalTtsError(
                "piper-tts 未安装，语音合成不可用，请执行 uv sync 修复依赖"
            ) from e
        onnx_path = _ensure_downloaded(voice)
        logger.info("加载本地 TTS 模型: %s", onnx_path)
        try:
            _models[voice] = PiperVoice.load(onnx_path)
        except Exception as e:
            raise LocalTtsError(f"加载 TTS 模型失败: {onnx_path} error={e}") from e
        return _models[voice]


def _resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """16bit mono PCM 重采样（sinc 插值，避免线性插值的混叠失真）"""
    import torch
    from torchaudio.functional import resample

    samples = torch.frombuffer(bytearray(pcm), dtype=torch.int16).float().div_(32768.0)
    resampled = resample(samples, orig_freq=src_rate, new_freq=dst_rate)
    return (
        (resampled * 32768.0).clamp(-32768, 32767).to(torch.int16).numpy().tobytes()
    )


def _encode_wav(pcm: bytes, sample_rate: int) -> bytes:
    """PCM 封装为 WAV（16bit/mono）"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _encode_mp3(pcm: bytes, sample_rate: int) -> bytes:
    """PCM 编码为 MP3（64kbps 单声道，语音场景足够）"""
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())


def synthesize(
    text: str, voice: str, speed: float, format_: str, sample_rate: int
) -> bytes:
    """本地合成语音（阻塞调用，须经线程池执行），返回编码后的音频字节。

    - voice 必须是 _VOICE_MODEL_FILES 中已注册的音色
    - speed 为播放倍速（0.5~2.0），映射 Piper length_scale = 1/speed
    - format_ 为 mp3/wav/pcm；输出采样率重采样至 sample_rate
    """
    if voice not in _VOICE_MODEL_FILES:
        raise LocalTtsError(f"不支持的音色: {voice}（可选: {'/'.join(_VOICE_MODEL_FILES)}）")
    model = _load_model(voice)

    try:
        from piper import SynthesisConfig

        config = SynthesisConfig(length_scale=1.0 / speed)
        pcm = bytearray()
        for chunk in model.synthesize(text, config):
            pcm.extend(chunk.audio_int16_bytes)
    except Exception as e:
        raise LocalTtsError(f"语音合成推理失败: {e}") from e
    if not pcm:
        raise LocalTtsError("合成结果为空音频（文本无可发音内容）")

    pcm_bytes = bytes(pcm)
    native_rate = model.config.sample_rate
    if sample_rate != native_rate:
        pcm_bytes = _resample(pcm_bytes, native_rate, sample_rate)

    if format_ == "pcm":
        return pcm_bytes
    if format_ == "wav":
        return _encode_wav(pcm_bytes, sample_rate)
    if format_ == "mp3":
        return _encode_mp3(pcm_bytes, sample_rate)
    raise LocalTtsError(f"不支持的音频格式: {format_}")


def run_in_executor(func, *args):
    """将阻塞的引擎调用提交到推理线程池，返回 awaitable"""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_executor, func, *args)
