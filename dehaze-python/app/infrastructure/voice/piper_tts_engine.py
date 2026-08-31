"""本地 TTS 引擎（Piper，进程内部署，GPU 优先 / CPU 回退）

对齐 FunASR 引擎的部署形态（后端实现 §3.3）：
- 模型懒加载：首次合成请求时才加载 ONNX 模型，未使用语音功能时不占内存
- GPU 优先：onnxruntime 带 CUDAExecutionProvider 时经 use_cuda 启用 GPU 合成
  （需安装 onnxruntime-gpu 替换 CPU 版），否则纯 CPU
- 模型自动下载：文件缺失时经 hf-mirror 下载（断点续传 + fcntl 跨进程锁，
  对齐 local_llm_model 的零手工部署策略；Windows 无 fcntl 退化为无锁直下），
  回退 huggingface.co
- 推理在专用线程池执行（不阻塞事件循环；espeak 音素化由 Piper
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

# 音色模型配置：onnx 文件名、远端大小、下载 URL 模板
class _VoiceConfig:
    __slots__ = ("onnx", "size", "url_templates")

    def __init__(self, onnx: str, size: int | None = None, url_templates: list[str] | None = None):
        self.onnx = onnx
        self.size = size
        self.url_templates = url_templates or []

# 音色注册表 {音色: VoiceConfig}，由 LocalTtsProvider 从 sys_voice_model 解析后注入
_voice_configs: dict[str, _VoiceConfig] = {}


def configure_voices(configs: dict[str, dict]) -> None:
    """注入音色注册表（{音色: {"onnx","size","urls"}}，替换式更新），元信息由 sys_voice_model 决定"""
    _voice_configs.clear()
    for voice, cfg in configs.items():
        _voice_configs[voice] = _VoiceConfig(
            onnx=cfg.get("onnx"),
            size=cfg.get("size"),
            url_templates=cfg.get("urls"),
        )

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
    return os.path.join(settings.MODEL_CACHE_DIR, "piper", _voice_configs[voice].onnx)


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
    cfg = _voice_configs[voice]
    onnx_path = _model_path(voice)
    json_path = onnx_path + ".json"
    if _file_ready(onnx_path, cfg.size) and _file_ready(json_path, None):
        return onnx_path
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    lock_file = open(onnx_path + ".lock", "w") if fcntl else None  # Windows：无锁直下
    try:
        if lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not _file_ready(onnx_path, cfg.size):
            _download_with_fallback(cfg.onnx, onnx_path, cfg.size, cfg.url_templates)
        if not _file_ready(json_path, None):
            _download_with_fallback(cfg.onnx + ".json", json_path, None, cfg.url_templates)
        return onnx_path
    finally:
        if lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()


def _download_with_fallback(
    filename: str, path: str, expected_size: int | None, url_templates: list[str]
) -> None:
    if not url_templates:
        raise LocalTtsError(f"TTS 模型 {filename} 未配置下载地址，请手动放置于 models/piper/ 目录")
    last_error: Exception | None = None
    for url_tpl in url_templates:
        try:
            _download_file(url_tpl.format(filename=filename), path, expected_size)
            logger.info("TTS 模型文件就绪：%s（%dMB）", path, os.path.getsize(path) // 1048576)
            return
        except Exception as exc:  # noqa: BLE001 逐镜像尝试，全部失败才抛出
            last_error = exc
            logger.warning("TTS 模型下载失败（%s）：%s，尝试下一个镜像", url_tpl, exc)
    raise LocalTtsError(
        f"TTS 模型自动下载失败（{filename}）：{last_error}。"
        "请检查网络可达 hf-mirror.com / hugginggingface.co，或手动下载放置于 models/piper/ 目录"
    ) from last_error


def _use_cuda() -> bool:
    """onnxruntime 带 CUDAExecutionProvider 时启用 GPU 合成（需安装 onnxruntime-gpu）"""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001 检测失败按纯 CPU 处理
        return False


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
        use_cuda = _use_cuda()
        logger.info("加载本地 TTS 模型: %s（%s）", onnx_path, "GPU" if use_cuda else "CPU")
        try:
            _models[voice] = PiperVoice.load(onnx_path, use_cuda=use_cuda)
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
    if voice not in _voice_configs:
        raise LocalTtsError(f"不支持的音色: {voice}（可选: {'/'.join(_voice_configs) or '未配置'}）")
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


def engine_status() -> dict[str, Any]:
    """查询引擎状态（不抛异常）。

    返回：engine_status("online"/"offline")、voice_model_loaded，
    基于默认音色模型是否加载到进程内单例判定。
    """
    if not _models:
        return {
            "engine_status": "offline",
            "voice_model_loaded": False,
        }
    default_voice = next(iter(_voice_configs), None)
    return {
        "engine_status": "online",
        "voice_model_loaded": default_voice in _models,
    }
