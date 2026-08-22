"""本地 TTS（Piper 引擎 + tts_service）单元测试

覆盖：
- 引擎真实合成（模型已在 models/piper/ 自动下载）：中文/对抗性脏语料（全角半角标点混杂、
  emoji、零宽字符、超长无分隔行）、语速边界 0.5/2.0、mp3/wav/pcm 三格式 × 四采样率不变量
- 服务层：参数校验、缓存命中不重复合成、引擎异常转业务异常
- 缓存 Key 与 AES-GCM 加解密（含空串/随机字节/篡改密文）

对抗性语料标准：不使用规整人造输入，断言通用性质（不崩溃、格式可解析、采样率正确、非空）。
"""

import io
import os
import random
import string
import wave
from types import SimpleNamespace

import pytest
from fakeredis import FakeAsyncRedis

from app.config import settings
from app.core.exceptions import BusinessException
from app.infrastructure.voice import piper_tts_engine
from app.infrastructure.voice.piper_tts_engine import LocalTtsError
from app.service.voice import tts_service as m
from app.service.voice.tts_service import (
    tts_service,
    _cache_key,
    decrypt_audio,
    encrypt_audio,
)

_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "piper",
)
_MODEL_READY = os.path.exists(os.path.join(_SAMPLE_DIR, "zh_CN-huayan-medium.onnx"))

# 对抗性脏语料：全角/半角标点混杂、emoji、零宽字符、CRLF、连续空行
_DIRTY_CORPUS = [
    "处理完成；done.第二段！third？mixed，标点。",
    "警告⚠️任务失败❌请重试😀",
    "零宽​字符﻿测试",
    "CRLF\r\n换行\n\n\n连续空行\r\r尾随",
    "无分隔超长行" + "去" * 300,
    "数字混合：第3.14版，共10,000条，增长率95%",
]


# ==================== 缓存 Key ====================


def test_cache_key_deterministic_and_distinct():
    key1 = _cache_key("相同文本", "huayan", 1.0)
    assert key1 == _cache_key("相同文本", "huayan", 1.0)
    assert key1 != _cache_key("不同文本", "huayan", 1.0)
    assert key1 != _cache_key("相同文本", "other", 1.0)
    assert key1 != _cache_key("相同文本", "huayan", 0.8)
    assert len(key1) == 64  # SHA256 hex


def test_cache_key_adversarial_inputs_stable():
    """对抗性输入下 Key 仍为稳定哈希（同输入同 Key，不同输入不同 Key）"""
    corpus = _DIRTY_CORPUS + ["", " ", " ​"]
    keys = [_cache_key(t, "huayan", 1.0) for t in corpus]
    assert all(len(k) == 64 for k in keys)
    assert len(set(keys)) == len(keys)


# ==================== AES-GCM 加解密 ====================


def test_encrypt_decrypt_roundtrip():
    rng = random.Random(20260822)
    cases = [b"", b"\x00", bytes(rng.randrange(256) for _ in range(1024)), os.urandom(65536)]
    for raw in cases:
        blob = encrypt_audio(raw)
        assert blob != raw  # 密文不含明文
        assert decrypt_audio(blob) == raw


def test_decrypt_tampered_ciphertext_fails():
    blob = bytearray(encrypt_audio(b"secret audio"))
    blob[-1] ^= 0xFF  # 篡改 GCM tag 附近字节
    with pytest.raises(Exception):
        decrypt_audio(bytes(blob))


def test_decrypt_garbage_fails():
    rng = random.Random(7)
    with pytest.raises(Exception):
        decrypt_audio(bytes(rng.randrange(256) for _ in range(64)))


# ==================== 参数校验 ====================


def test_validate_params_rejects_bad_input():
    cases = [
        ("", "huayan", "mp3", 16000),  # 空文本
        ("  \n\t ", "huayan", "mp3", 16000),  # 纯空白
        ("x", "aixia", "mp3", 16000),  # 已下线音色
        ("x", "huayan", "flac", 16000),  # 非法格式
        ("x", "huayan", "mp3", 44100),  # 非法采样率
        ("x", "huayan", "mp3", 0),
    ]
    for text, voice, fmt, rate in cases:
        with pytest.raises(BusinessException):
            tts_service._validate_params(text, voice, fmt, rate)


def test_validate_params_rejects_overlong_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "VOICE_TTS_MAX_TEXT_LENGTH", 5)
    with pytest.raises(BusinessException):
        tts_service._validate_params("超过五个字的文本", "huayan", "mp3", 16000)
    # 边界内通过
    tts_service._validate_params("五个字内", "huayan", "mp3", 16000)


def test_validate_params_accepts_catalog_voices():
    from app.models.schema.voice_tts import VOICE_CATALOG

    for v in VOICE_CATALOG:
        tts_service._validate_params("文本", v["id"], "wav", 8000)


# ==================== 引擎真实合成（模型自动下载） ====================


@pytest.fixture(scope="module")
def synthesized_default() -> bytes:
    """模块级一次性合成（模型懒加载仅一次），供多条断言复用"""
    return piper_tts_engine.synthesize("图像去雾处理完成，请查看结果。", "huayan", 1.0, "wav", 16000)


@pytest.mark.skipif(not _MODEL_READY, reason="TTS 模型未就绪（首次运行会自动下载，本机应存在）")
class TestEngineSynthesis:
    def test_wav_format_invariants(self, synthesized_default: bytes):
        assert synthesized_default[:4] == b"RIFF"
        with wave.open(io.BytesIO(synthesized_default)) as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getnframes() > 0

    @pytest.mark.parametrize(
        "sample_rate", [8000, 16000, 24000, 48000], ids=lambda r: f"{r}Hz"
    )
    def test_resample_to_all_sample_rates(self, sample_rate: int):
        wav = piper_tts_engine.synthesize("采样率测试", "huayan", 1.0, "wav", sample_rate)
        with wave.open(io.BytesIO(wav)) as w:
            assert w.getframerate() == sample_rate
            assert w.getnframes() > 0

    def test_pcm_format_is_raw_int16_mono(self):
        pcm = piper_tts_engine.synthesize("裸流测试", "huayan", 1.0, "pcm", 16000)
        assert pcm[:4] != b"RIFF"  # 无 WAV 头
        assert len(pcm) % 2 == 0  # 16bit 对齐
        assert len(pcm) > 1000

    def test_mp3_format_is_mpeg_stream(self):
        mp3 = piper_tts_engine.synthesize("编码测试", "huayan", 1.0, "mp3", 16000)
        assert mp3[0] == 0xFF and mp3[1] & 0xE0 == 0xE0  # MPEG 帧同步字
        assert len(mp3) > 1000

    @pytest.mark.parametrize("text", _DIRTY_CORPUS, ids=lambda t: repr(t[:12]))
    def test_adversarial_corpus_never_crashes(self, text: str):
        """对抗性脏语料：不崩溃，输出可解析、非空、采样率正确"""
        wav = piper_tts_engine.synthesize(text, "huayan", 1.0, "wav", 16000)
        with wave.open(io.BytesIO(wav)) as w:
            assert w.getframerate() == 16000
            assert w.getnframes() > 0

    def test_speed_boundaries_scale_audio_length(self):
        """语速边界：speed=2.0 的音频显著短于 speed=0.5（length_scale 反比；合成带随机
        噪声，按生成时长比例 1.5 倍以上放宽断言，避免抖动）"""
        slow = piper_tts_engine.synthesize("语速边界测试，一句话。", "huayan", 0.5, "pcm", 16000)
        fast = piper_tts_engine.synthesize("语速边界测试，一句话。", "huayan", 2.0, "pcm", 16000)
        assert len(fast) * 1.5 < len(slow)

    def test_unknown_voice_rejected(self):
        with pytest.raises(LocalTtsError, match="不支持的音色"):
            piper_tts_engine.synthesize("文本", "aixia", 1.0, "mp3", 16000)

    def test_unsupported_format_rejected(self):
        with pytest.raises(LocalTtsError, match="不支持的音频格式"):
            piper_tts_engine.synthesize("文本", "huayan", 1.0, "flac", 16000)

    def test_whitespace_only_text_yields_error(self):
        """无可发音内容（纯空白）应显式报错而非返回空音频"""
        with pytest.raises(LocalTtsError, match="空音频"):
            piper_tts_engine.synthesize("   ", "huayan", 1.0, "wav", 16000)

    def test_random_text_invariants(self):
        """固定 seed 随机文本：不崩溃、输出为合法 WAV（通用性质，不校验内容）"""
        rng = random.Random(42)
        for _ in range(3):
            text = "".join(rng.choice(string.printable) for _ in range(80)).strip()
            if not text:
                continue
            wav = piper_tts_engine.synthesize(text, "huayan", 1.0, "wav", 8000)
            with wave.open(io.BytesIO(wav)) as w:
                assert w.getframerate() == 8000


# ==================== 服务层（缓存命中 / 异常转换） ====================


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


def _install_service_stubs(monkeypatch: pytest.MonkeyPatch, synth_calls: list[str]):
    """桩掉文件存储与计费，仅考察 tts_service 缓存编排；合成走真实引擎"""

    async def _upload_file(db, *, filename, content, content_type):
        synth_calls.append("store")
        return SimpleNamespace(id=101)

    async def _ensure_balance(db, user_id, estimated):
        synth_calls.append("check")

    async def _charge_tts(db, user_id, text_chars):
        synth_calls.append("charge")

    monkeypatch.setattr(m.file_service, "upload_file", staticmethod(_upload_file))
    monkeypatch.setattr(m.voice_billing_service, "ensure_balance", staticmethod(_ensure_balance))
    monkeypatch.setattr(m.voice_billing_service, "charge_tts", staticmethod(_charge_tts))


@pytest.mark.skipif(not _MODEL_READY, reason="TTS 模型未就绪（首次运行会自动下载，本机应存在）")
async def test_synthesize_caches_second_call(redis: FakeAsyncRedis, monkeypatch):
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)

    first = await tts_service.synthesize(
        None, redis, 1, "缓存命中测试", "huayan", 1.0, "wav", 16000
    )
    assert first["audioUrl"].startswith("/api/v1/voice/tts/audio/")
    assert first["format"] == "wav"
    assert calls == ["check", "store", "charge"]

    calls.clear()
    second = await tts_service.synthesize(
        None, redis, 1, "缓存命中测试", "huayan", 1.0, "wav", 16000
    )
    assert second["audioUrl"] == first["audioUrl"]  # 命中缓存
    assert calls == []  # 不再合成、不重复扣费

    # 语速不同 → 缓存 Key 不同 → 重新合成
    third = await tts_service.synthesize(
        None, redis, 1, "缓存命中测试", "huayan", 0.8, "wav", 16000
    )
    assert third["audioUrl"] != first["audioUrl"]
    assert calls == ["check", "store", "charge"]


@pytest.mark.skipif(not _MODEL_READY, reason="TTS 模型未就绪（首次运行会自动下载，本机应存在）")
async def test_synthesize_default_voice_applied(redis: FakeAsyncRedis, monkeypatch):
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)
    # voice=None → 使用默认音色配置
    result = await tts_service.synthesize(None, redis, 1, "默认音色", None, 1.0, "mp3", 16000)
    assert result["format"] == "mp3"


@pytest.mark.skipif(not _MODEL_READY, reason="TTS 模型未就绪（首次运行会自动下载，本机应存在）")
async def test_synthesize_engine_error_wrapped(redis: FakeAsyncRedis, monkeypatch):
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)

    def _boom(*args, **kwargs):
        raise LocalTtsError("引擎故障模拟")

    # 引擎提交入口在服务层经 run_in_executor 调用（线程池内同步执行），桩引擎 synthesize 函数
    monkeypatch.setattr(m.piper_tts_engine, "synthesize", _boom)
    with pytest.raises(BusinessException):
        await tts_service.synthesize(None, redis, 1, "触发故障", "huayan", 1.0, "wav", 16000)
    assert calls == ["check"]  # 失败不存储不扣费


async def test_synthesize_rejects_unknown_voice_without_synth(redis: FakeAsyncRedis):
    with pytest.raises(BusinessException):
        await tts_service.synthesize(None, redis, 1, "文本", "aixia", 1.0, "wav", 16000)
