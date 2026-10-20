"""TTS 语音合成服务单元测试

覆盖：
- 服务层：参数校验、缓存命中不重复合成、引擎异常转业务异常（含引擎已映射错误码不降级）
- 缓存 Key 与 AES-GCM 加解密（含空串/随机字节/篡改密文）

合成本身由外置引擎服务（clarity-service/clarity-voice）承担，此处以 Provider 桩替代；
引擎的音色权重解析、编码格式与采样率不变量由 clarity-voice 侧测试覆盖。
"""

import os
import random
from types import SimpleNamespace

import pytest
from cryptography.exceptions import InvalidTag
from fakeredis import FakeAsyncRedis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.voice import tts_service as m
from app.service.voice.tts_service import (
    _cache_key,
    decrypt_audio,
    encrypt_audio,
    tts_service,
)

# 直连测试的 db 替身：合成链路中 db 仅透传给被替换的 file/billing 桩，不真正落库
_DB = AsyncSession()


class FakeAudioClient:
    """语音协议适配器桩：可预设合成字节或抛异常。

    tts_service 经模块级 `get_audio_client(protocol_type)` 取适配器（按供应商协议族分派）、
    经 `resolve_voice_model` 解析平台默认 tts 模型，实扣透传统一 model_id（默认 huayan）。
    """

    model_id = "huayan"
    provider_id = 78

    def __init__(self, audio=b"fake-audio", error=None):
        self._audio = audio
        self._error = error

    async def synthesize(
        self, provider, model_id, text, voice, speed, response_format, sample_rate
    ):
        if self._error is not None:
            raise self._error
        return self._audio


_ROUTE_STATE = {"model_id": "huayan"}


async def _fake_resolve_tts_model(db, model_type):
    """统一模型路由桩：返回固定 tts 模型与供应商（模型经 _make_svc 的 route_model_id 定制）。"""
    return SimpleNamespace(
        model_id=_ROUTE_STATE["model_id"],
        provider=SimpleNamespace(id=78, provider_code="local-voice", protocol_type="openai_audio"),
    )


def _make_svc(monkeypatch, client=None, route_model_id=None):
    """构造 TtsService 并注入适配器/路由桩（file_service/voice_billing_service 走模块单例）。"""
    if route_model_id is not None:
        _ROUTE_STATE["model_id"] = route_model_id
    monkeypatch.setattr(m, "get_audio_client", lambda protocol_type: client or FakeAudioClient())
    monkeypatch.setattr(m, "resolve_voice_model", _fake_resolve_tts_model)
    return tts_service.__class__()


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
    corpus = [*_DIRTY_CORPUS, "", " ", " \u200b"]
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
    with pytest.raises(InvalidTag):
        decrypt_audio(bytes(blob))


def test_decrypt_garbage_fails():
    rng = random.Random(7)
    with pytest.raises(InvalidTag):
        decrypt_audio(bytes(rng.randrange(256) for _ in range(64)))


# ==================== 参数校验 ====================


def test_validate_params_rejects_bad_input():
    cases = [
        ("", "huayan", "mp3", 16000, "openai_audio"),  # 空文本
        ("  \n\t ", "huayan", "mp3", 16000, "openai_audio"),  # 纯空白
        ("x", "aixia", "mp3", 16000, "openai_audio"),  # 已下线音色
        ("x", "huayan", "flac", 16000, "openai_audio"),  # 非法格式
        ("x", "huayan", "mp3", 44100, "openai_audio"),  # 非法采样率
        ("x", "huayan", "mp3", 0, "openai_audio"),
        ("x", "huayan", "mp3", 16000, "volc_realtime"),  # 内置音色不属于火山协议族目录
    ]
    for text, voice, fmt, rate, protocol in cases:
        with pytest.raises(BusinessException):
            tts_service._validate_params(text, voice, fmt, rate, protocol)


def test_validate_params_rejects_overlong_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "VOICE_TTS_MAX_TEXT_LENGTH", 5)
    with pytest.raises(BusinessException):
        tts_service._validate_params("超过五个字的文本", "huayan", "mp3", 16000, "openai_audio")
    # 边界内通过
    tts_service._validate_params("五个字内", "huayan", "mp3", 16000, "openai_audio")


def test_validate_params_accepts_catalog_voices():
    """每个协议族的目录内音色均通过；火山音色 ID 超旧 32 字符上限仍可校验通过"""
    from app.models.schema.voice_tts import get_voice_catalog

    for protocol in ("openai_audio", "volc_realtime"):
        for v in get_voice_catalog(protocol):
            tts_service._validate_params("文本", v["id"], "wav", 8000, protocol)


def test_validate_params_accepts_volc_voice_beyond_old_length_limit():
    """回归：火山 voice_type（如 zh_female_shuangkuaisisi_uranus_bigtts）长度 > 32，
    放开 TtsForm.voice 上限后可用"""
    voice = "zh_female_shuangkuaisisi_uranus_bigtts"
    assert len(voice) > 32
    tts_service._validate_params("文本", voice, "wav", 8000, "volc_realtime")


# ==================== 服务层（缓存命中 / 异常转换） ====================


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


def _install_service_stubs(monkeypatch: pytest.MonkeyPatch, synth_calls: list[str]):
    """桩掉文件存储与计费，仅考察 tts_service 缓存编排；合成走真实引擎

    单价口径与字典读取由 test_voice_billing_service 覆盖，此处仅保留编排所需返回值；
    实扣记录 "charge:{engine_code}" 以断言引擎编码透传。
    """

    async def _upload_file(db, *, filename, content, content_type):
        synth_calls.append("store")
        return SimpleNamespace(id=101)

    async def _estimate_tts(db, model_id, provider_id, text_chars):
        return text_chars

    async def _ensure_balance(db, user_id, estimated):
        synth_calls.append("check")

    async def _charge_tts(db, user_id, model_id, provider_id, text_chars):
        synth_calls.append(f"charge:{model_id}")

    monkeypatch.setattr(m.file_service, "upload_file", staticmethod(_upload_file))
    monkeypatch.setattr(
        m.voice_billing_service, "estimate_tts_credits", staticmethod(_estimate_tts)
    )
    monkeypatch.setattr(m.voice_billing_service, "ensure_balance", staticmethod(_ensure_balance))
    monkeypatch.setattr(m.voice_billing_service, "charge_tts", staticmethod(_charge_tts))


async def test_synthesize_caches_second_call(redis: FakeAsyncRedis, monkeypatch):
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)
    svc = _make_svc(monkeypatch)

    first = await svc.synthesize(_DB, redis, 1, "缓存命中测试", "huayan", 1.0, "wav", 16000)
    assert first["audioUrl"].startswith("/api/v1/voice/tts/audio/")
    assert first["format"] == "wav"
    assert calls == ["check", "store", "charge:huayan"]

    calls.clear()
    second = await svc.synthesize(_DB, redis, 1, "缓存命中测试", "huayan", 1.0, "wav", 16000)
    assert second["audioUrl"] == first["audioUrl"]  # 命中缓存
    assert calls == []  # 不再合成、不重复扣费

    # 语速不同 → 缓存 Key 不同 → 重新合成
    third = await svc.synthesize(_DB, redis, 1, "缓存命中测试", "huayan", 0.8, "wav", 16000)
    assert third["audioUrl"] != first["audioUrl"]
    assert calls == ["check", "store", "charge:huayan"]


async def test_synthesize_charges_with_engine_provider_code(redis: FakeAsyncRedis, monkeypatch):
    """实扣携带实际合成引擎编码（第三方引擎按供应商对账）"""
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)
    svc = _make_svc(monkeypatch, route_model_id="test-tts-model")

    await svc.synthesize(_DB, redis, 1, "模型编码", "huayan", 1.0, "mp3", 16000)

    assert calls == ["check", "store", "charge:test-tts-model"]


async def test_synthesize_default_voice_applied(redis: FakeAsyncRedis, monkeypatch):
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)
    svc = _make_svc(monkeypatch)
    # voice=None → 使用默认音色配置
    result = await svc.synthesize(_DB, redis, 1, "默认音色", None, 1.0, "mp3", 16000)
    assert result["format"] == "mp3"


async def test_synthesize_engine_error_wrapped(redis: FakeAsyncRedis, monkeypatch):
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)
    # Provider 合成抛未映射异常 → 服务层包装为业务异常，不存储不扣费
    svc = _make_svc(monkeypatch, client=FakeAudioClient(error=RuntimeError("引擎故障模拟")))
    with pytest.raises(BusinessException):
        await svc.synthesize(_DB, redis, 1, "触发故障", "huayan", 1.0, "wav", 16000)
    assert calls == ["check"]  # 失败不存储不扣费


async def test_synthesize_keeps_engine_mapped_error_code(redis: FakeAsyncRedis, monkeypatch):
    """引擎侧映射后的业务异常（如 A0400 参数类）原样上抛，不降级为 A0500"""
    calls: list[str] = []
    _install_service_stubs(monkeypatch, calls)
    mapped = BusinessException(ResultCode.PARAM_ERROR, "语音引擎服务错误：sample_rate 非法")
    svc = _make_svc(monkeypatch, client=FakeAudioClient(error=mapped))
    with pytest.raises(BusinessException) as exc:
        await svc.synthesize(_DB, redis, 1, "触发故障", "huayan", 1.0, "wav", 16000)
    assert exc.value.code == ResultCode.PARAM_ERROR
    assert calls == ["check"]


async def test_synthesize_rejects_unknown_voice_without_synth(
    redis: FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
):
    svc = _make_svc(monkeypatch)
    with pytest.raises(BusinessException):
        await svc.synthesize(_DB, redis, 1, "文本", "aixia", 1.0, "wav", 16000)
