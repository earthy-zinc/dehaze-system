"""语音服务状态聚合单元测试（T-VS-060~T-VS-068）。

对齐 SDK ServiceStatusVO（camelCase）：asr.concurrentSessions / engineStatus 等。
引擎状态由基础设施层 engine_status() 查询，测试中 monkeypatch 两个引擎函数，
仅断言聚合返回的字段结构与并发计数（与 Redis ZCARD 一致）。
"""

from types import SimpleNamespace

import pytest
from fakeredis import FakeAsyncRedis

from app.service.voice import voice_service_status as vs_module
from app.service.voice.voice_service_status import voice_service_status, _CONCURRENT_KEY


def _patch_engines(monkeypatch, *, asr_online=True, tts_online=True):
    # funasr_status / piper_status 是 voice_service_status 模块级导入的引擎函数别名，
    # 需 patch 模块命名空间内的引用
    monkeypatch.setattr(
        vs_module,
        "funasr_status",
        lambda: {
            "engine_status": "online" if asr_online else "offline",
            "stream_model_loaded": asr_online,
            "offline_model_loaded": asr_online,
        },
    )
    monkeypatch.setattr(
        vs_module,
        "piper_status",
        lambda: {
            "engine_status": "online" if tts_online else "offline",
            "voice_model_loaded": tts_online,
        },
    )


# ── T-VS-060/061：引擎 online 时字段正确、concurrentSessions 与 Redis ZCARD 一致 ──


@pytest.mark.asyncio
async def test_status_online_reports_correct_fields_and_concurrency(monkeypatch):
    import time

    _patch_engines(monkeypatch, asr_online=True, tts_online=True)
    redis = FakeAsyncRedis(decode_responses=True)
    now = time.time()
    for i in range(3):
        await redis.zadd(_CONCURRENT_KEY, {f"sess-{i}": now - i})

    status = await voice_service_status.get_status(redis)

    assert status["asr"]["engineStatus"] == "online"
    assert status["asr"]["streamModelLoaded"] is True
    assert status["asr"]["offlineModelLoaded"] is True
    assert status["asr"]["concurrentSessions"] == 3
    assert status["asr"]["maxConcurrentSessions"] == 50
    assert status["tts"]["engineStatus"] == "online"
    assert status["tts"]["voiceModelLoaded"] is True


# ── T-VS-062：引擎 offline 时不抛异常 ──


@pytest.mark.asyncio
async def test_status_offline_does_not_raise(monkeypatch):
    _patch_engines(monkeypatch, asr_online=False, tts_online=False)
    redis = FakeAsyncRedis(decode_responses=True)

    status = await voice_service_status.get_status(redis)

    assert status["asr"]["engineStatus"] == "offline"
    assert status["asr"]["streamModelLoaded"] is False
    assert status["asr"]["offlineModelLoaded"] is False
    assert status["tts"]["engineStatus"] == "offline"
    assert status["tts"]["voiceModelLoaded"] is False


# ── T-VS-063：concurrentSessions 与 Redis ZCARD 一致（含剪枝）──


@pytest.mark.asyncio
async def test_status_concurrent_sessions_matches_zcard(monkeypatch):
    _patch_engines(monkeypatch)
    redis = FakeAsyncRedis(decode_responses=True)
    # 一个过期会话（score 在剪枝窗口之前）应被剔除
    await redis.zadd(_CONCURRENT_KEY, {"expired": 0})
    await redis.zadd(_CONCURRENT_KEY, {"active": __import__("time").time()})

    status = await voice_service_status.get_status(redis)

    assert status["asr"]["concurrentSessions"] == 1


# ── T-VS-064/065：返回结构对齐 SDK ServiceStatusVO（camelCase 字段齐全）──


@pytest.mark.asyncio
async def test_status_response_shape_matches_sdk_vo(monkeypatch):
    _patch_engines(monkeypatch)
    redis = FakeAsyncRedis(decode_responses=True)

    status = await voice_service_status.get_status(redis)

    assert set(status.keys()) == {"asr", "tts"}
    assert set(status["asr"].keys()) == {
        "engineStatus",
        "concurrentSessions",
        "maxConcurrentSessions",
        "streamModelLoaded",
        "offlineModelLoaded",
    }
    assert set(status["tts"].keys()) == {"engineStatus", "voiceModelLoaded"}
