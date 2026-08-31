"""语音服务状态聚合单元测试（T-VS-060~T-VS-068）。

对齐 SDK ServiceStatusVO（camelCase）：asr.concurrentSessions / engineStatus 等。
聚合经注册表解析默认引擎：default=local 时引擎状态由基础设施层 engine_status()
查询（测试 monkeypatch 引擎函数）；default=cloud 时按健康开关 + 熔断标记上报。
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis

from app.service.voice import voice_service_status as vs_module
from app.service.voice.voice_service_status import (
    VoiceServiceStatusService,
    _CIRCUIT_KEY,
    _CONCURRENT_KEY,
    voice_service_status,
)


def _local_row(engine_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=1, provider_code="local", engine_type=engine_type, health_check_enabled=1
    )


def _cloud_row(engine_type: str, *, health_check_enabled: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        id=9, provider_code="aliyun", engine_type=engine_type,
        health_check_enabled=health_check_enabled,
    )


def _make_service(default_row: SimpleNamespace | None) -> VoiceServiceStatusService:
    """注入桩注册表：resolve_default_engine 返回预置默认引擎配置行"""
    return VoiceServiceStatusService(
        engine_registry=SimpleNamespace(
            resolve_default_engine=AsyncMock(return_value=default_row)
        )
    )


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


# ── T-VS-060/061：默认 local 引擎 online 时字段正确、concurrentSessions 与 Redis ZCARD 一致 ──


@pytest.mark.asyncio
async def test_status_online_reports_correct_fields_and_concurrency(monkeypatch):
    _patch_engines(monkeypatch, asr_online=True, tts_online=True)
    service = _make_service(_local_row("asr"))
    redis = FakeAsyncRedis(decode_responses=True)
    now = time.time()
    for i in range(3):
        await redis.zadd(_CONCURRENT_KEY, {f"sess-{i}": now - i})

    status = await service.get_status(redis)

    assert status["asr"]["engineStatus"] == "online"
    assert status["asr"]["streamModelLoaded"] is True
    assert status["asr"]["offlineModelLoaded"] is True
    assert status["asr"]["concurrentSessions"] == 3
    assert status["asr"]["maxConcurrentSessions"] == 50
    assert status["tts"]["engineStatus"] == "online"
    assert status["tts"]["voiceModelLoaded"] is True


# ── T-VS-062：默认 local 引擎 offline 时不抛异常 ──


@pytest.mark.asyncio
async def test_status_offline_does_not_raise(monkeypatch):
    _patch_engines(monkeypatch, asr_online=False, tts_online=False)
    service = _make_service(_local_row("asr"))
    redis = FakeAsyncRedis(decode_responses=True)

    status = await service.get_status(redis)

    assert status["asr"]["engineStatus"] == "offline"
    assert status["asr"]["streamModelLoaded"] is False
    assert status["asr"]["offlineModelLoaded"] is False
    assert status["tts"]["engineStatus"] == "offline"
    assert status["tts"]["voiceModelLoaded"] is False


# ── T-VS-063：concurrentSessions 与 Redis ZCARD 一致（含剪枝）──


@pytest.mark.asyncio
async def test_status_concurrent_sessions_matches_zcard(monkeypatch):
    _patch_engines(monkeypatch)
    service = _make_service(_local_row("asr"))
    redis = FakeAsyncRedis(decode_responses=True)
    # 一个过期会话（score 在剪枝窗口之前）应被剔除
    await redis.zadd(_CONCURRENT_KEY, {"expired": 0})
    await redis.zadd(_CONCURRENT_KEY, {"active": time.time()})

    status = await service.get_status(redis)

    assert status["asr"]["concurrentSessions"] == 1


# ── T-VS-064/065：返回结构对齐 SDK ServiceStatusVO（camelCase 字段齐全）──


@pytest.mark.asyncio
async def test_status_response_shape_matches_sdk_vo(monkeypatch):
    _patch_engines(monkeypatch)
    service = _make_service(_local_row("asr"))
    redis = FakeAsyncRedis(decode_responses=True)

    status = await service.get_status(redis)

    assert set(status.keys()) == {"asr", "tts"}
    assert set(status["asr"].keys()) == {
        "engineStatus",
        "concurrentSessions",
        "maxConcurrentSessions",
        "streamModelLoaded",
        "offlineModelLoaded",
    }
    assert set(status["tts"].keys()) == {"engineStatus", "voiceModelLoaded"}


# ── 默认引擎为 cloud 时状态上报（后端实现 §2.4：健康开关 + 熔断标记）──


@pytest.mark.asyncio
async def test_cloud_default_health_check_enabled_reports_offline():
    """默认云端引擎 + 健康检查开启（未熔断）：厂商协议未接入无法真实探活 → offline"""
    service = _make_service(_cloud_row("asr"))
    redis = FakeAsyncRedis(decode_responses=True)

    status = await service.get_status(redis)

    assert status["asr"]["engineStatus"] == "offline"
    # 云端引擎无进程内模型概念，模型加载字段按 False 上报，响应结构不变
    assert status["asr"]["streamModelLoaded"] is False
    assert status["asr"]["concurrentSessions"] == 0


@pytest.mark.asyncio
async def test_cloud_default_circuit_open_reports_offline():
    """默认云端引擎 + 熔断标记（voice:provider:{id}:circuit_open）→ offline"""
    row = _cloud_row("tts")
    service = _make_service(row)
    redis = FakeAsyncRedis(decode_responses=True)
    await redis.set(_CIRCUIT_KEY.format(row.id), 1)

    status = await service.get_status(redis)

    assert status["tts"]["engineStatus"] == "offline"


@pytest.mark.asyncio
async def test_cloud_default_health_check_disabled_reports_online():
    """默认云端引擎 + 健康检查关闭：不参与判定视为健康（对齐 provider_health_service）→ online"""
    service = _make_service(_cloud_row("asr", health_check_enabled=0))
    redis = FakeAsyncRedis(decode_responses=True)

    status = await service.get_status(redis)

    assert status["asr"]["engineStatus"] == "online"


# ── T-VS-067：默认引擎未配置/解析失败时不抛异常，按 offline 上报 ──


@pytest.mark.asyncio
async def test_unconfigured_default_engine_reports_offline():
    service = _make_service(None)
    redis = FakeAsyncRedis(decode_responses=True)

    status = await service.get_status(redis)

    assert status["asr"]["engineStatus"] == "offline"
    assert status["tts"]["engineStatus"] == "offline"


@pytest.mark.asyncio
async def test_registry_resolution_failure_reports_offline():
    """注册表解析抛错（如 DB 异常）→ 状态聚合不抛异常，按 offline 上报"""
    service = VoiceServiceStatusService(
        engine_registry=SimpleNamespace(
            resolve_default_engine=AsyncMock(side_effect=RuntimeError("db down"))
        )
    )
    redis = FakeAsyncRedis(decode_responses=True)

    status = await service.get_status(redis)

    assert status["asr"]["engineStatus"] == "offline"
    assert status["tts"]["engineStatus"] == "offline"


# ── 兼容入口：模块单例仍暴露 get_status（router 直接消费）──


@pytest.mark.asyncio
async def test_singleton_get_status_available(monkeypatch):
    _patch_engines(monkeypatch)
    # 单例持真实注册表，测试中替换为桩以避免触达 DB
    monkeypatch.setattr(
        voice_service_status,
        "engine_registry",
        SimpleNamespace(resolve_default_engine=AsyncMock(return_value=_local_row("asr"))),
    )
    redis = FakeAsyncRedis(decode_responses=True)

    status = await voice_service_status.get_status(redis)

    assert status["asr"]["engineStatus"] == "online"
