"""语音服务状态聚合单元测试（T-VS-060~T-VS-068，阶段 2 统一模型路由口径）。

对齐 SDK ServiceStatusVO（camelCase）：asr.concurrentSessions / engineStatus 等。
状态取自平台默认语音模型（sys_ai_model + sys_ai_provider）对应引擎：
经 `resolve_voice_model` 按 engine_type 独立解析（**不跨 asr/tts 互借**），再按供应商
`protocol_type` 经 `get_audio_client` 分派探测；未配置默认模型、解析失败或引擎不可达
时按 offline 上报，状态接口始终可用（T-VS-067）。
"""

import time
from types import SimpleNamespace

import pytest

from app.service.voice import voice_service_status as status_module
from app.service.voice.voice_service_status import (
    _CONCURRENT_KEY,
    VoiceServiceStatusService,
    voice_service_status,
)


def _fake_route(protocol_type="openai_audio"):
    return SimpleNamespace(
        model_id="sensevoice",
        model=SimpleNamespace(model_id="sensevoice", model_type="asr"),
        provider=SimpleNamespace(id=78, provider_code="local-voice", protocol_type=protocol_type),
    )


class _DbSessionCtx:
    """get_db_session 的异步上下文管理器替身（状态探测经它取 db 后即被路由桩接管）"""

    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc):
        return False


def _make_svc(monkeypatch, *, route=None, route_error=None, status_result=None, status_error=None):
    """构造状态服务并注入路由/分派/引擎探测桩。

    route_error 模拟「未配置默认模型/路由失败」；status_error 模拟引擎不可达，
    两者都应按 offline 上报且不抛异常。
    """

    async def _resolve(db, model_type):
        if route_error is not None:
            raise route_error
        return route if route is not None else _fake_route()

    class _Client:
        async def status(self, provider):
            if status_error is not None:
                raise status_error
            if status_result is not None:
                return status_result
            return {
                "engine_status": "online",
                "stream_model_loaded": True,
                "offline_model_loaded": True,
                "voice_model_loaded": True,
            }

    monkeypatch.setattr(status_module, "resolve_voice_model", _resolve)
    monkeypatch.setattr(status_module, "get_db_session", lambda: _DbSessionCtx())
    monkeypatch.setattr(status_module, "get_audio_client", lambda protocol_type: _Client())
    return VoiceServiceStatusService()


@pytest.mark.asyncio
async def test_status_online_reports_correct_fields_and_concurrency(monkeypatch, mock_redis):
    """在线：engineStatus/loaded 字段如实上报，并发数取 ZCARD"""
    await mock_redis.zadd(_CONCURRENT_KEY, {f"s{i}": time.time() for i in range(3)})
    svc = _make_svc(monkeypatch)

    data = await svc.get_status(mock_redis)

    assert data["asr"]["engineStatus"] == "online"
    assert data["asr"]["concurrentSessions"] == 3
    assert data["asr"]["maxConcurrentSessions"] == 50
    assert data["asr"]["streamModelLoaded"] is True
    assert data["asr"]["offlineModelLoaded"] is True
    assert data["tts"]["engineStatus"] == "online"
    assert data["tts"]["voiceModelLoaded"] is True


@pytest.mark.asyncio
async def test_status_offline_does_not_raise(monkeypatch, mock_redis):
    """引擎不可达：按 offline 上报，状态接口不抛异常（T-VS-067）"""
    svc = _make_svc(monkeypatch, status_error=RuntimeError("engine down"))

    data = await svc.get_status(mock_redis)

    assert data["asr"]["engineStatus"] == "offline"
    assert data["asr"]["streamModelLoaded"] is False
    assert data["tts"]["engineStatus"] == "offline"


@pytest.mark.asyncio
async def test_status_concurrent_sessions_matches_zcard(monkeypatch, mock_redis):
    """并发计数 = 剪枝过期会话后的 ZCARD（与 ASR 服务共用键）"""
    await mock_redis.zadd(_CONCURRENT_KEY, {"alive": time.time(), "expired": time.time() - 99999})
    svc = _make_svc(monkeypatch)

    data = await svc.get_status(mock_redis)

    assert data["asr"]["concurrentSessions"] == 1


@pytest.mark.asyncio
async def test_status_response_shape_matches_sdk_vo(monkeypatch, mock_redis):
    """响应字段名对齐 SDK ServiceStatusVO（camelCase）"""
    svc = _make_svc(monkeypatch)

    data = await svc.get_status(mock_redis)

    assert set(data.keys()) == {"asr", "tts"}
    assert set(data["asr"].keys()) == {
        "engineStatus",
        "concurrentSessions",
        "maxConcurrentSessions",
        "streamModelLoaded",
        "offlineModelLoaded",
    }
    assert set(data["tts"].keys()) == {"engineStatus", "voiceModelLoaded"}


@pytest.mark.asyncio
async def test_unconfigured_default_model_reports_offline(monkeypatch, mock_redis):
    """未配置默认语音模型（字典缺项/路由失败）：按 offline 上报"""
    from app.core.code import ResultCode
    from app.core.exceptions import BusinessException

    svc = _make_svc(
        monkeypatch,
        route_error=BusinessException(ResultCode.BUSINESS_ERROR, "平台未配置默认asr模型"),
    )

    data = await svc.get_status(mock_redis)

    assert data["asr"]["engineStatus"] == "offline"
    assert data["tts"]["engineStatus"] == "offline"


@pytest.mark.asyncio
async def test_asr_does_not_borrow_tts_route(monkeypatch, mock_redis):
    """asr 默认模型缺配时**不**借用 tts 侧路由（两端可属不同供应商）"""
    calls: list[str] = []

    async def _resolve(db, model_type):
        calls.append(model_type)
        if model_type == "asr":
            raise RuntimeError("asr 未配置")
        return _fake_route()

    class _Client:
        async def status(self, provider):
            return {"engine_status": "online"}

    monkeypatch.setattr(status_module, "resolve_voice_model", _resolve)
    monkeypatch.setattr(status_module, "get_db_session", lambda: _DbSessionCtx())
    monkeypatch.setattr(status_module, "get_audio_client", lambda protocol_type: _Client())
    svc = VoiceServiceStatusService()

    data = await svc.get_status(mock_redis)

    assert data["asr"]["engineStatus"] == "offline"
    assert data["tts"]["engineStatus"] == "online"
    assert calls == ["asr", "tts"]  # 两端各自解析，无互借


@pytest.mark.asyncio
async def test_singleton_get_status_available(monkeypatch, mock_redis):
    """模块单例可用（路由/分派桩注入后）"""
    _make_svc(monkeypatch)  # 注入桩（单例方法读模块级引用）

    data = await voice_service_status.get_status(mock_redis)

    assert data["asr"]["engineStatus"] in ("online", "offline")
