"""ASR 语音识别服务单元测试（T-VS-001~T-VS-030）。

AsrService 为无状态编排类，依赖以模块级单例（voice_billing_service /
funasr_client / hotword_service）形式引用，方法以 redis / db 为入参。
测试按 05-python-test-rules：monkeypatch 模块级依赖桩，仅断言业务结果。
"""

import asyncio
import io
import json
import wave
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis
from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.voice.funasr_client import FunASRClientError
from app.infrastructure.voice.provider.base import ASRProvider
from app.infrastructure.voice.provider.registry import VoiceEngineRegistry
from app.service.voice import asr_service as asr_service_module
from app.service.voice.asr_service import _CONCURRENT_KEY, _SESSION_KEY, AsrService
from app.service.voice.hotword_service import HotwordService
from app.service.voice.voice_billing_service import VoiceBillingService

# ── 测试桩 ──────────────────────────────────────────────────────────────────


class FakeAsrProvider(ASRProvider):
    """ASR Provider 桩：记录调用、可预设返回值/异常/流式会话。"""

    def __init__(self, offline_text="识别文本", stream_session=None):
        self._offline_text = offline_text
        self._stream_session = stream_session
        self._offline_error = None
        self.registered_hotwords = None
        self.offline_calls = 0

    async def recognize_offline(self, audio_bytes):
        self.offline_calls += 1
        if self._offline_error is not None:
            raise self._offline_error
        return self._offline_text

    async def recognize_stream(self):
        assert self._stream_session is not None, "未向 FakeAsrProvider 预设流式会话"
        return self._stream_session

    async def register_hotwords(self, words):
        self.registered_hotwords = list(words)

    async def engine_status(self):
        return {"status": "ok"}

    def set_offline_error(self, exc):
        self._offline_error = exc


class FakeEngineRegistry(VoiceEngineRegistry):
    """注册表桩：get_asr_provider 返回预设的 Provider。"""

    def __init__(self, provider):
        self._provider = provider

    async def get_asr_provider(self):
        return self._provider


class StubBillingService(VoiceBillingService):
    """计费桩：ensure_balance 可预设抛异常。"""

    def __init__(self, ensure_balance_raise=None):
        self._ensure_balance_raise = ensure_balance_raise
        self.ensure_balance_calls = 0

    async def ensure_balance(self, db, user_id, estimated_credits):
        self.ensure_balance_calls += 1
        if self._ensure_balance_raise is not None:
            raise self._ensure_balance_raise

    async def charge_asr(self, db, user_id, audio_seconds):
        return 1


class StubHotwordService(HotwordService):
    """热词桩：按 user_id 返回合并后的生效热词。"""

    def __init__(self, global_words, user_words):
        self._global = global_words
        self._user = user_words

    async def get_effective_words(self, db, user_id):
        return list(self._global) + list(self._user.get(user_id, []))


def _patch_deps(monkeypatch, *, provider=None, billing=None, hotword=None) -> AsrService:
    """构造注入 AsrService 的依赖桩（Provider 经引擎注册表桩注入）。"""
    return AsrService(
        voice_billing_service=billing or StubBillingService(),
        hotword_service=hotword or StubHotwordService([], {}),
        engine_registry=FakeEngineRegistry(provider or FakeAsrProvider()),
    )


def _make_wav_bytes(duration_sec=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * duration_sec))
    return buf.getvalue()


# ── T-VS-001/002：流式会话创建（返回 sessionId、Redis 写入、并发计数递增）──


@pytest.mark.asyncio
async def test_create_stream_session_returns_session_id_and_writes_redis(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)

    session_id = await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 1001, None)

    assert session_id
    key = _SESSION_KEY.format(session_id=session_id)
    assert await redis.exists(key)
    data = json.loads(await redis.get(key))
    assert data["user_id"] == 1001
    assert data["status"] == "processing"
    assert data["model"] == ""
    assert await redis.zcard(_CONCURRENT_KEY) == 1


@pytest.mark.asyncio
async def test_create_stream_session_increments_concurrent_counter(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)

    s1 = await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 2001, None)
    s2 = await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 2001, None)

    assert s1 != s2
    assert await redis.zcard(_CONCURRENT_KEY) == 2


# ── T-VS-003/004：并发上限（超限拒绝）──


@pytest.mark.asyncio
async def test_create_stream_session_rejects_when_concurrent_limit_exceeded(monkeypatch):
    import time

    from app.config import settings as app_settings

    _patch_deps(monkeypatch)
    monkeypatch.setattr(app_settings, "VOICE_ASR_MAX_CONCURRENT_SESSIONS", 2)
    redis = FakeAsyncRedis(decode_responses=True)
    now = time.time()
    for i in range(3):
        await redis.zadd(_CONCURRENT_KEY, {f"sess-{i}": now - i})
    svc = _patch_deps(monkeypatch)

    with pytest.raises(BusinessException) as exc:
        await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 3001, None)

    assert exc.value.code == ResultCode.BUSINESS_ERROR
    # 拒绝时不新增并发计数（仍保留预置的 3 个活动会话）
    assert await redis.zcard(_CONCURRENT_KEY) == 3


# ── T-VS-005：计费预校验（余额不足拒绝）──


@pytest.mark.asyncio
async def test_create_stream_session_rejects_when_balance_insufficient(monkeypatch):
    billing = StubBillingService(
        ensure_balance_raise=BusinessException(ResultCode.QUOTA_INSUFFICIENT, "余额不足")
    )
    _patch_deps(monkeypatch, billing=billing)
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, billing=billing)

    with pytest.raises(BusinessException) as exc:
        await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 4001, None)

    assert exc.value.code == ResultCode.QUOTA_INSUFFICIENT
    assert await redis.zcard(_CONCURRENT_KEY) == 0


# ── T-VS-006/007：热词注册（create 时合并全局+用户热词注册到 FunASR）──


@pytest.mark.asyncio
async def test_create_stream_session_registers_merged_hotwords_to_funasr(monkeypatch):
    funasr = FakeAsrProvider()
    hotword = StubHotwordService(
        global_words=["全局词A", "全局词B"], user_words={5001: ["用户词X"]}
    )
    _patch_deps(monkeypatch, provider=funasr, hotword=hotword)
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=funasr, hotword=hotword)

    await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 5001, None)

    assert funasr.registered_hotwords is not None
    assert set(funasr.registered_hotwords) == {"全局词A", "全局词B", "用户词X"}


# ── T-VS-008/009：结果查询（命中返回文本、无效 sessionId 404）──


@pytest.mark.asyncio
async def test_get_result_returns_text_when_session_exists(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)
    session_id = await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 6001, None)
    await redis.set(
        _SESSION_KEY.format(session_id=session_id),
        json.dumps({"user_id": 6001, "status": "completed", "text": "你好世界"}),
    )

    result = await svc.get_result(redis, session_id, 6001)

    assert result["status"] == "completed"
    assert result["text"] == "你好世界"
    assert result["sessionId"] == session_id


@pytest.mark.asyncio
async def test_get_result_raises_for_invalid_session(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)

    with pytest.raises(BusinessException) as exc:
        await svc.get_result(redis, "not-exist", 6001)

    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


# ── T-VS-010/011/012：离线识别（正常识别、格式错误、失败降级）──


@pytest.mark.asyncio
async def test_offline_asr_returns_text_for_valid_wav(monkeypatch):
    _patch_deps(monkeypatch, provider=FakeAsrProvider(offline_text="离线识别结果"))
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=FakeAsrProvider(offline_text="离线识别结果"))

    result = await svc.offline_asr(
        redis, AsyncMock(spec=AsyncSession), 7001, _make_wav_bytes(), None
    )

    assert result["text"] == "离线识别结果"
    assert result["sessionId"]


@pytest.mark.asyncio
async def test_offline_asr_rejects_unsupported_format(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 7001, b"fake-mp3-bytes", None)

    assert exc.value.code == ResultCode.PARAM_ERROR


@pytest.mark.asyncio
async def test_offline_asr_fails_on_funasr_error(monkeypatch):
    funasr = FakeAsrProvider()
    funasr.set_offline_error(FunASRClientError("engine down"))
    _patch_deps(monkeypatch, provider=funasr)
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=funasr)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 7001, _make_wav_bytes(), None)

    # FunASR 调用异常被包装为业务异常（降级为识别失败）
    assert exc.value.code == ResultCode.BUSINESS_ERROR


# ── 伪装音频文件对抗（超 3200 字节的非音频二进制不得进入推理并误扣费）──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "magic",
    [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff\xe0", b"%PDF-1.7\n", b"PK\x03\x04", b"ID3\x03", b"OggS"],
    ids=["png", "jpeg", "pdf", "zip", "mp3", "ogg"],
)
async def test_offline_asr_rejects_disguised_binary_file(monkeypatch, magic):
    provider = FakeAsrProvider()
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider)

    disguised = magic + b"\x00" * 4000  # 超过 3200 字节下限，伪装为可通行体积
    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 8001, disguised, None)

    assert exc.value.code == ResultCode.PARAM_ERROR
    assert provider.offline_calls == 0  # 未进入推理


@pytest.mark.asyncio
async def test_offline_asr_rejects_odd_length_raw_pcm(monkeypatch):
    """裸 PCM 字节数为奇数（非 16bit 采样对齐）→ 参数错误"""
    provider = FakeAsrProvider()
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 8001, b"\x00" * 4001, None)

    assert exc.value.code == ResultCode.PARAM_ERROR
    assert provider.offline_calls == 0


@pytest.mark.asyncio
async def test_offline_asr_accepts_raw_pcm_even_bytes(monkeypatch):
    """偶数字节长度的裸 PCM（≥0.1 秒）合法进入推理"""
    provider = FakeAsrProvider(offline_text="裸流识别结果")
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider)

    result = await svc.offline_asr(
        redis, AsyncMock(spec=AsyncSession), 8001, b"\x00\x00" * 16000, None
    )

    assert result["text"] == "裸流识别结果"
    assert provider.offline_calls == 1


# ── 流式 WebSocket 建连鉴权（后端实现 §6.1：校验登录态 + 会话归属）──


class FakeWebSocket:
    """WebSocket 桩：记录 accept/send/close，query_params 模拟握手 URL 参数。"""

    def __init__(self, query_params=None):
        self.query_params = query_params or {}
        self.accepted = False
        self.sent: list = []
        self.closed: int | None = None

    async def accept(self):
        self.accepted = True

    async def send_json(self, msg):
        self.sent.append(msg)

    async def close(self, code=None):
        self.closed = code


def _seed_login_session(redis, sid, user_id):
    import json as _json

    return redis.set(
        f"session:{sid}",
        _json.dumps({"userId": user_id, "username": "tester"}),
    )


@pytest.mark.asyncio
async def test_ws_rejects_when_sid_missing(monkeypatch):
    """未携带会话凭证（sid）→ 拒绝建连（4001），不进入识别流程"""
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    svc = _patch_deps(monkeypatch)
    ws = FakeWebSocket({"sessionId": "s1"})

    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "s1")

    assert ws.closed == 4001
    assert ws.sent
    assert ws.sent[0]["type"] == "error"


@pytest.mark.asyncio
async def test_ws_rejects_when_sid_expired(monkeypatch):
    """sid 对应登录会话已过期 → 拒绝建连"""
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    svc = _patch_deps(monkeypatch)
    ws = FakeWebSocket({"sessionId": "s1", "sid": "expired-sid"})

    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "s1")

    assert ws.closed == 4001


@pytest.mark.asyncio
async def test_ws_rejects_when_session_owned_by_other_user(monkeypatch):
    """登录用户与 ASR 会话归属不一致（拿他人 sessionId 建连）→ 拒绝建连"""
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    svc = _patch_deps(monkeypatch)
    await redis.set(
        _SESSION_KEY.format(session_id="victim-session"),
        json.dumps({"user_id": 2002, "status": "processing", "text": "", "model": ""}),
    )
    await _seed_login_session(redis, "attacker-sid", 1001)
    ws = FakeWebSocket({"sessionId": "victim-session", "sid": "attacker-sid"})

    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "victim-session")

    assert ws.closed == 4001
    assert "无权" in ws.sent[0]["message"]


@pytest.mark.asyncio
async def test_ws_accepts_owner_with_valid_sid(monkeypatch):
    """归属一致的合法登录用户：通过鉴权进入双向代理（空闲超时后正常完成会话）"""
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    await redis.set(
        _SESSION_KEY.format(session_id="my-session"),
        json.dumps({"user_id": 1001, "status": "processing", "text": "", "model": ""}),
    )
    await _seed_login_session(redis, "owner-sid", 1001)

    class FakeStreamSession:
        """流式会话桩：下行直接推送最终结果"""

        async def send_audio(self, chunk):
            pass

        async def send_eos(self):
            pass

        async def recv_messages(self):
            yield json.dumps({"text": "最终文本", "is_final": True})

    class OwnerRegistry(FakeEngineRegistry):
        async def get_asr_provider(self):
            return FakeAsrProvider(stream_session=FakeStreamSession())

    class HangingWebSocket(FakeWebSocket):
        """上行 receive 持续挂起，由空闲超时（调至 0.05s）触发正常结束路径"""

        async def receive(self):
            await asyncio.sleep(3600)

    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "VOICE_ASR_WS_IDLE_TIMEOUT", 0.05)
    svc = AsrService(
        voice_billing_service=StubBillingService(),
        hotword_service=StubHotwordService([], {}),
        engine_registry=OwnerRegistry(FakeAsrProvider()),
    )

    ws = HangingWebSocket({"sessionId": "my-session", "sid": "owner-sid"})
    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "my-session")

    assert ws.accepted
    assert ws.closed != 4001  # 未被拒绝
    # 会话已完成并写入最终文本
    final = json.loads(await redis.get(_SESSION_KEY.format(session_id="my-session")))
    assert final["status"] == "completed"
    assert final["text"] == "最终文本"
