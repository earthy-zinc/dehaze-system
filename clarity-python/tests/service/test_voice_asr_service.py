import asyncio
import io
import json
import math
import wave
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fakeredis import FakeAsyncRedis
from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.llm.client.openai_audio_client import EngineStreamClosed
from app.service.voice import asr_service as asr_service_module
from app.service.voice.asr_service import _CONCURRENT_KEY, _SESSION_KEY, AsrService
from app.service.voice.hotword_service import HotwordService
from app.service.voice.voice_billing_service import VoiceBillingService

# ── 测试桩 ──────────────────────────────────────────────────────────────────

_FAKE_MODEL_ID = "sensevoice"
_FAKE_PROVIDER_ID = 78


class FakeAudioClient:
    """语音协议适配器桩：记录调用参数（热词/会话键），可预设返回值/异常/流式会话。

    asr_service 经模块级 `get_audio_client` / `resolve_voice_model` 引用统一路由，
    测试经 monkeypatch 注入本桩；实扣透传的是统一 model_id 而非引擎编码。
    """

    def __init__(self, offline_text="识别文本", stream_session=None):
        self._offline_text = offline_text
        self._stream_session = stream_session
        self._offline_error = None
        self.offline_calls = 0
        self.offline_hotwords: list[str] | None = None
        self.stream_hotwords: list[str] | None = None
        self.stream_session_id: str | None = None

    async def transcribe(self, provider, model_id, audio_bytes, hotwords=None):
        self.offline_calls += 1
        self.offline_hotwords = list(hotwords or [])
        if self._offline_error is not None:
            raise self._offline_error
        return self._offline_text

    async def open_stream(self, provider, model_id, hotwords=None, session_id=None):
        assert self._stream_session is not None, "未向 FakeAudioClient 预设流式会话"
        self.stream_hotwords = list(hotwords or [])
        self.stream_session_id = session_id
        return self._stream_session

    def set_offline_error(self, exc):
        self._offline_error = exc


async def _fake_resolve_voice_model(db, model_type):
    """统一模型路由桩：返回固定 asr 模型与供应商。"""

    class _P:
        id = _FAKE_PROVIDER_ID
        provider_code = "local-voice"
        protocol_type = "openai_audio"

    class _M:
        model_id = _FAKE_MODEL_ID
        provider_id = _FAKE_PROVIDER_ID

    class _Route:
        model = _M()
        provider = _P()

        @property
        def model_id(self):
            return _FAKE_MODEL_ID

    return _Route()


class StubBillingService(VoiceBillingService):
    """计费桩：ensure_balance 可预设抛异常，charge_asr 记录每次实扣秒数与模型。

    单价口径（模型价格表换算）由 test_voice_billing_service 覆盖，此处仅保留
    编排所需的返回值。
    """

    def __init__(self, ensure_balance_raise=None):
        self._ensure_balance_raise = ensure_balance_raise
        self.ensure_balance_calls = 0
        self.charged_seconds: list[float] = []
        self.charged_models: list[str] = []

    async def ensure_balance(self, db, user_id, estimated_credits):
        self.ensure_balance_calls += 1
        if self._ensure_balance_raise is not None:
            raise self._ensure_balance_raise

    async def estimate_asr_credits(self, db, model_id, provider_id, audio_seconds: float) -> int:
        return math.ceil(audio_seconds)

    async def charge_asr(self, db, user_id, model_id, provider_id, audio_seconds: float):
        self.charged_seconds.append(audio_seconds)
        self.charged_models.append(model_id)
        return 1


class StubHotwordService(HotwordService):
    """热词桩：按 user_id 返回合并后的生效热词。"""

    def __init__(self, global_words, user_words):
        self._global = global_words
        self._user = user_words

    async def get_effective_words(self, db, user_id):
        return list(self._global) + list(self._user.get(user_id, []))


class _FakeRouteRepo:
    """_restore_route 的仓储桩：按 id/model_id 返回最小路由对象（避免真实查库）。"""

    async def get_by_model_id(self, db, model_id):
        return SimpleNamespace(model_id=_FAKE_MODEL_ID, model_type="asr")

    async def get_by_id(self, db, provider_id):
        return SimpleNamespace(
            id=_FAKE_PROVIDER_ID,
            provider_code="local-voice",
            protocol_type="openai_audio",
        )


def _patch_deps(monkeypatch, *, provider=None, billing=None, hotword=None) -> AsrService:
    """构造 AsrService 并注入依赖桩（引擎侧经统一路由桩接入）。"""
    client = provider or FakeAudioClient()
    monkeypatch.setattr(asr_service_module, "get_audio_client", lambda protocol_type: client)
    monkeypatch.setattr(asr_service_module, "resolve_voice_model", _fake_resolve_voice_model)
    monkeypatch.setattr(asr_service_module, "ai_model_repository", _FakeRouteRepo())
    monkeypatch.setattr(asr_service_module, "ai_provider_repository", _FakeRouteRepo())
    return AsrService(
        voice_billing_service=billing or StubBillingService(),
        hotword_service=hotword or StubHotwordService([], {}),
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


# ── T-VS-006/007：热词随请求下发（引擎侧无全局热词状态）──


@pytest.mark.asyncio
async def test_create_stream_session_does_not_touch_hotwords(monkeypatch):
    """创建会话只落会话元数据：热词在 WS 建连时随会话下发，引擎不保存全局热词"""
    provider = FakeAudioClient()
    hotword = StubHotwordService(global_words=["全局词"], user_words={5001: ["用户词"]})
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider, hotword=hotword)

    await svc.create_stream_session(redis, AsyncMock(spec=AsyncSession), 5001, None)

    assert provider.offline_hotwords is None
    assert provider.stream_hotwords is None


@pytest.mark.asyncio
async def test_offline_asr_passes_merged_hotwords_to_engine(monkeypatch):
    """离线识别：合并全局+用户热词随请求下发给引擎"""
    provider = FakeAudioClient()
    hotword = StubHotwordService(
        global_words=["全局词A", "全局词B"], user_words={5001: ["用户词X"]}
    )
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider, hotword=hotword)

    await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 5001, _make_wav_bytes(), None)

    assert set(provider.offline_hotwords or []) == {"全局词A", "全局词B", "用户词X"}


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
    _patch_deps(monkeypatch, provider=FakeAudioClient(offline_text="离线识别结果"))
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=FakeAudioClient(offline_text="离线识别结果"))

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
async def test_offline_asr_fails_on_engine_error(monkeypatch):
    provider = FakeAudioClient()
    provider.set_offline_error(RuntimeError("engine down"))
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 7001, _make_wav_bytes(), None)

    # 引擎调用异常被包装为业务异常（降级为识别失败）
    assert exc.value.code == ResultCode.BUSINESS_ERROR


@pytest.mark.asyncio
async def test_offline_asr_keeps_engine_mapped_error_code(monkeypatch):
    """引擎错误码已在 engine_client 映射为业务异常，服务层不再降级覆盖"""
    provider = FakeAudioClient()
    provider.set_offline_error(BusinessException(ResultCode.PARAM_ERROR, "仅支持 WAV/PCM 音频格式"))
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 7001, _make_wav_bytes(), None)

    assert exc.value.code == ResultCode.PARAM_ERROR


# ── 伪装音频文件对抗（超 3200 字节的非音频二进制不得进入推理并误扣费）──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "magic",
    [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff\xe0", b"%PDF-1.7\n", b"PK\x03\x04", b"ID3\x03", b"OggS"],
    ids=["png", "jpeg", "pdf", "zip", "mp3", "ogg"],
)
async def test_offline_asr_rejects_disguised_binary_file(monkeypatch, magic):
    provider = FakeAudioClient()
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
    provider = FakeAudioClient()
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, provider=provider)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, AsyncMock(spec=AsyncSession), 8001, b"\x00" * 4001, None)

    assert exc.value.code == ResultCode.PARAM_ERROR
    assert provider.offline_calls == 0


@pytest.mark.asyncio
async def test_offline_asr_accepts_raw_pcm_even_bytes(monkeypatch):
    """偶数字节长度的裸 PCM（≥0.1 秒）合法进入推理"""
    provider = FakeAudioClient(offline_text="裸流识别结果")
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


class _FakeDbSessionCtx:
    """get_db_session 桩：WS 场景无请求级事务，自建会话（此处以替身返回）"""

    async def __aenter__(self):
        return AsyncMock(spec=AsyncSession)

    async def __aexit__(self, *exc):
        return False


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
        json.dumps(
            {
                "user_id": 2002,
                "status": "processing",
                "text": "",
                "model": "",
                "model_id": _FAKE_MODEL_ID,
                "provider_id": _FAKE_PROVIDER_ID,
            }
        ),
    )
    await _seed_login_session(redis, "attacker-sid", 1001)
    ws = FakeWebSocket({"sessionId": "victim-session", "sid": "attacker-sid"})

    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "victim-session")

    assert ws.closed == 4001
    assert "无权" in ws.sent[0]["message"]


@pytest.mark.asyncio
async def test_ws_accepts_owner_with_valid_sid(monkeypatch):
    """归属一致的合法登录用户：通过鉴权进入双向代理（空闲超时后正常完成会话）

    建连时把合并热词与 sessionId（幂等键）随会话下发给引擎。
    """
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    monkeypatch.setattr(asr_service_module, "get_db_session", lambda: _FakeDbSessionCtx())
    await redis.set(
        _SESSION_KEY.format(session_id="my-session"),
        json.dumps(
            {
                "user_id": 1001,
                "status": "processing",
                "text": "",
                "model": "",
                "model_id": _FAKE_MODEL_ID,
                "provider_id": _FAKE_PROVIDER_ID,
            }
        ),
    )
    await _seed_login_session(redis, "owner-sid", 1001)

    class FakeStreamSession:
        """流式会话桩：下行直接推送最终结果"""

        def __init__(self):
            self.eos_calls = 0
            self.closed = False

        async def send_audio(self, chunk):
            pass

        async def send_eos(self):
            self.eos_calls += 1

        async def recv_messages(self):
            yield json.dumps({"text": "最终文本", "is_final": True})

        async def aclose(self):
            self.closed = True

    class HangingWebSocket(FakeWebSocket):
        """上行 receive 持续挂起，由空闲超时（调至 0.05s）触发正常结束路径"""

        async def receive(self):
            await asyncio.sleep(3600)

    from app.config import settings as app_settings

    engine_session = FakeStreamSession()
    provider = FakeAudioClient(stream_session=engine_session)
    monkeypatch.setattr(app_settings, "VOICE_ASR_WS_IDLE_TIMEOUT", 0.05)
    svc = _patch_deps(
        monkeypatch, provider=provider, hotword=StubHotwordService(["全局词"], {1001: ["用户词"]})
    )

    ws = HangingWebSocket({"sessionId": "my-session", "sid": "owner-sid"})
    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "my-session")

    assert ws.accepted
    assert ws.closed != 4001  # 未被拒绝
    # 热词与幂等键随会话下发；空闲超时走 EOS 收尾并关闭引擎会话
    assert set(provider.stream_hotwords or []) == {"全局词", "用户词"}
    assert provider.stream_session_id == "my-session"
    assert engine_session.eos_calls == 1
    assert engine_session.closed
    # 会话已完成并写入最终文本
    final = json.loads(await redis.get(_SESSION_KEY.format(session_id="my-session")))
    assert final["status"] == "completed"
    assert final["text"] == "最终文本"


@pytest.mark.asyncio
async def test_ws_client_disconnect_completes_session_normally(monkeypatch):
    """客户端断开（receive 返回 websocket.disconnect）属正常结束：

    按已收音频计费并置 completed、不抛异常。修复前会再次 receive 抛 RuntimeError
    落入兜底 except → _fail_session 把会话误判为 failed（SDK 侧表现为 onError）。
    """
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    monkeypatch.setattr(asr_service_module, "get_db_session", lambda: _FakeDbSessionCtx())
    await redis.set(
        _SESSION_KEY.format(session_id="disc-session"),
        json.dumps(
            {
                "user_id": 1001,
                "status": "processing",
                "text": "",
                "model": "",
                "model_id": _FAKE_MODEL_ID,
                "provider_id": _FAKE_PROVIDER_ID,
            }
        ),
    )
    await _seed_login_session(redis, "disc-sid", 1001)

    class FakeStreamSession:
        async def send_audio(self, chunk):
            pass

        async def send_eos(self):
            pass

        async def recv_messages(self):
            yield json.dumps({"text": "断开前文本", "is_final": True})

        async def aclose(self):
            pass

    class DisconnectingWebSocket(FakeWebSocket):
        """首帧即回 websocket.disconnect；再次 receive 按 Starlette 语义抛 RuntimeError"""

        def __init__(self, query_params=None):
            super().__init__(query_params)
            self.receive_calls = 0

        async def receive(self):
            self.receive_calls += 1
            if self.receive_calls == 1:
                return {"type": "websocket.disconnect", "code": 1000}
            raise RuntimeError('Cannot call "receive" once a disconnect message has been received.')

    billing = StubBillingService()
    svc = _patch_deps(
        monkeypatch, billing=billing, provider=FakeAudioClient(stream_session=FakeStreamSession())
    )

    ws = DisconnectingWebSocket({"sessionId": "disc-session", "sid": "disc-sid"})
    # 替身：FakeWebSocket 为结构型 WS 桩（记录 accept/send/close/receive），无法子类化真 WebSocket
    await svc.handle_stream_websocket(cast(WebSocket, ws), "disc-session")

    assert ws.receive_calls == 1  # 断开后不再上行读取
    session = json.loads(await redis.get(_SESSION_KEY.format(session_id="disc-session")))
    assert session["status"] == "completed"
    assert billing.charged_seconds == []  # 未收到音频：不产生计费，也不重复计秒


@pytest.mark.asyncio
async def test_ws_engine_disconnect_completes_with_received_bytes(monkeypatch):
    """引擎侧断开：按已收音频实扣一次并置 completed（不重复计秒、不误判失败）"""
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    monkeypatch.setattr(asr_service_module, "get_db_session", lambda: _FakeDbSessionCtx())
    await redis.set(
        _SESSION_KEY.format(session_id="eng-session"),
        json.dumps(
            {
                "user_id": 1001,
                "status": "processing",
                "text": "",
                "model": "",
                "model_id": _FAKE_MODEL_ID,
                "provider_id": _FAKE_PROVIDER_ID,
            }
        ),
    )
    await _seed_login_session(redis, "eng-sid", 1001)

    class DeadEngineSession:
        """引擎侧会话桩：首块音频即报连接断开（EngineStreamClosed）"""

        def __init__(self):
            self.eos_calls = 0
            self.closed = False

        async def send_audio(self, chunk):
            raise EngineStreamClosed("引擎侧流式会话已断开，音频未送达")

        async def send_eos(self):
            self.eos_calls += 1

        async def recv_messages(self):
            yield json.dumps({"text": "断开前文本", "is_final": True})

        async def aclose(self):
            self.closed = True

    class AudioWebSocket(FakeWebSocket):
        """首帧推送 1 秒音频，随后挂起等待收尾"""

        def __init__(self, query_params=None):
            super().__init__(query_params)
            self.receive_calls = 0

        async def receive(self):
            self.receive_calls += 1
            if self.receive_calls == 1:
                return {"type": "websocket.receive", "bytes": b"\x00\x00" * 16000}
            # 引擎已断开时不应再有上行读取：真出现则永久挂起，由 receive_calls 断言暴露
            await asyncio.sleep(3600)
            return None

    engine_session = DeadEngineSession()
    billing = StubBillingService()
    svc = _patch_deps(
        monkeypatch, billing=billing, provider=FakeAudioClient(stream_session=engine_session)
    )

    ws = AudioWebSocket({"sessionId": "eng-session", "sid": "eng-sid"})
    await svc.handle_stream_websocket(cast(WebSocket, ws), "eng-session")

    assert ws.receive_calls == 1  # 引擎断开后不再上行读取
    assert engine_session.eos_calls == 0  # 连接已断：不再发 EOS
    assert engine_session.closed
    assert billing.charged_seconds == [1]  # 1 秒音频按已收字节实扣一次
    assert billing.charged_models == ["sensevoice"]  # 统一 model_id 随实扣落账
    session = json.loads(await redis.get(_SESSION_KEY.format(session_id="eng-session")))
    assert session["status"] == "completed"
    assert session["text"] == "断开前文本"


# ── 引擎未起效（错误帧 / 未产出任何结果即断开）：前端 4001 + 会话 failed ──────


def _engine_downlink(*messages, error=None):
    """引擎下行桩：依次产出消息行，再按需抛出引擎错误（等价于引擎错误帧）"""

    async def _generate():
        for message in messages:
            yield json.dumps(message)
        if error is not None:
            raise error

    return _generate()


class StubStreamEngine:
    """流式引擎会话桩：按预设下行结果/异常产出，记录收尾调用"""

    def __init__(self, *messages, error=None):
        self._messages = messages
        self._error = error
        self.eos_calls = 0
        self.closed = False

    async def send_audio(self, chunk):
        pass

    async def send_eos(self):
        self.eos_calls += 1

    def recv_messages(self):
        return _engine_downlink(*self._messages, error=self._error)

    async def aclose(self):
        self.closed = True


async def _engine_failure_case(monkeypatch, engine_session, *, session_id, sid):
    """构造引擎失败用例：预置登录态与会话，返回 (redis, 计费桩, 服务实例)"""
    redis = FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(asr_service_module, "get_redis_client", AsyncMock(return_value=redis))
    monkeypatch.setattr(asr_service_module, "get_db_session", lambda: _FakeDbSessionCtx())
    await redis.set(
        _SESSION_KEY.format(session_id=session_id),
        json.dumps(
            {
                "user_id": 1001,
                "status": "processing",
                "text": "",
                "model": "",
                "model_id": _FAKE_MODEL_ID,
                "provider_id": _FAKE_PROVIDER_ID,
            }
        ),
    )
    await _seed_login_session(redis, sid, 1001)
    billing = StubBillingService()
    svc = _patch_deps(
        monkeypatch, billing=billing, provider=FakeAudioClient(stream_session=engine_session)
    )
    return redis, billing, svc


class _NoAudioWebSocket(FakeWebSocket):
    """未推送任何音频即断开的前端连接桩"""

    async def receive(self):
        return {"type": "websocket.disconnect", "code": 1000}


@pytest.mark.asyncio
async def test_ws_engine_error_frame_fails_session_and_closes_frontend_4001(monkeypatch):
    """引擎错误帧（如模型加载失败）：会话置 failed、前端以 4001 关闭，未收音频不扣费"""
    engine_session = StubStreamEngine(
        error=BusinessException(ResultCode.BUSINESS_ERROR, "加载 FunASR 模型失败")
    )
    redis, billing, svc = await _engine_failure_case(
        monkeypatch, engine_session, session_id="fail-session", sid="fail-sid"
    )
    ws = _NoAudioWebSocket({"sessionId": "fail-session", "sid": "fail-sid"})

    await svc.handle_stream_websocket(cast(WebSocket, ws), "fail-session")

    assert ws.accepted
    assert ws.closed == 4001
    assert engine_session.closed
    assert billing.charged_seconds == []  # 未收音频：不产生计费
    session = json.loads(await redis.get(_SESSION_KEY.format(session_id="fail-session")))
    assert session["status"] == "failed"


@pytest.mark.asyncio
async def test_ws_engine_closed_without_result_fails_session(monkeypatch):
    """引擎建连后未产出任何结果即断开（无错误帧）：同样置 failed 并以 4001 收尾"""
    engine_session = StubStreamEngine()
    redis, billing, svc = await _engine_failure_case(
        monkeypatch, engine_session, session_id="empty-session", sid="empty-sid"
    )
    ws = _NoAudioWebSocket({"sessionId": "empty-session", "sid": "empty-sid"})

    await svc.handle_stream_websocket(cast(WebSocket, ws), "empty-session")

    assert ws.closed == 4001
    assert billing.charged_seconds == []
    session = json.loads(await redis.get(_SESSION_KEY.format(session_id="empty-session")))
    assert session["status"] == "failed"


@pytest.mark.asyncio
async def test_ws_engine_error_after_audio_keeps_billing_invariant(monkeypatch):
    """引擎中途出错：会话 failed，但已收音频仍按计费不变式实扣一次"""
    engine_session = StubStreamEngine(
        {"text": "半句", "is_final": False},
        error=BusinessException(ResultCode.BUSINESS_ERROR, "推理失败"),
    )
    redis, billing, svc = await _engine_failure_case(
        monkeypatch, engine_session, session_id="mid-session", sid="mid-sid"
    )

    class OneSecondWebSocket(FakeWebSocket):
        """推送 1 秒音频后发 EOS"""

        def __init__(self, query_params=None):
            super().__init__(query_params)
            self.receive_calls = 0

        async def receive(self):
            self.receive_calls += 1
            if self.receive_calls == 1:
                return {"type": "websocket.receive", "bytes": b"\x00\x00" * 16000}
            return {"type": "websocket.receive", "text": "EOS"}

    ws = OneSecondWebSocket({"sessionId": "mid-session", "sid": "mid-sid"})

    await svc.handle_stream_websocket(cast(WebSocket, ws), "mid-session")

    assert ws.closed == 4001
    assert billing.charged_seconds == [1]
    assert billing.charged_models == ["sensevoice"]
    session = json.loads(await redis.get(_SESSION_KEY.format(session_id="mid-session")))
    assert session["status"] == "failed"
