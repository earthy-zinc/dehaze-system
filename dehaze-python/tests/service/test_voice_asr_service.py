"""ASR 语音识别服务单元测试（T-VS-001~T-VS-030）。

AsrService 为无状态编排类，依赖以模块级单例（voice_billing_service /
funasr_client / hotword_service）形式引用，方法以 redis / db 为入参。
测试按 05-python-test-rules：monkeypatch 模块级依赖桩，仅断言业务结果。
"""

import io
import json
import wave

import pytest
from fakeredis import FakeAsyncRedis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.voice.funasr_client import FunASRClientError
from app.service.voice import asr_service
from app.service.voice.asr_service import AsrService, _CONCURRENT_KEY, _SESSION_KEY


# ── 测试桩 ──────────────────────────────────────────────────────────────────


class FakeFunASRClient:
    """FunASR 客户端桩：记录调用、可预设返回值/异常。"""

    def __init__(self, offline_text="识别文本"):
        self._offline_text = offline_text
        self._offline_error = None
        self.registered_hotwords = None
        self.offline_calls = 0

    async def offline(self, audio, *, model=None):
        self.offline_calls += 1
        if self._offline_error is not None:
            raise self._offline_error
        return self._offline_text

    async def register_hotwords(self, words):
        self.registered_hotwords = list(words)

    def set_offline_error(self, exc):
        self._offline_error = exc


class StubBillingService:
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


class StubHotwordService:
    """热词桩：按 user_id 返回合并后的生效热词。"""

    def __init__(self, global_words, user_words):
        self._global = global_words
        self._user = user_words

    async def get_effective_words(self, db, user_id):
        return list(self._global) + list(self._user.get(user_id, []))


def _patch_deps(monkeypatch, *, funasr=None, billing=None, hotword=None):
    """构造注入 AsrService 的依赖桩（默认参数在 import 时即冻结，故走构造注入）。"""
    return AsrService(
        funasr_client=funasr or FakeFunASRClient(),
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

    session_id = await svc.create_stream_session(redis, None, 1001, None)

    assert session_id
    key = _SESSION_KEY.format(session_id=session_id)
    assert await redis.exists(key)
    data = json.loads(await redis.get(key))
    assert data["user_id"] == 1001
    assert data["status"] == "processing"
    assert data["model"] == "sensevoice"
    assert await redis.zcard(_CONCURRENT_KEY) == 1


@pytest.mark.asyncio
async def test_create_stream_session_increments_concurrent_counter(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)

    s1 = await svc.create_stream_session(redis, None, 2001, None)
    s2 = await svc.create_stream_session(redis, None, 2001, None)

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
        await svc.create_stream_session(redis, None, 3001, None)

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
        await svc.create_stream_session(redis, None, 4001, None)

    assert exc.value.code == ResultCode.QUOTA_INSUFFICIENT
    assert await redis.zcard(_CONCURRENT_KEY) == 0


# ── T-VS-006/007：热词注册（create 时合并全局+用户热词注册到 FunASR）──


@pytest.mark.asyncio
async def test_create_stream_session_registers_merged_hotwords_to_funasr(monkeypatch):
    funasr = FakeFunASRClient()
    hotword = StubHotwordService(
        global_words=["全局词A", "全局词B"], user_words={5001: ["用户词X"]}
    )
    _patch_deps(monkeypatch, funasr=funasr, hotword=hotword)
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, funasr=funasr, hotword=hotword)

    await svc.create_stream_session(redis, None, 5001, None)

    assert funasr.registered_hotwords is not None
    assert set(funasr.registered_hotwords) == {"全局词A", "全局词B", "用户词X"}


# ── T-VS-008/009：结果查询（命中返回文本、无效 sessionId 404）──


@pytest.mark.asyncio
async def test_get_result_returns_text_when_session_exists(monkeypatch):
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch)
    session_id = await svc.create_stream_session(redis, None, 6001, None)
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
    _patch_deps(monkeypatch, funasr=FakeFunASRClient(offline_text="离线识别结果"))
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, funasr=FakeFunASRClient(offline_text="离线识别结果"))

    result = await svc.offline_asr(redis, None, 7001, _make_wav_bytes(), None)

    assert result["text"] == "离线识别结果"
    assert result["sessionId"]


@pytest.mark.asyncio
async def test_offline_asr_rejects_unsupported_format(monkeypatch):
    _patch_deps(monkeypatch)
    redis = FakeAsyncRedis(decode_responses=True)
    svc = AsrService()

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, None, 7001, b"fake-mp3-bytes", None)

    assert exc.value.code == ResultCode.PARAM_ERROR


@pytest.mark.asyncio
async def test_offline_asr_fails_on_funasr_error(monkeypatch):
    funasr = FakeFunASRClient()
    funasr.set_offline_error(FunASRClientError("engine down"))
    _patch_deps(monkeypatch, funasr=funasr)
    redis = FakeAsyncRedis(decode_responses=True)
    svc = _patch_deps(monkeypatch, funasr=funasr)

    with pytest.raises(BusinessException) as exc:
        await svc.offline_asr(redis, None, 7001, _make_wav_bytes(), None)

    # FunASR 调用异常被包装为业务异常（降级为识别失败）
    assert exc.value.code == ResultCode.BUSINESS_ERROR
