"""AI 对话 SSE 推理失败收尾测试

验证 run_reasoning 后台任务的收尾契约：
- 推理异常时 SSE error 事件只推一份（失败态落库由推理侧用独立 session 完成），
  error 后补 message.end 保证客户端走统一完成处理
- error 载荷脱敏：业务异常透出业务码，未知异常统一 A0600，内部细节不外泄
- 后台任务不持有请求作用域 session（其生命周期超出请求）
"""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.sse import sse_emitter_manager as sse_module
from app.service.ai import message_streaming

_IDEM_KEY = "ai:msg:idempotent:2:key"


class _FakeReasoning:
    def __init__(
        self,
        exc: Exception | None = None,
        result: dict | None = None,
        before_return=None,
    ) -> None:
        self.exc = exc
        self.result = result or {}
        self.before_return = before_return

    async def run(self, **_kwargs) -> dict:
        if self.before_return is not None:
            await self.before_return()
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
def emitter(monkeypatch, mock_redis):
    """真实 SseEmitterManager + 假 Redis：断言落在 Redis 缓存上的事件序列"""
    manager = sse_module.SseEmitterManager()

    async def _get_redis_client():
        return mock_redis

    monkeypatch.setattr(sse_module, "get_redis_client", _get_redis_client)
    return manager, mock_redis


async def _cached_events(redis, stream_session_id: str = "stream-1") -> list[dict]:
    raw = await redis.lrange(f"ai:stream:{stream_session_id}", 0, -1)
    return [json.loads(r) for r in raw]


async def _run(
    emitter: tuple,
    exc: Exception | None = None,
    result: dict | None = None,
    before_return=None,
) -> None:
    manager, redis = emitter

    async def get_redis_client():
        return redis

    await message_streaming.run_reasoning(
        reasoning_service=_FakeReasoning(exc, result, before_return),
        get_redis_client=get_redis_client,
        sse_emitter_manager=manager,
        conv_id=1,
        user_id=2,
        model="qwen3-0.6b",
        assistant_msg_id=100,
        stream_session_id="stream-1",
        idem_key=_IDEM_KEY,
    )


async def test_unknown_exception_pushes_single_sanitized_error(emitter):
    await _run(emitter, exc=RuntimeError("连接 postgresql://u:p@10.0.0.1 失败"))

    events = await _cached_events(emitter[1])
    errors = [ev for ev in events if ev["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"] == {
        "code": ResultCode.AI_LLM_CALL_FAILED.code,
        "message": ResultCode.AI_LLM_CALL_FAILED.msg,
    }
    # 内部细节（连接串/异常文本）不随 error 事件外泄
    assert "postgresql" not in json.dumps(events, ensure_ascii=False)


async def test_business_exception_preserves_business_code(emitter):
    exc = BusinessException(ResultCode.OPERATION_NOT_ALLOW, "AI 对话积分已达上限需升级 VIP")
    await _run(emitter, exc=exc)

    errors = [ev for ev in await _cached_events(emitter[1]) if ev["event"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["code"] == ResultCode.OPERATION_NOT_ALLOW.code
    assert "积分" in errors[0]["data"]["message"]


async def test_error_is_followed_by_message_end(emitter):
    await _run(emitter, exc=RuntimeError("boom"))

    events = await _cached_events(emitter[1])
    types = [ev["event"] for ev in events]
    assert types.index("error") < types.index("message.end")
    end = next(ev for ev in events if ev["event"] == "message.end")
    assert end["data"]["stopReason"] == "error"
    assert end["data"]["usage"] == {
        "inputTokens": 0,
        "outputTokens": 0,
        "cachedInputTokens": 0,
        "credits": 0,
    }
    # stop_stream 收尾：终结事件与流关闭各一次
    assert types == ["error", "message.end", "message.end"]


async def test_failure_clears_idempotent_key(emitter):
    redis = emitter[1]
    await redis.set(_IDEM_KEY, "pending")

    await _run(emitter, exc=RuntimeError("boom"))

    assert await redis.get(_IDEM_KEY) is None


async def test_success_writes_idempotent_result(emitter):
    redis = emitter[1]

    await _run(emitter, result={"stop_reason": "stop"})

    assert json.loads(await redis.get(_IDEM_KEY)) == {"messageId": 100, "status": 2}


async def test_canceled_clears_idempotent_key(emitter):
    redis = emitter[1]
    await redis.set(_IDEM_KEY, "pending")

    await _run(emitter, result={"stop_reason": "canceled"})

    assert await redis.get(_IDEM_KEY) is None


async def test_stream_generator_closes_without_waiting_for_reasoning(emitter, monkeypatch):
    """空闲超时后生成器立即返回关闭连接，不得挂起等待后台推理（否则代理超时保护失效）。"""
    manager, redis = emitter
    monkeypatch.setattr(sse_module.settings, "AI_MESSAGE_HEARTBEAT_INTERVAL", 0.06)
    monkeypatch.setattr(sse_module.settings, "AI_MESSAGE_STREAM_TIMEOUT", 0.05)
    finished = asyncio.Event()

    async def _slow_reasoning(**_kwargs):
        await asyncio.sleep(1.5)
        finished.set()
        return {"stop_reason": "stop"}

    async def get_redis_client():
        return redis

    loop = asyncio.get_running_loop()
    started = loop.time()
    chunks = [
        chunk
        async for chunk in message_streaming.stream_generator(
            sse_emitter_manager=manager,
            reasoning_service=SimpleNamespace(run=_slow_reasoning),
            get_redis_client=get_redis_client,
            conv_id=1,
            user_id=2,
            model="qwen3-0.6b",
            assistant_msg_id=100,
            stream_session_id="stream-gen",
            idem_key=_IDEM_KEY,
        )
    ]
    elapsed = loop.time() - started

    assert [chunk for chunk in chunks if "message.start" in chunk]
    assert elapsed < 1
    # 推理仍在跑（连接已关闭，执行与连接解耦）
    assert finished.is_set() is False
    await asyncio.wait_for(finished.wait(), 3)


async def test_long_reasoning_keeps_pending_alive(emitter):
    """推理期间周期性续期 pending：初始 TTL 过短也不得中途过期（否则同 key 重复落库）。"""
    redis = emitter[1]
    await redis.set(_IDEM_KEY, "pending", ex=2)

    async def _slow_reasoning():
        await asyncio.sleep(2.5)

    task = asyncio.create_task(
        _run(emitter, result={"stop_reason": "stop"}, before_return=_slow_reasoning)
    )
    # 越过初始 TTL（2s）后仍须是本轮 pending
    await asyncio.sleep(2.2)
    assert await redis.get(_IDEM_KEY) == "pending:stream-1"
    assert await redis.ttl(_IDEM_KEY) > 0

    await task
    assert json.loads(await redis.get(_IDEM_KEY)) == {"messageId": 100, "status": 2}


async def test_completed_write_keeps_other_attempt_pending(emitter):
    """pending 已被新一轮抢占时，上一轮的成功结果不得覆盖（否则第二次发送被误判完成）。"""
    redis = emitter[1]
    await redis.set(_IDEM_KEY, "pending", ex=180)

    async def _taken_over_by_newer_attempt():
        await redis.set(_IDEM_KEY, "pending:newer-attempt", ex=180)

    await _run(emitter, result={"stop_reason": "stop"}, before_return=_taken_over_by_newer_attempt)

    assert await redis.get(_IDEM_KEY) == "pending:newer-attempt"


async def test_failure_release_keeps_other_attempt_pending(emitter):
    """同理：失败/取消的释放写入也不得删掉新一轮的 pending。"""
    redis = emitter[1]
    await redis.set(_IDEM_KEY, "pending", ex=180)

    async def _taken_over_by_newer_attempt():
        await redis.set(_IDEM_KEY, "pending:newer-attempt", ex=180)

    await _run(emitter, exc=RuntimeError("boom"), before_return=_taken_over_by_newer_attempt)

    assert await redis.get(_IDEM_KEY) == "pending:newer-attempt"


def test_run_reasoning_takes_no_request_session():
    """后台任务生命周期超出请求作用域，不得再接收请求级 session。"""
    assert "db" not in inspect.signature(message_streaming.run_reasoning).parameters
    assert "db" not in inspect.signature(message_streaming.stream_generator).parameters
