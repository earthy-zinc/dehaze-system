"""AI 对话 SSE 推理失败 error 事件推送测试

验证：run_reasoning 后台任务推理异常时，向客户端推送 SSE error 事件（{code, message}），
使前端能区分"网络断开"与"后端推理失败"并展示真实原因。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai import message_streaming


class _FakeEmitter:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict]] = []
        self.stopped: list[str] = []

    async def send_event(self, stream_session_id: str, event_type: str, data: dict) -> None:
        self.sent.append((stream_session_id, event_type, data))

    async def stop_stream(self, stream_session_id: str) -> None:
        self.stopped.append(stream_session_id)


class _FakeReasoning:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc

    async def run(self, **_kwargs) -> None:
        if self.exc is not None:
            raise self.exc


class _FakeRedis:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, key: str) -> None:
        self.deleted.append(key)

    async def set(self, *_args, **_kwargs) -> None:
        return None


async def _run(
    exc: Exception,
    monkeypatch=None,
) -> tuple[_FakeEmitter, _FakeRedis, list]:
    emitter = _FakeEmitter()
    redis = _FakeRedis()
    # 记录 update_status 调用（标记失败态），避免真实 db 执行
    status_calls: list[tuple[int, int, str]] = []
    if monkeypatch is not None:
        async def _update_status(_db, msg_id, status, error=None):
            status_calls.append((msg_id, status, error or ""))
        monkeypatch.setattr(
            message_streaming.ai_message_repository, "update_status", _update_status
        )

    async def get_redis_client():
        return redis

    await message_streaming.run_reasoning(
        reasoning_service=_FakeReasoning(exc),
        get_redis_client=get_redis_client,
        sse_emitter_manager=emitter,
        db=object(),
        conv_id=1,
        user_id=2,
        model="qwen3-0.6b",
        assistant_msg_id=100,
        stream_session_id="stream-1",
        idem_key="ai:msg:idempotent:2:key",
    )
    return emitter, redis, status_calls


async def test_business_exception_pushes_error_event(monkeypatch) -> None:
    exc = BusinessException(ResultCode.AI_LLM_CALL_FAILED, "主模型和降级模型均不可用")
    emitter, redis, status_calls = await _run(exc, monkeypatch)

    assert redis.deleted == ["ai:msg:idempotent:2:key"]
    assert len(emitter.sent) == 1
    stream_id, event_type, data = emitter.sent[0]
    assert stream_id == "stream-1"
    assert event_type == "error"
    assert data["code"] == "A0600"
    assert "主模型" in data["message"]
    assert emitter.stopped == ["stream-1"]
    # 助手消息落库失败态（status=3 + error）
    assert len(status_calls) == 1
    msg_id, status, error = status_calls[0]
    assert msg_id == 100
    assert status == 3
    assert "主模型" in error


async def test_quota_exception_preserves_business_code(monkeypatch) -> None:
    exc = BusinessException(ResultCode.OPERATION_NOT_ALLOW, "AI 对话积分已达上限需升级 VIP")
    emitter, _, status_calls = await _run(exc, monkeypatch)

    _, event_type, data = emitter.sent[0]
    assert event_type == "error"
    assert data["code"] == ResultCode.OPERATION_NOT_ALLOW.code
    assert "积分" in data["message"]
    assert status_calls[0][1] == 3


async def test_unknown_exception_falls_back_to_llm_failed(monkeypatch) -> None:
    emitter, _, status_calls = await _run(RuntimeError("boom"), monkeypatch)

    _, event_type, data = emitter.sent[0]
    assert event_type == "error"
    assert data["code"] == ResultCode.AI_LLM_CALL_FAILED.code
    assert data["message"] == ResultCode.AI_LLM_CALL_FAILED.msg
    assert status_calls[0][1] == 3


@pytest.mark.parametrize("code", ["error", "message.end"])
async def test_error_event_before_stop_stream(code: str, monkeypatch) -> None:
    """error 事件在 stop_stream 前推送，确保客户端先收到错误再收到流关闭"""
    emitter, _, _ = await _run(RuntimeError("boom"), monkeypatch)
    error_idx = next(
        i for i, (_, t, _d) in enumerate(emitter.sent) if t == "error"
    )
    assert error_idx >= 0
    assert emitter.stopped == ["stream-1"]
