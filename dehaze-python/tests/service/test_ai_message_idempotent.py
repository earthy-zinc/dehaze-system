import json
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai_message_service import AiMessageService
from tests.stubs.factories import fake_redis


def _conv():
    return SimpleNamespace(
        id=1,
        status=1,
        model_config={},
        model=None,
        agent_code=None,
        message_count=2,
        title="标题",
        title_source="auto",
        current_branch_message_id=1,
    )


def _make_service(redis, *, message_get_by_id=None, acquire_lock=None):
    class _ConvRepo:
        async def get_by_id_and_user(self, db, conv_id, user_id):
            return _conv()

    class _MsgRepo:
        async def get_by_id(self, db, msg_id):
            if message_get_by_id is None:
                return None
            return message_get_by_id(db, msg_id)

    async def _no_lock(conv_id):
        return False

    return AiMessageService(
        ai_conversation_repository=_ConvRepo(),
        ai_message_repository=_MsgRepo(),
        get_redis_client=lambda: redis,
        sse_emitter_manager=SimpleNamespace(acquire_lock=acquire_lock or _no_lock),
    )


async def _call(svc, redis, *, user_id=7, key="key-1"):
    return await svc.send_message(
        object(),
        conv_id=1,
        user_id=user_id,
        form=SimpleNamespace(content="你好", model=None),
        idempotency_key=key,
    )


async def test_pending_hit_raises_conflict(mock_redis):
    await mock_redis.set("ai:msg:idempotent:7:key-1", "pending")
    svc = _make_service(mock_redis)
    with pytest.raises(BusinessException) as exc:
        await _call(svc, mock_redis)
    assert exc.value.code == ResultCode.REPEAT_SUBMIT_ERROR


async def test_completed_hit_returns_existing(mock_redis):
    await mock_redis.set(
        "ai:msg:idempotent:7:key-1", json.dumps({"messageId": 99, "status": 2})
    )

    def get_by_id(db, msg_id):
        return SimpleNamespace(
            id=99,
            conversation_id=1,
            role="assistant",
            content="已有回复",
            model="gpt-4o",
            status=2,
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=0,
            credits=3,
            error=None,
            deleted=0,
            task_id=None,
            edited=0,
            original_content=None,
            tool_calls=None,
            metadata_=None,
            create_time=None,
            update_time=None,
        )

    svc = _make_service(mock_redis, message_get_by_id=get_by_id)
    resp = await _call(svc, mock_redis)
    assert isinstance(resp, JSONResponse)
    assert "已有回复" in resp.body.decode()


async def test_miss_sets_pending_with_ttl_aligned(mock_redis):
    svc = _make_service(mock_redis)
    with pytest.raises(BusinessException):
        await _call(svc, mock_redis)
    key = "ai:msg:idempotent:7:key-1"
    assert await mock_redis.get(key) == "pending"
    assert await mock_redis.ttl(key) == settings.AI_MESSAGE_STREAM_TIMEOUT + 60


async def test_idempotency_key_isolated_by_user(mock_redis):
    await mock_redis.set(
        "ai:msg:idempotent:7:key-1", json.dumps({"messageId": 99, "status": 2})
    )
    svc = _make_service(mock_redis)
    with pytest.raises(BusinessException):
        await _call(svc, mock_redis, user_id=8)
    assert await mock_redis.get("ai:msg:idempotent:7:key-1") == json.dumps(
        {"messageId": 99, "status": 2}
    )
    assert await mock_redis.get("ai:msg:idempotent:8:key-1") == "pending"


async def test_failure_clears_key_for_retry():
    redis = await fake_redis()
    deleted = []

    async def _delete(*keys):
        deleted.extend(keys)
        return len(keys)

    redis.delete = _delete

    async def _run_fail(**kwargs):
        raise RuntimeError("推理失败")

    async def _noop_stop(stream_session_id):
        return None

    svc = AiMessageService(
        get_redis_client=lambda: redis,
        reasoning_service=SimpleNamespace(run=_run_fail),
        sse_emitter_manager=SimpleNamespace(stop_stream=_noop_stop),
    )

    await svc._run_reasoning(1, 7, "gpt", 1, "s1", "k")
    assert deleted == ["k"]
