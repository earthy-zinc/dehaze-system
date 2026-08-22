from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service import ai_message_service as m
from tests.stubs import fake_redis


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


@pytest.fixture
async def env(monkeypatch, mock_redis):
    async def _get_conv(db, conv_id, user_id):
        return _conv()

    monkeypatch.setattr(m, "get_redis_client", lambda: mock_redis)
    monkeypatch.setattr(m.ai_conversation_repository, "get_by_id_and_user", _get_conv)
    return mock_redis


async def _call(env):
    return await m.AiMessageService.send_message(
        object(),
        conv_id=1,
        user_id=7,
        form=SimpleNamespace(content="你好", model=None),
        idempotency_key="key-1",
    )


async def _acquire_lock(conv_id):
    return False


async def test_pending_hit_raises_conflict(env):
    await env.set("ai:msg:idempotent:7:key-1", "pending")
    with pytest.raises(BusinessException) as exc:
        await _call(env)
    assert exc.value.code == ResultCode.REPEAT_SUBMIT_ERROR


async def test_completed_hit_returns_existing(env, monkeypatch):
    import json

    await env.set("ai:msg:idempotent:7:key-1", json.dumps({"messageId": 99, "status": 2}))

    async def get_by_id(db, msg_id):
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

    monkeypatch.setattr(m.ai_message_repository, "get_by_id", get_by_id)
    resp = await _call(env)
    assert isinstance(resp, JSONResponse)
    body = resp.body.decode()
    assert "已有回复" in body


async def test_miss_sets_pending_with_ttl_aligned(env, monkeypatch):
    monkeypatch.setattr(m.sse_emitter_manager, "acquire_lock", _acquire_lock)
    with pytest.raises(BusinessException):
        await _call(env)
    key = "ai:msg:idempotent:7:key-1"
    assert await env.get(key) == "pending"
    assert await env.ttl(key) == settings.AI_MESSAGE_STREAM_TIMEOUT + 60


async def test_idempotency_key_isolated_by_user(env, monkeypatch):
    import json

    await env.set("ai:msg:idempotent:7:key-1", json.dumps({"messageId": 99, "status": 2}))
    monkeypatch.setattr(m.ai_message_repository, "get_by_id", lambda db, mid: None)
    monkeypatch.setattr(m.sse_emitter_manager, "acquire_lock", _acquire_lock)
    with pytest.raises(BusinessException):
        await m.AiMessageService.send_message(
            object(),
            conv_id=1,
            user_id=8,
            form=SimpleNamespace(content="你好", model=None),
            idempotency_key="key-1",
        )
    assert await env.get("ai:msg:idempotent:7:key-1") == json.dumps({"messageId": 99, "status": 2})
    assert await env.get("ai:msg:idempotent:8:key-1") == "pending"


async def test_failure_clears_key_for_retry(monkeypatch):
    redis = await fake_redis()
    deleted = []

    async def _delete(*keys):
        deleted.extend(keys)
        return len(keys)

    redis.delete = _delete

    async def _get_redis():
        return redis

    async def _run_fail(**kwargs):
        raise RuntimeError("推理失败")

    async def _noop_stop(stream_session_id):
        return None

    monkeypatch.setattr(m, "get_redis_client", _get_redis)
    monkeypatch.setattr(m.reasoning_service, "run", _run_fail)
    monkeypatch.setattr(m.sse_emitter_manager, "stop_stream", _noop_stop)

    await m._run_reasoning(1, 7, "gpt", 1, "s1", "k")
    assert deleted == ["k"]
