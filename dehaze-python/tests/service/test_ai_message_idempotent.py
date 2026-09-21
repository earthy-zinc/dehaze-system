import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.sse.sse_emitter_manager import SseEmitterManager
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_model import SysAiModel
from app.repository.ai_agent_repository import AiAgentRepository
from app.repository.ai_conversation_repository import AiConversationRepository
from app.repository.ai_message_repository import AiMessageRepository
from app.repository.ai_model_repository import AiModelRepository
from app.service.ai.middleware.interrupt_handler import InterruptHandler
from app.service.ai.service.reasoning_service import ReasoningService
from app.service.ai_message_service import AiMessageService
from app.service.ai_model_service import AiModelService
from tests.stubs.factories import fake_redis


def _conv():
    return SysAiConversation(
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


class _ConvRepo(AiConversationRepository):
    """测试替身：仅覆写 get_by_id_and_user/update_last_message。"""

    async def get_by_id_and_user(self, db, conv_id, user_id):
        return _conv()

    async def update_last_message(self, db, *args, **kwargs):
        return None


class _MsgRepo(AiMessageRepository):
    """测试替身：仅覆写 get_by_id/create（create 落自增 id 并记录，供并发用例统计落库条数）。"""

    def __init__(self, message_get_by_id=None):
        self.next_id = 100
        self.created = []
        self._message_get_by_id = message_get_by_id

    async def get_by_id(self, db, id, *, with_deleted=False):
        if self._message_get_by_id is None:
            return None
        return self._message_get_by_id(db, id)

    async def create(self, db, entity):
        entity.id = self.next_id
        self.next_id += 1
        self.created.append(entity)
        return entity


class _SseManager(SseEmitterManager):
    """测试替身：仅覆写 acquire_lock/stop_stream/send_error。"""

    def __init__(self, acquire_lock=None):
        self._acquire_lock = acquire_lock

    async def acquire_lock(self, conversation_id) -> bool:
        if self._acquire_lock is not None:
            return await self._acquire_lock(conversation_id)
        return False

    async def stop_stream(self, stream_session_id) -> None:
        return None

    async def send_error(self, stream_session_id, error) -> None:
        return None


class _AgentRepo(AiAgentRepository):
    """测试替身：仅覆写 get_by_code。"""

    async def get_by_code(self, db, *args, **kwargs):
        return None


class _ModelRepo(AiModelRepository):
    """测试替身：仅覆写 get_by_model_id（返回启用中的模型）。"""

    async def get_by_model_id(self, db, model_id):
        return SysAiModel(status=1)


class _ModelSvc(AiModelService):
    """测试替身：仅覆写 validate_model_caps。"""

    async def validate_model_caps(self, *args, **kwargs):
        return None


class _InterruptHandler(InterruptHandler):
    """测试替身：仅覆写 get_interrupt（恒无中断）。"""

    async def get_interrupt(self, thread_id):
        return None


class _Reasoning(ReasoningService):
    """测试替身：仅覆写 run（行为由用例注入）。"""

    def __init__(self, run):
        self._run = run

    async def run(self, *args, **kwargs):
        return await self._run(*args, **kwargs)


def _body_text(resp) -> str:
    """JSONResponse.body 运行时恒为 bytes；starlette Response.render 声明为
    bytes | memoryview，此处收敛为 bytes 再做解码断言。"""
    assert isinstance(resp.body, bytes)
    return resp.body.decode()


def _make_service(redis, *, message_get_by_id=None, acquire_lock=None):
    async def _no_lock(conv_id):
        return False

    return AiMessageService(
        ai_conversation_repository=_ConvRepo(),
        ai_message_repository=_MsgRepo(message_get_by_id),
        get_redis_client=lambda: redis,
        sse_emitter_manager=_SseManager(acquire_lock or _no_lock),
    )


async def _acquire_ok(conv_id):
    return True


def _make_full_service(redis):
    """完整发送链路桩：中断检查/模型校验/会话与消息写入全部放行"""
    msg_repo = _MsgRepo()
    svc = AiMessageService(
        ai_conversation_repository=_ConvRepo(),
        ai_message_repository=msg_repo,
        ai_agent_repository=_AgentRepo(),
        ai_model_repository=_ModelRepo(),
        ai_model_service=_ModelSvc(),
        get_redis_client=lambda: redis,
        sse_emitter_manager=_SseManager(_acquire_ok),
        interrupt_handler=_InterruptHandler(),
    )
    return msg_repo, svc


async def _call(svc, redis, *, user_id=7, key="key-1"):
    return await svc.send_message(
        AsyncMock(),
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


async def test_pending_taken_over_by_stream_raises_conflict(mock_redis):
    """推理进行中 run_reasoning 会用本轮标识接管该键，仍须判冲突而非当完成态重放"""
    await mock_redis.set("ai:msg:idempotent:7:key-1", "pending:s-1")
    svc = _make_service(mock_redis)
    with pytest.raises(BusinessException) as exc:
        await _call(svc, mock_redis)
    assert exc.value.code == ResultCode.REPEAT_SUBMIT_ERROR


async def test_completed_hit_returns_existing(mock_redis):
    await mock_redis.set("ai:msg:idempotent:7:key-1", json.dumps({"messageId": 99, "status": 2}))

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
    assert "已有回复" in _body_text(resp)


async def test_lock_conflict_clears_pending_key(mock_redis):
    """抢流式锁失败：pending 必须一并回收，否则 TTL 内同 key 重试全被误判为重复提交"""
    svc = _make_service(mock_redis)
    with pytest.raises(BusinessException) as exc:
        await _call(svc, mock_redis)
    assert "正在生成回复" in str(exc.value.message)
    assert await mock_redis.get("ai:msg:idempotent:7:key-1") is None


async def test_pending_held_with_ttl_until_stream_completes(mock_redis):
    repo, svc = _make_full_service(mock_redis)
    await _call(svc, mock_redis)
    key = "ai:msg:idempotent:7:key-1"
    assert await mock_redis.get(key) == "pending"
    assert await mock_redis.ttl(key) == settings.AI_MESSAGE_STREAM_TIMEOUT + 60
    assert [m.role for m in repo.created] == ["user", "assistant"]


async def test_concurrent_same_key_only_one_sender(mock_redis):
    """并发同幂等键：SETNX 保证只有一个请求进入发送流程，其余命中 pending 冲突"""
    repo, svc = _make_full_service(mock_redis)
    results = await asyncio.gather(
        *(_call(svc, mock_redis) for _ in range(8)), return_exceptions=True
    )
    sent = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, BusinessException)]
    assert len(sent) == 1
    assert len(conflicts) == 7
    assert all(e.code == ResultCode.REPEAT_SUBMIT_ERROR for e in conflicts)
    assert len(repo.created) == 2


async def test_idempotency_key_isolated_by_user(mock_redis):
    await mock_redis.set("ai:msg:idempotent:7:key-1", json.dumps({"messageId": 99, "status": 2}))
    repo, svc = _make_full_service(mock_redis)
    await _call(svc, mock_redis, user_id=8)
    assert await mock_redis.get("ai:msg:idempotent:7:key-1") == json.dumps(
        {"messageId": 99, "status": 2}
    )
    assert await mock_redis.get("ai:msg:idempotent:8:key-1") == "pending"
    assert len(repo.created) == 2


async def test_failure_clears_key_for_retry():
    redis = await fake_redis()

    async def _run_fail(**kwargs):
        raise RuntimeError("推理失败")

    svc = AiMessageService(
        get_redis_client=lambda: redis,
        reasoning_service=_Reasoning(_run_fail),
        sse_emitter_manager=_SseManager(),
    )

    # 推测失败：本轮抢占的 pending 必须释放（Lua CAS，仅释自己的标识）以允许重试
    await redis.set("k", "pending", ex=180)
    await svc._run_reasoning(1, 7, "gpt", 1, "s1", "k")
    assert await redis.get("k") is None
