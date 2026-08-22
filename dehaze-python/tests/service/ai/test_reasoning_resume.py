from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai.reasoning_service import reasoning_service
from tests.stubs import NullDBSession, RecorderEmitter, StubInterruptHandler, fake_redis


class _Conv:
    id = 10
    user_id = 1
    model = "gpt-4o-mini"
    status = 1
    current_branch_message_id = 5


class _Graph:
    def __init__(self):
        self.resumed = None
        self.config = None

    async def astream(self, command, config=None, **kw):
        self.resumed = command.resume
        self.config = config
        return
        yield  # pragma: no cover

    async def aget_state(self, config):
        return SimpleNamespace(
            values={"final_response": "按你的选择执行", "stop_reason": "stop", "usage": {}}
        )


class _MsgRepo:
    async def get_by_id(self, db, msg_id):
        return SimpleNamespace(id=msg_id, model="gpt-4o-mini")


class _ConvRepo:
    async def get_by_id_and_user(self, db, cid, uid):
        return _Conv()


def _patch_resume_deps(monkeypatch, interrupt, graph=None, confirmation=None):
    monkeypatch.setattr("app.service.ai.reasoning_service.get_db_session", NullDBSession)

    async def _load_anchor(db, conv):
        return (1, 1)

    async def _build_graph(db, redis, a, v, model_id=None):
        return graph or _Graph()

    monkeypatch.setattr(reasoning_service, "_load_agent_anchor", _load_anchor)
    monkeypatch.setattr(reasoning_service, "_build_graph", _build_graph)

    async def _get_redis():
        return await fake_redis()

    monkeypatch.setattr("app.dependencies.redis.get_redis_client", _get_redis)

    ih = StubInterruptHandler(interrupt)
    monkeypatch.setattr("app.service.ai.reasoning_service.interrupt_handler", ih)

    emitter = RecorderEmitter()
    monkeypatch.setattr("app.service.ai.reasoning_service.sse_emitter_manager", emitter)
    monkeypatch.setattr("app.service.ai.reasoning_service.ai_message_repository", _MsgRepo())
    monkeypatch.setattr("app.service.ai.reasoning_service.ai_conversation_repository", _ConvRepo())
    monkeypatch.setattr(
        "app.service.ai.reasoning_service._schedule_conversation_sync", lambda *a, **k: None
    )

    finalized = {}

    async def _finalize(msg_id, result, model_id, used_memory_ids=None):
        finalized["hit"] = True
        return 0

    monkeypatch.setattr(reasoning_service, "_finalize_message", _finalize)

    conf = {}

    if confirmation is not None:

        async def _handle(conv_id, msg_id, user_id, confirmed, algorithm_id):
            conf["confirmed"] = confirmed
            conf["algorithm_id"] = algorithm_id

        monkeypatch.setattr(
            "app.service.ai.algorithm_recommend_service.handle_user_confirmation", _handle
        )

    service = reasoning_service
    return service, ih, emitter, finalized, conf


def _assert_resume_succeeded(ih, emitter, finalized):
    assert ih.cleared == ["10:5"]
    assert emitter.events[-1] == (
        "message.end",
        {
            "stopReason": "stop",
            "usage": {
                "inputTokens": 0,
                "outputTokens": 0,
                "cachedInputTokens": 0,
                "credits": 0,
            },
        },
    )
    assert finalized.get("hit") is True


async def test_resume_confirm_injects_user_choice(monkeypatch):
    graph = _Graph()
    interrupt = {
        "type": "confirm",
        "data": {"stream_session_id": "s1", "algorithms": [1, 2, 3]},
    }
    service, ih, emitter, finalized, conf = _patch_resume_deps(
        monkeypatch, interrupt, graph=graph, confirmation=True
    )

    result = await service.resume(10, 1, 5, {"confirmed": True, "algorithmId": 2})

    assert conf == {"confirmed": True, "algorithm_id": 2}
    assert graph.resumed == {"confirmed": True, "algorithmId": 2}
    assert graph.config == {"configurable": {"thread_id": "10:5"}}
    assert result["final_response"] == "按你的选择执行"
    _assert_resume_succeeded(ih, emitter, finalized)


async def test_resume_quota_uses_resume_true(monkeypatch):
    graph = _Graph()
    interrupt = {"type": "quota", "data": {"stream_session_id": "s1"}}
    service, ih, emitter, finalized, conf = _patch_resume_deps(monkeypatch, interrupt, graph=graph)

    result = await service.resume(10, 1, 5, {})

    assert graph.resumed is True
    assert result["stop_reason"] == "stop"
    _assert_resume_succeeded(ih, emitter, finalized)


async def test_resume_async_wait_injects_task_result(monkeypatch):
    graph = _Graph()
    interrupt = {"type": "async_wait", "data": {"stream_session_id": "s1"}}
    service, ih, emitter, finalized, conf = _patch_resume_deps(monkeypatch, interrupt, graph=graph)

    summary = {"total": 4, "success": 4, "failed": 0}
    await service.resume(10, 1, 5, {"async_task": summary})

    assert graph.resumed == {"async_task": summary}
    _assert_resume_succeeded(ih, emitter, finalized)


async def test_resume_missing_interrupt_raises(monkeypatch):
    service, ih, *_ = _patch_resume_deps(monkeypatch, None)

    with pytest.raises(BusinessException) as exc:
        await service.resume(10, 1, 5, {"confirmed": True})
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
    assert ih.cleared == []
