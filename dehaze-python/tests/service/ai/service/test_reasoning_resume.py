from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai.middleware.interrupt_handler import ConfirmKind
from app.service.ai.service.reasoning_service import reasoning_service
from tests.stubs.factories import fake_redis
from tests.stubs.fakes import RecorderEmitter, StubInterruptHandler

pytestmark = pytest.mark.requires_db


async def _noop(**_kwargs):
    return None


def _confirm_interrupt(kind, **payload) -> dict:
    """构造 confirm 中断点（与 interrupt_handler.save_interrupt 落库形状一致）"""
    return {
        "type": "confirm",
        "data": {
            "type": "confirm",
            "stream_session_id": "s1",
            "data": {"confirmKind": kind, **payload},
        },
    }


class _Conv:
    id = 10
    user_id = 1
    model = "gpt-4o-mini"
    agent_code = None
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


class _FailingGraph(_Graph):
    async def astream(self, command, config=None, **kw):
        raise RuntimeError("checkpoint 反序列化失败 secret=abc123")
        yield  # pragma: no cover


class _MsgRepo:
    async def get_by_id(self, db, msg_id):
        return SimpleNamespace(id=msg_id, model="gpt-4o-mini")


class _ConvRepo:
    async def get_by_id_and_user(self, db, cid, uid):
        return _Conv()


def _patch_resume_deps(db, monkeypatch, interrupt, graph=None, confirmation=None):
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
    monkeypatch.setattr("app.service.ai.service.reasoning_service.interrupt_handler", ih)

    emitter = RecorderEmitter()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.sse_emitter_manager", emitter)
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.ai_message_repository", _MsgRepo()
    )
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.ai_conversation_repository", _ConvRepo()
    )
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service._schedule_conversation_sync", lambda *a, **k: None
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
            "app.service.ai.service.algorithm_recommend_service.handle_user_confirmation", _handle
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


async def test_resume_confirm_injects_user_choice(db, monkeypatch):
    graph = _Graph()
    interrupt = _confirm_interrupt(
        ConfirmKind.ALGORITHM_RECOMMEND, algorithms=[1, 2, 3], artifactId=9
    )
    service, ih, emitter, finalized, conf = _patch_resume_deps(
        db, monkeypatch, interrupt, graph=graph, confirmation=True
    )

    result = await service.resume(10, 1, 5, {"confirmed": True, "algorithmId": 2})

    assert conf == {"confirmed": True, "algorithm_id": 2}
    assert graph.resumed == {"confirmed": True, "algorithmId": 2}
    assert graph.config == {"configurable": {"thread_id": "10:5"}}
    assert result["final_response"] == "按你的选择执行"
    _assert_resume_succeeded(ih, emitter, finalized)


async def test_resume_quota_uses_resume_true(db, monkeypatch):
    graph = _Graph()
    interrupt = {"type": "quota", "data": {"stream_session_id": "s1"}}
    service, ih, emitter, finalized, _conf = _patch_resume_deps(
        db, monkeypatch, interrupt, graph=graph
    )

    result = await service.resume(10, 1, 5, {})

    assert graph.resumed is True
    assert result["stop_reason"] == "stop"
    _assert_resume_succeeded(ih, emitter, finalized)


async def test_resume_async_wait_injects_task_result(db, monkeypatch):
    graph = _Graph()
    interrupt = {"type": "async_wait", "data": {"stream_session_id": "s1"}}
    service, ih, emitter, finalized, _conf = _patch_resume_deps(
        db, monkeypatch, interrupt, graph=graph
    )

    summary = {"total": 4, "success": 4, "failed": 0}
    await service.resume(10, 1, 5, {"async_task": summary})

    assert graph.resumed == {"async_task": summary}
    _assert_resume_succeeded(ih, emitter, finalized)


async def test_resume_failure_pushes_single_sanitized_error(db, monkeypatch):
    """resume 失败：error 事件只推一份（SSE 侧），失败态落库由 _mark_failed 负责。"""
    service, ih, emitter, _finalized, _conf = _patch_resume_deps(
        db,
        monkeypatch,
        _confirm_interrupt(ConfirmKind.ALGORITHM_RECOMMEND),
        graph=_FailingGraph(),
        confirmation=True,
    )
    marked = []

    async def _mark_failed(msg_id, error):
        marked.append((msg_id, str(error)))

    monkeypatch.setattr(reasoning_service, "_mark_failed", _mark_failed)
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.trace_collector",
        SimpleNamespace(
            start=lambda **_kw: None,
            current=lambda: SimpleNamespace(
                agent_code=None, model_id=None, record_event=lambda **_kw: None
            ),
            finalize_unsettled=_noop,
            error_type_of=lambda _e: "RuntimeError",
            TRACE_STATUS_FAILED="failed",
            TRACE_STATUS_INTERRUPTED="interrupted",
        ),
    )

    with pytest.raises(RuntimeError):
        await service.resume(10, 1, 5, {"confirmed": True})

    assert marked[0][0] == 5
    # 中断点保留：resume 中途失败后用户仍可再次确认重试（清理只在成功后）
    assert ih.cleared == []
    assert [event_type for event_type, _ in emitter.events] == ["error", "message.end"]
    assert emitter.events[0][1] == {
        "code": ResultCode.AI_LLM_CALL_FAILED.code,
        "message": ResultCode.AI_LLM_CALL_FAILED.msg,
    }
    assert "secret=abc123" not in str(emitter.events)


async def test_resume_dangerous_op_forwards_confirmation_only(db, monkeypatch):
    """危险操作确认：不走算法推荐反馈，用户确认结果原样透传给中断点。"""
    graph = _Graph()
    service, ih, _emitter, _finalized, conf = _patch_resume_deps(
        db,
        monkeypatch,
        _confirm_interrupt(ConfirmKind.DANGEROUS_OP, command="rm -rf /tmp/x"),
        graph=graph,
        confirmation=True,
    )

    await service.resume(10, 1, 5, {"confirmed": True})

    assert conf == {}
    assert graph.resumed == {"confirmed": True}
    assert ih.cleared == ["10:5"]


async def test_resume_tool_permission_forwards_confirmation_only(db, monkeypatch):
    graph = _Graph()
    service, _ih, _emitter, _finalized, conf = _patch_resume_deps(
        db,
        monkeypatch,
        _confirm_interrupt(ConfirmKind.TOOL_PERMISSION, tool="execute_code"),
        graph=graph,
        confirmation=True,
    )

    await service.resume(10, 1, 5, {"confirmed": True})

    assert conf == {}
    assert graph.resumed == {"confirmed": True}


async def test_resume_unknown_confirm_kind_raises(db, monkeypatch):
    """未标识子类型的确认中断不得按算法推荐处理，须显式报错。"""
    service, ih, _emitter, _finalized, _conf = _patch_resume_deps(
        db, monkeypatch, _confirm_interrupt("legacy_confirm"), confirmation=True
    )

    with pytest.raises(BusinessException) as exc:
        await service.resume(10, 1, 5, {"confirmed": True})

    assert exc.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    assert ih.cleared == []


async def test_resume_missing_interrupt_raises(db, monkeypatch):
    service, ih, *_ = _patch_resume_deps(db, monkeypatch, None)

    with pytest.raises(BusinessException) as exc:
        await service.resume(10, 1, 5, {"confirmed": True})
    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND
    assert ih.cleared == []
