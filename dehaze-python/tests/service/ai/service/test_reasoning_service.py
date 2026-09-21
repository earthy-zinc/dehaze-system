import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai.builders.deep_agent_builder import DeepAgentBuilder
from app.service.ai.service.reasoning_service import (
    _GRAPH_CACHE_MAX,
    ReasoningService,
    _pending_tasks,
    reasoning_service,
)
from tests.stubs.fakes import RecorderEmitter


@asynccontextmanager
async def _db_session():
    yield object()


def _patch_failed_deps(monkeypatch, emitter, update_status=None):
    monkeypatch.setattr("app.service.ai.service.reasoning_service.sse_emitter_manager", emitter)
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.ai_message_repository",
        SimpleNamespace(update_status=update_status),
    )
    monkeypatch.setattr("app.service.ai.service.reasoning_service.get_db_session", _db_session)


async def test_mark_failed_only_persists_failed_status(monkeypatch):
    """失败态落库由推理侧负责，SSE error 事件由入口最外层推——此处不推事件。"""
    service = reasoning_service
    emitter = RecorderEmitter()
    updates = []

    async def _update_status(db, msg_id, status, error=None):
        updates.append((msg_id, status, error))

    _patch_failed_deps(monkeypatch, emitter, _update_status)

    await service._mark_failed(1, RuntimeError("minio 连接失败 secret=abc123"))

    assert updates == [(1, 3, ResultCode.AI_LLM_CALL_FAILED.msg)]
    assert emitter.events == []


async def test_mark_failed_keeps_business_message(monkeypatch):
    service = reasoning_service
    emitter = RecorderEmitter()
    updates = []

    async def _update_status(db, msg_id, status, error=None):
        updates.append((msg_id, status, error))

    _patch_failed_deps(monkeypatch, emitter, _update_status)

    await service._mark_failed(2, BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在"))

    assert updates == [(2, 3, "会话不存在")]


async def test_mark_failed_logs_persistence_error(monkeypatch, caplog):
    service = reasoning_service
    emitter = RecorderEmitter()

    async def _update_status(*_args, **_kwargs):
        raise RuntimeError("db down")

    _patch_failed_deps(monkeypatch, emitter, _update_status)
    caplog.set_level(logging.ERROR, logger="app.service.ai.service.reasoning_service")

    await service._mark_failed(3, RuntimeError("boom"))

    assert "标记消息失败态失败" in caplog.text


async def test_run_raises_business_error_when_conversation_missing(monkeypatch):
    """会话不存在/越权：抛业务异常，不得裸抛 AttributeError。"""

    async def _no_conv(db, conv_id, user_id):
        return None

    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.ai_conversation_repository",
        SimpleNamespace(get_by_id_and_user=_no_conv),
    )
    monkeypatch.setattr("app.service.ai.service.reasoning_service.get_db_session", _db_session)

    with pytest.raises(BusinessException) as exc:
        await reasoning_service.run(999, 1, 5, "gpt-4o-mini", "s1")

    assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


def _patch_graph_builders(monkeypatch, built: list):
    async def _snapshot(db, redis, agent_id, version_no=None):
        return {"is_team": False}

    async def _build(db, redis, snapshot, checkpointer=None):
        model_id = snapshot.get("model_id", "")
        built.append(model_id)
        return f"graph:{model_id}"

    monkeypatch.setattr(
        "app.service.ai_agent_service.agent_service.get_published_snapshot", _snapshot
    )
    monkeypatch.setattr(DeepAgentBuilder, "build_from_snapshot", _build)
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.checkpoint_manager",
        SimpleNamespace(get_checkpointer=lambda: None),
    )


async def test_graph_cache_reuses_built_graph(monkeypatch):
    service = ReasoningService()
    built = []
    _patch_graph_builders(monkeypatch, built)

    first = await service._build_graph(None, None, 1, 2, "gpt-4o-mini")
    second = await service._build_graph(None, None, 1, 2, "gpt-4o-mini")

    assert built == ["gpt-4o-mini"]
    assert first == second


async def test_graph_cache_evicts_least_recently_used(monkeypatch):
    service = ReasoningService()
    built = []
    _patch_graph_builders(monkeypatch, built)

    for i in range(_GRAPH_CACHE_MAX):
        await service._build_graph(None, None, 1, 1, f"m{i}")
    assert len(service._graphs) == _GRAPH_CACHE_MAX

    # m0 被重新访问 → 最久未访问的是 m1
    await service._build_graph(None, None, 1, 1, "m0")
    await service._build_graph(None, None, 1, 1, "m-new")

    assert len(service._graphs) == _GRAPH_CACHE_MAX
    assert (1, 1, "m0") in service._graphs
    assert (1, 1, "m1") not in service._graphs
    assert (1, 1, "m-new") in service._graphs
    assert built[-1] == "m-new"


def _patch_suggestion_service(monkeypatch, generate):
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.suggestion_service",
        SimpleNamespace(generate=generate),
    )


async def _trigger_suggestions_and_drain(*args) -> None:
    """触发 fire-and-forget 推荐并等待其后台 task 结束。

    _trigger_suggestions 把 task 注册进模块级 _pending_tasks，取差集即可确定性等待，
    避免用 sleep 猜测后台完成时机。
    """
    before = set(_pending_tasks)
    reasoning_service._trigger_suggestions(*args)
    for task in _pending_tasks - before:
        await task


async def test_trigger_suggestions_skips_on_cancel(monkeypatch):
    called = False

    async def _generate(**kwargs):
        nonlocal called
        called = True
        return ["追问一"]

    _patch_suggestion_service(monkeypatch, _generate)
    await _trigger_suggestions_and_drain(
        1, 2, {"final_response": "x", "stop_reason": "canceled"}, 1, "s1"
    )
    assert called is False


async def test_trigger_suggestions_calls_generate(monkeypatch):
    captured = {}

    async def _generate(**kwargs):
        captured.update(kwargs)
        return ["追问一"]

    _patch_suggestion_service(monkeypatch, _generate)
    await _trigger_suggestions_and_drain(
        1, 2, {"final_response": "回答", "stop_reason": "stop"}, 1, "s1"
    )
    assert captured.get("message_id") == 2
    assert captured.get("reply_content") == "回答"
    assert captured.get("stream_session_id") == "s1"


async def test_run_direct_masks_pii_before_persisting(monkeypatch):
    """direct 路径落库内容须脱敏：SSE 出口由 converter 脱敏，落库用的是原始累积文本。"""
    from langchain_core.messages import AIMessageChunk

    from app.infrastructure.llm.client import dehaze_chat_model
    from app.infrastructure.sse import sse_event_converter
    from app.repository.ai_model_repository import ai_model_repository

    service = reasoning_service
    emitter = RecorderEmitter()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.sse_emitter_manager", emitter)
    monkeypatch.setattr(sse_event_converter, "sse_emitter_manager", emitter)
    monkeypatch.setattr("app.service.ai.service.reasoning_service.get_db_session", _db_session)
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service._schedule_conversation_sync",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.trace_collector",
        SimpleNamespace(
            finalize_success=_noop_finalize,
            current=lambda: None,
            TRACE_STATUS_FAILED="failed",
        ),
    )

    class _FakeModel:
        _last_usage = {"input_tokens": 3, "output_tokens": 4}

        def __init__(self, model):
            pass

        async def astream(self, messages):
            for text in ("联系我 ", "13812345678", "，密钥 sk-abcdef123456"):
                yield AIMessageChunk(content=text)

    monkeypatch.setattr(dehaze_chat_model, "DehazeChatModel", _FakeModel)

    async def _no_model(db, model_id):
        return None

    async def _pre_charge(db, user_id, conv_id, msg_id, text, model_id):
        return {"billing_id": 1}

    async def _settle(db, *args, **kwargs):
        return None

    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.billing_service",
        SimpleNamespace(pre_charge=_pre_charge, settle=_settle),
    )
    monkeypatch.setattr(ai_model_repository, "get_by_model_id", _no_model)
    persisted = {}

    async def _finalize(msg_id, result, model_id, used_memory_ids=None):
        persisted.update(result)
        return 0

    monkeypatch.setattr(service, "_finalize_message", _finalize)

    result = await service._run_direct(
        conv_id=1,
        user_id=2,
        msg_id=3,
        model_id="gpt-4o-mini",
        stream_session_id="s1",
        messages=[{"role": "user", "content": "我的联系方式"}],
        system_prompt=None,
    )

    assert persisted["final_response"] == "联系我 ***，密钥 ***"
    assert "13812345678" not in persisted["final_response"]
    assert result["final_response"] == persisted["final_response"]
    # SSE 出口同样不含明文（converter 侧脱敏）
    assert "13812345678" not in str(emitter.events)


async def _noop_finalize(**_kwargs):
    return None


def test_state_result_raises_when_values_missing():
    """推理未产出 state.values（异常中断）不得伪装成成功态，须显式抛错。"""

    class _BrokenState:
        values = None

    with pytest.raises(RuntimeError, match=r"推理未产出有效 state\.values"):
        reasoning_service._state_result(_BrokenState())


def test_state_result_extracts_from_values():
    """正常路径：从 state.values 提取 final_response/stop_reason/usage。"""

    class _State:
        values = {
            "final_response": "回答",
            "stop_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    result = reasoning_service._state_result(_State())
    assert result["final_response"] == "回答"
    assert result["stop_reason"] == "stop"
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5
