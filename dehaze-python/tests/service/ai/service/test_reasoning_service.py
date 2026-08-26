from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.service.ai.service.reasoning_service import _pending_tasks, reasoning_service
from tests.stubs.fakes import RecorderEmitter


@asynccontextmanager
async def _db_session():
    yield object()


async def test_fail_pushes_error_then_message_end(monkeypatch):
    service = reasoning_service
    emitter = RecorderEmitter()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.sse_emitter_manager", emitter)

    class _Repo:
        async def update_status(self, db, msg_id, status, error=None):
            return None

    monkeypatch.setattr("app.service.ai.service.reasoning_service.ai_message_repository", _Repo())
    monkeypatch.setattr("app.service.ai.service.reasoning_service.get_db_session", _db_session)

    await service._fail(1, "s1", Exception("boom"))
    assert [t for t, _ in emitter.events] == ["error", "message.end"]
    assert emitter.events[1][1]["stopReason"] == "error"


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


def test_state_result_raises_when_values_missing():
    """推理未产出 state.values（异常中断）不得伪装成成功态，须显式抛错。"""
    class _BrokenState:
        values = None

    with pytest.raises(RuntimeError, match="推理未产出有效 state.values"):
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
