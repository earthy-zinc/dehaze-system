import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.service.ai.reasoning_service import reasoning_service
from tests.stubs import RecorderEmitter


@asynccontextmanager
async def _db_session():
    yield object()


async def test_fail_pushes_error_then_message_end(monkeypatch):
    service = reasoning_service
    emitter = RecorderEmitter()
    monkeypatch.setattr("app.service.ai.reasoning_service.sse_emitter_manager", emitter)

    class _Repo:
        async def update_status(self, db, msg_id, status, error=None):
            return None

    monkeypatch.setattr("app.service.ai.reasoning_service.ai_message_repository", _Repo())
    monkeypatch.setattr("app.service.ai.reasoning_service.get_db_session", _db_session)

    await service._fail(1, "s1", Exception("boom"))
    assert [t for t, _ in emitter.events] == ["error", "message.end"]
    assert emitter.events[1][1]["stopReason"] == "error"


def _patch_suggestion_service(monkeypatch, generate):
    monkeypatch.setattr(
        "app.service.ai.reasoning_service.suggestion_service",
        SimpleNamespace(generate=generate),
    )


async def test_trigger_suggestions_skips_on_cancel(monkeypatch):
    called = False

    async def _generate(**kwargs):
        nonlocal called
        called = True
        return ["追问一"]

    _patch_suggestion_service(monkeypatch, _generate)
    reasoning_service._trigger_suggestions(
        1, 2, {"final_response": "x", "stop_reason": "canceled"}, 1, "s1"
    )
    await asyncio.sleep(0.05)
    assert called is False


async def test_trigger_suggestions_calls_generate(monkeypatch):
    captured = {}

    async def _generate(**kwargs):
        captured.update(kwargs)
        return ["追问一"]

    _patch_suggestion_service(monkeypatch, _generate)
    reasoning_service._trigger_suggestions(
        1, 2, {"final_response": "回答", "stop_reason": "stop"}, 1, "s1"
    )
    await asyncio.sleep(0.05)
    assert captured.get("message_id") == 2
    assert captured.get("reply_content") == "回答"
    assert captured.get("stream_session_id") == "s1"
