from langchain_core.messages import AIMessage

from app.service.ai.dehaze_hooks_middleware import DehazeHooksMiddleware
from tests.stubs import StubInterruptHandler


def _make_middleware():
    ctx = {
        "conversation_id": 1,
        "message_id": 2,
        "user_id": 10,
        "stream_session_id": "s1",
        "model_id": "gpt-4o-mini",
        "token_budget": 500,
        "max_steps": 20,
        "token_used": 0,
        "step_count": 0,
    }
    return ctx, DehazeHooksMiddleware(ctx)


class _Request:
    state = {"messages": []}
    system_message = None

    def override(self, system_message=None):
        self.system_message = system_message
        return self


class _Resp:
    def __init__(self, content="ok"):
        self.result = AIMessage(content=content)


def _install(monkeypatch, handler_flag_key="hit"):
    captured = {}
    handler_called = {}

    def _fake_interrupt(data):
        captured["interrupt_data"] = data
        return True

    async def _handler(request):
        handler_called[handler_flag_key] = True
        return _Resp()

    monkeypatch.setattr("app.service.ai.dehaze_hooks_middleware.interrupt", _fake_interrupt)
    monkeypatch.setattr("app.service.ai.dehaze_hooks_middleware.interrupt_handler", StubInterruptHandler())
    return captured, handler_called, _handler


async def test_precharge_quota_becomes_interrupt(monkeypatch, mock_redis):
    ctx, mw = _make_middleware()
    ctx["precharge_blocked"] = {"final_response": "配额不足", "stop_reason": "quota_exceeded"}

    captured, handler_called, handler = _install(monkeypatch)

    await mw.awrap_model_call(request=_Request(), handler=handler)

    assert captured["interrupt_data"]["type"] == "quota"
    assert captured["interrupt_data"]["stream_session_id"] == "s1"
    assert "upgrade_tip" in captured["interrupt_data"]["data"]
    assert handler_called.get("hit")
    assert "precharge_blocked" not in ctx


async def test_before_model_quota_becomes_interrupt(monkeypatch, mock_redis):
    ctx, mw = _make_middleware()

    class _Hooks:
        async def run_hooks(self, point, state):
            return {
                "final_response": "预算不足",
                "stop_reason": "quota_exceeded",
                "interrupt": {"type": "quota"},
            }

    captured, handler_called, handler = _install(monkeypatch)
    monkeypatch.setattr("app.service.ai.dehaze_hooks_middleware.agent_hooks", _Hooks())

    await mw.awrap_model_call(request=_Request(), handler=handler)
    assert captured["interrupt_data"]["type"] == "quota"
    assert handler_called.get("hit")


async def test_non_quota_block_short_circuits(monkeypatch):
    ctx, mw = _make_middleware()

    class _Hooks:
        async def run_hooks(self, point, state):
            return {"final_response": "已达最大步数", "stop_reason": "max_steps"}

    captured, handler_called, handler = _install(monkeypatch, handler_flag_key="handler")
    monkeypatch.setattr("app.service.ai.dehaze_hooks_middleware.agent_hooks", _Hooks())

    result = await mw.awrap_model_call(request=_Request(), handler=handler)
    assert captured.get("interrupt_data") is None
    assert handler_called.get("handler") is None
    assert isinstance(result, AIMessage)
    assert ctx["stop_reason"] == "max_steps"
