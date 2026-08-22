import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.service.ai.agent_hooks import _tool_recovery_hook, agent_hooks
from app.service.ai.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.tool_recovery import classify_tool_error
from tests.stubs import StubInterruptHandler

logging.disable(logging.WARNING)


class TestClassification:
    async def test_param_error_retry(self):
        action = classify_tool_error(ValueError("invalid argument"))
        assert action.action == "retry"

    async def test_timeout_fail(self):
        action = classify_tool_error(TimeoutError())
        assert action.action == "fail"
        assert action.status == 2

    async def test_permission_interrupt(self):
        action = classify_tool_error(PermissionError("denied"))
        assert action.action == "interrupt"

    async def test_5xx_skip(self):
        req = httpx.Request("POST", "http://x")
        exc = httpx.HTTPStatusError("500", request=req, response=httpx.Response(503, request=req))
        action = classify_tool_error(exc)
        assert action.action == "skip"
        assert action.status == 3

    async def test_connect_error_skip(self):
        action = classify_tool_error(httpx.ConnectError("conn"))
        assert action.action == "skip"
        assert action.status == 3

    async def test_unrecoverable_fail(self):
        action = classify_tool_error(RuntimeError("boom"))
        assert action.action == "fail"
        assert action.status == 2


class TestAfterToolHook:
    async def test_param_error_retry_within_limit(self):
        result = await _tool_recovery_hook(
            {
                "tool_error": ValueError("bad"),
                "retry_count": 0,
                "retry_max": 2,
            }
        )
        assert result == {"action": "retry", "status": 2, "reason": "工具参数错误: bad"}

    async def test_param_error_retry_exhausted(self):
        result = await _tool_recovery_hook(
            {
                "tool_error": ValueError("bad"),
                "retry_count": 2,
                "retry_max": 2,
            }
        )
        assert result["action"] == "fail"
        assert "重试次数耗尽" in result["reason"]

    async def test_no_error_returns_none(self):
        assert await _tool_recovery_hook({"tool_error": None}) is None

    async def test_registered_in_hooks(self):
        assert any(hook_point == "after_tool" for hook_point in agent_hooks._hooks)


class TestMiddlewareAwrapToolCall:
    def _ctx(self, **overrides):
        ctx = {
            "conversation_id": 1,
            "message_id": 2,
            "stream_session_id": "s",
            "user_id": 3,
            "tool_timeout": 60,
            "retry_max": 2,
            "step_count": 0,
            "token_used": 0,
            "token_budget": 1000,
            "max_steps": 20,
        }
        ctx.update(overrides)
        return ctx

    def _request(self, tool_call):
        return SimpleNamespace(
            tool_call=tool_call,
            state=SimpleNamespace(get=lambda k: None),
        )

    @staticmethod
    def _compat_patch():
        return patch.object(DehazeHooksMiddleware, "_compat_state", return_value={})

    async def test_tool_timeout_applied(self):
        mw = DehazeHooksMiddleware(self._ctx(tool_timeout=1))

        async def slow_handler(request):
            await asyncio.sleep(0.5)
            raise TimeoutError()

        from langchain_core.messages import ToolMessage

        with self._compat_patch():
            result = await mw.awrap_tool_call(
                self._request({"id": "t1", "name": "dehaze", "args": {}}),
                slow_handler,
            )
        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["_dehaze_status"] == 2
        assert "超时" in result.content

    async def test_param_error_retry_returns_toolmessage(self):
        mw = DehazeHooksMiddleware(self._ctx())

        async def failing_handler(request):
            raise ValueError("invalid args")

        with self._compat_patch():
            result = await mw.awrap_tool_call(
                self._request({"id": "t1", "name": "tool", "args": {}}),
                failing_handler,
            )
        assert result.additional_kwargs["_dehaze_status"] == 2
        assert "修正" in result.content
        assert mw.ctx["tool_retries"]["t1"] == 1

    async def test_permission_interrupt_saves_interrupt(self):
        mw = DehazeHooksMiddleware(self._ctx())

        async def failing_handler(request):
            raise PermissionError("no permission")

        handler = StubInterruptHandler()

        with (
            self._compat_patch(),
            patch("app.service.ai.dehaze_hooks_middleware.interrupt_handler", handler),
            patch("app.service.ai.dehaze_hooks_middleware.interrupt", lambda *a, **k: None),
        ):
            from langchain_core.messages import ToolMessage

            result = await mw.awrap_tool_call(
                self._request({"id": "t2", "name": "write_file", "args": {}}),
                failing_handler,
            )
        assert isinstance(result, ToolMessage)
        thread_id, itype, data = handler.saved[0]
        assert data["type"] == "confirm"
        assert data["data"]["tool"] == "write_file"

    async def test_5xx_skip_returns_skipped(self):
        mw = DehazeHooksMiddleware(self._ctx())
        req = httpx.Request("POST", "http://x")
        exc = httpx.HTTPStatusError("503", request=req, response=httpx.Response(503, request=req))

        async def failing_handler(request):
            raise exc

        with self._compat_patch():
            result = await mw.awrap_tool_call(
                self._request({"id": "t3", "name": "mcp", "args": {}}),
                failing_handler,
            )
        assert result.additional_kwargs["_dehaze_status"] == 3
        assert "跳过" in result.content

    async def test_success_passthrough(self):
        mw = DehazeHooksMiddleware(self._ctx())
        from langchain_core.messages import ToolMessage

        ok = ToolMessage(content="ok", tool_call_id="t4", name="tool")

        async def ok_handler(request):
            return ok

        result = await mw.awrap_tool_call(
            self._request({"id": "t4", "name": "tool", "args": {}}), ok_handler
        )
        assert result is ok
