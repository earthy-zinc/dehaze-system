"""并行子 Agent 写冲突仲裁测试（before_tool 注册点，智能体管理 §5.3.1）

覆盖：写入资源键识别、低优先级让行（等待/超时跳过）、同优先级失败重试与升级中断、
run 内串行覆盖中断提示回滚（确认后放行）、释放闭环（成功/异常路径）、
非写操作不参与仲裁。
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import ToolMessage

from app.service.ai.middleware import agent_hooks as hooks_module
from app.service.ai.middleware.agent_hooks import (
    _write_conflict_hook,
    release_write_resource,
    write_resource_key,
)
from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.middleware.interrupt_handler import ConfirmKind
from tests.stubs.fakes import StubInterruptHandler

_RESOURCE = "file:/work/a.txt"


def _holder(name, priority=0, instance=None):
    return {"name": name, "priority": priority, "instance": instance}


async def _arbitrate(ctx, holder, resource=_RESOURCE, **overrides):
    return await _write_conflict_hook(
        {"run_ctx": ctx, "resource": resource, "holder": holder, "retry_max": 2, **overrides}
    )


def _running_holder(ctx, name, priority):
    """登记一个"正在写入该资源"的持有者（尚未写完，不计入写入来源）"""
    ctx["write_holders"] = {
        _RESOURCE: {
            "name": name,
            "priority": priority,
            "instance": None,
            "depth": 1,
            "released": asyncio.Event(),
        }
    }


class TestResourceKey:
    def test_file_write_tools_keyed_by_path(self):
        for tool in ("write_file", "edit_file", "delete"):
            assert write_resource_key(tool, {"file_path": "/work/a.txt"}) == _RESOURCE

    def test_file_write_tool_without_path_not_arbitrated(self):
        assert write_resource_key("write_file", {}) is None

    def test_mcp_write_api_keyed_by_tool_and_params(self):
        key = write_resource_key(
            "mcp_execute_tool",
            {"tool_name": "post_api_v1_prediction", "arguments": {"b": 2, "a": 1}},
        )
        assert key == 'api:post_api_v1_prediction:{"a": 1, "b": 2}'

    def test_mcp_read_api_and_read_tools_not_arbitrated(self):
        assert (
            write_resource_key(
                "mcp_execute_tool", {"tool_name": "get_api_v1_prediction_logs", "arguments": {}}
            )
            is None
        )
        assert write_resource_key("read_file", {"file_path": "/work/a.txt"}) is None


class TestWriteConflictHook:
    async def test_same_holder_reentrant_and_reacquirable_after_release(self):
        ctx = {}
        assert await _arbitrate(ctx, _holder("A", 1)) is None
        assert await _arbitrate(ctx, _holder("A", 1)) is None
        assert ctx["write_holders"][_RESOURCE]["depth"] == 2
        release_write_resource(ctx, _RESOURCE, _holder("A", 1))
        assert ctx["write_holders"][_RESOURCE]["depth"] == 1
        release_write_resource(ctx, _RESOURCE, _holder("A", 1))
        assert ctx["write_holders"] == {}
        assert await _arbitrate(ctx, _holder("A", 1)) is None

    async def test_parallel_instances_of_same_subagent_conflict(self):
        """同一子 Agent 被并行派发（实例 id 不同）属冲突而非重入"""
        ctx = {}
        assert await _arbitrate(ctx, _holder("X", 1, instance="task-1")) is None
        verdict = await _arbitrate(ctx, _holder("X", 1, instance="task-2"))
        assert verdict is not None
        assert verdict["action"] == "retry"

    async def test_low_priority_yields_then_prompts_overwrite(self):
        ctx = {}
        assert await _arbitrate(ctx, _holder("A", 1)) is None
        waiter = asyncio.create_task(_arbitrate(ctx, _holder("B", 5)))
        await asyncio.sleep(0.05)
        assert not waiter.done()  # 让行等待中，未抢占
        release_write_resource(ctx, _RESOURCE, _holder("A", 1))
        verdict = await asyncio.wait_for(waiter, 1)
        assert verdict is not None
        assert verdict["action"] == "interrupt"
        assert verdict["previousWriter"] == "A"

    async def test_low_priority_yield_timeout_skips(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "_WRITE_YIELD_TIMEOUT", 0.1)
        ctx = {}
        assert await _arbitrate(ctx, _holder("A", 1)) is None
        verdict = await _arbitrate(ctx, _holder("B", 5))
        assert verdict is not None
        assert verdict["action"] == "skip"
        assert ctx["write_holders"][_RESOURCE]["name"] == "A"

    async def test_same_priority_retries_then_escalates_to_interrupt(self):
        ctx = {}
        assert await _arbitrate(ctx, _holder("A", 1)) is None
        retry_first = await _arbitrate(ctx, _holder("B", 1))
        assert retry_first is not None
        assert retry_first["action"] == "retry"
        retry_second = await _arbitrate(ctx, _holder("B", 1))
        assert retry_second is not None
        assert retry_second["action"] == "retry"
        verdict = await _arbitrate(ctx, _holder("B", 1))  # 重试上限 retry_max=2
        assert verdict is not None
        assert verdict["action"] == "interrupt"

    async def test_higher_priority_cannot_preempt_and_retries(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "_WRITE_YIELD_TIMEOUT", 0.1)
        ctx = {}
        assert await _arbitrate(ctx, _holder("B", 5)) is None
        verdict = await _arbitrate(ctx, _holder("A", 1))
        assert verdict is not None
        assert verdict["action"] == "retry"

    async def test_serial_overwrite_prompted_once_then_approved(self):
        ctx = {}
        assert await _arbitrate(ctx, _holder("A", 1)) is None
        release_write_resource(ctx, _RESOURCE, _holder("A", 1))
        verdict = await _arbitrate(ctx, _holder("B", 5))
        assert verdict is not None
        assert verdict["action"] == "interrupt"
        assert await _arbitrate(ctx, _holder("B", 5), overwrite_approved=True) is None


class TestMiddlewareWriteArbitration:
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

    @staticmethod
    def _request(tool_call):
        return SimpleNamespace(tool_call=tool_call, state=SimpleNamespace(get=lambda k: None))

    @staticmethod
    def _compat_patch():
        return patch.object(DehazeHooksMiddleware, "_compat_state", return_value={})

    @staticmethod
    def _write_call(tool_call_id="t1"):
        return {
            "id": tool_call_id,
            "name": "write_file",
            "args": {"file_path": "/work/a.txt"},
        }

    async def test_write_tool_holds_resource_until_finished(self):
        ctx = self._ctx()
        mw = DehazeHooksMiddleware(ctx)
        observed = []

        async def handler(request):
            observed.append(ctx["write_holders"][_RESOURCE]["name"])
            return ToolMessage(content="ok", tool_call_id="t1", name="write_file")

        with self._compat_patch():
            result = await mw.awrap_tool_call(self._request(self._write_call()), handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "ok"
        assert observed == ["主 Agent"]
        assert ctx["write_holders"] == {}

    async def test_resource_released_on_tool_exception(self):
        ctx = self._ctx()
        mw = DehazeHooksMiddleware(ctx)

        async def handler(request):
            raise ValueError("boom")

        with self._compat_patch():
            result = await mw.awrap_tool_call(self._request(self._write_call()), handler)

        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["_dehaze_status"] == 2
        assert ctx["write_holders"] == {}

    async def test_read_tool_not_arbitrated(self):
        ctx = self._ctx()
        mw = DehazeHooksMiddleware(ctx)
        called = []

        async def handler(request):
            called.append(1)
            return ToolMessage(content="ok", tool_call_id="t1", name="read_file")

        tool_call = {"id": "t1", "name": "read_file", "args": {"file_path": "/work/a.txt"}}
        result = await mw.awrap_tool_call(self._request(tool_call), handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "ok"
        assert called == [1]
        assert ctx.get("write_holders") is None

    async def test_low_priority_conflict_skips_without_call(self, monkeypatch):
        monkeypatch.setattr(hooks_module, "_WRITE_YIELD_TIMEOUT", 0.1)
        ctx = self._ctx()
        _running_holder(ctx, "高优子Agent", 1)
        mw = DehazeHooksMiddleware(ctx, subagent={"name": "低优子Agent", "priority": 9})
        called = []

        async def handler(request):
            called.append(1)
            return ToolMessage(content="ok", tool_call_id="t1", name="write_file")

        with self._compat_patch():
            result = await mw.awrap_tool_call(self._request(self._write_call()), handler)

        assert isinstance(result, ToolMessage)
        assert called == []
        assert result.additional_kwargs["_dehaze_status"] == 3
        assert "资源写入冲突" in result.content

    async def test_same_priority_conflict_returns_retry(self):
        ctx = self._ctx()
        _running_holder(ctx, "同优子Agent", 3)
        mw = DehazeHooksMiddleware(ctx, subagent={"name": "另一同优子Agent", "priority": 3})

        async def handler(request):
            return ToolMessage(content="ok", tool_call_id="t1", name="write_file")

        with self._compat_patch():
            result = await mw.awrap_tool_call(self._request(self._write_call()), handler)

        assert isinstance(result, ToolMessage)
        assert result.additional_kwargs["_dehaze_status"] == 2
        assert "请稍后重试" in result.content

    async def test_overwrite_conflict_interrupts_and_reject_aborts_write(self):
        ctx = self._ctx()
        ctx["write_written_by"] = {_RESOURCE: "子AgentA"}
        mw = DehazeHooksMiddleware(ctx, subagent={"name": "子AgentB", "priority": 5})
        stub = StubInterruptHandler()
        called = []

        async def handler(request):
            called.append(1)
            return ToolMessage(content="written", tool_call_id="t1", name="write_file")

        with (
            self._compat_patch(),
            patch("app.service.ai.middleware.dehaze_hooks_middleware.interrupt_handler", stub),
            patch(
                "app.service.ai.middleware.dehaze_hooks_middleware.interrupt",
                lambda data: {"confirmed": False},
            ),
        ):
            result = await mw.awrap_tool_call(self._request(self._write_call()), handler)

        assert isinstance(result, ToolMessage)
        assert called == []
        assert result.additional_kwargs["_dehaze_status"] == 3
        assert "用户拒绝覆盖" in result.content
        _thread_id, itype, data = stub.saved[0]
        assert itype == "confirm"
        assert data["data"]["confirmKind"] == ConfirmKind.DANGEROUS_OP
        assert data["data"]["action"] == "write_conflict"
        assert data["data"]["previousWriter"] == "子AgentA"
        assert data["data"]["resource"] == _RESOURCE

    async def test_overwrite_confirmed_writes_and_releases(self):
        ctx = self._ctx()
        ctx["write_written_by"] = {_RESOURCE: "子AgentA"}
        mw = DehazeHooksMiddleware(ctx, subagent={"name": "子AgentB", "priority": 5})

        async def handler(request):
            return ToolMessage(content="written", tool_call_id="t1", name="write_file")

        with (
            self._compat_patch(),
            patch(
                "app.service.ai.middleware.dehaze_hooks_middleware.interrupt_handler",
                StubInterruptHandler(),
            ),
            patch(
                "app.service.ai.middleware.dehaze_hooks_middleware.interrupt",
                lambda data: {"confirmed": True},
            ),
        ):
            result = await mw.awrap_tool_call(self._request(self._write_call()), handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "written"
        assert ctx["write_holders"] == {}
        assert ctx["write_written_by"][_RESOURCE] == "子AgentB"
