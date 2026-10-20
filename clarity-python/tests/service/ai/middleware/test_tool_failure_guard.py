"""连续失败保护回归：跳过不计失败、缺失 conversation_id 时按调用处隔离。"""

from langchain_core.messages import ToolMessage

from app.service.ai.middleware.tool_failure_guard import ToolFailureGuardMiddleware


class _Req:
    def __init__(self, tool_call, state=None):
        self.tool_call = tool_call
        self.state = state if state is not None else {}


def _make(state=None, fail_limit=3):
    mw = ToolFailureGuardMiddleware(fail_limit=fail_limit)
    calls = {"ok": 0}

    async def handler(req):
        calls["ok"] += 1
        return ToolMessage(content="ok", tool_call_id="c1")

    return mw, calls, handler


def _skip_message():
    return ToolMessage(
        content="服务不可用", tool_call_id="c1", additional_kwargs={"_dehaze_status": 3}
    )


def _fail_message():
    return ToolMessage(content="执行失败", tool_call_id="c1", status="error")


async def _call(mw, handler, tool_name="search", state=None, result=None):
    async def _handler(req):
        return result if result is not None else await handler(req)

    return await mw.awrap_tool_call(
        _Req({"name": tool_name, "id": "c1"}, state or {"conversation_id": 1}), _handler
    )


class TestSkipNotCounted:
    async def test_skip_never_disables_tool(self):
        """status=3 是跳过（未真正执行），计入会误剔除可用工具。"""
        mw, _calls, handler = _make()
        for _ in range(5):
            msg = await _call(mw, handler, result=_skip_message())
            assert msg.content == "服务不可用"
        msg = await _call(mw, handler)
        assert msg.content == "ok"

    async def test_real_failure_disables_after_limit(self):
        mw, _calls, handler = _make()
        messages = [await _call(mw, handler, result=_fail_message()) for _ in range(3)]
        assert "临时禁用" in messages[-1].content
        blocked = await _call(mw, handler)
        assert "临时禁用" in blocked.content

    async def test_success_resets_counter(self):
        mw, _calls, handler = _make()
        await _call(mw, handler, result=_fail_message())
        await _call(mw, handler, result=_fail_message())
        await _call(mw, handler)
        msg = await _call(mw, handler, result=_fail_message())
        assert "临时禁用" not in msg.content

    async def test_dehaze_status_failure_counted(self):
        mw, _calls, handler = _make()
        for _ in range(3):
            result = ToolMessage(
                content="失败", tool_call_id="c1", additional_kwargs={"_dehaze_status": 2}
            )
            await _call(mw, handler, result=result)
        blocked = await _call(mw, handler)
        assert "临时禁用" in blocked.content


class TestIsolation:
    async def test_conversations_isolated(self):
        mw, _calls, handler = _make()
        for _ in range(3):
            await _call(mw, handler, state={"conversation_id": 1}, result=_fail_message())
        blocked = await _call(mw, handler, state={"conversation_id": 1})
        assert "临时禁用" in blocked.content
        other = await _call(mw, handler, state={"conversation_id": 2})
        assert other.content == "ok"

    async def test_missing_conversation_id_not_shared(self):
        """无 conversation_id 时若都归入 0 号桶，互不相关的调用会互相禁用工具。"""
        mw, _calls, handler = _make()
        for _ in range(3):
            await _call(mw, handler, state={"message_id": 11}, result=_fail_message())
        blocked = await _call(mw, handler, state={"message_id": 11})
        assert "临时禁用" in blocked.content
        other = await _call(mw, handler, state={"message_id": 12})
        assert other.content == "ok"

    async def test_before_agent_resets_bucket(self):
        mw, _calls, handler = _make()
        for _ in range(3):
            await _call(mw, handler, result=_fail_message())
        await mw.abefore_agent({"conversation_id": 1}, object())
        msg = await _call(mw, handler)
        assert msg.content == "ok"
