"""停止语义与异步任务取消回归

- reasoning_service.stop：置协作式停止标志 + 清理流/中断点 + 取消批量任务
- async_resume.cancel_task：先清反查键再取消，回调静默跳过
- execute_code Shell 确认：中断点须持久化，否则 resume 端点无法恢复
"""

import asyncio

import pytest

from app.service.ai.builders.dehaze_tools_builder import build_business_tools
from app.service.ai.middleware import async_resume
from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.middleware.interrupt_handler import interrupt_handler
from app.service.ai.middleware.run_context import set_run_ctx
from app.service.ai.service import reasoning_service as rs_mod


async def test_stop_sets_flag_and_cleans_up(mock_redis, monkeypatch):
    calls = {"stop_stream": [], "update_status": [], "cancel_task": [], "clear_interrupt": []}

    async def _stop_stream(sid):
        calls["stop_stream"].append(sid)

    async def _update_status(db, msg_id, status, error=None):
        calls["update_status"].append((msg_id, status))

    async def _cancel_task(task_id):
        calls["cancel_task"].append(task_id)

    async def _clear_interrupt(thread_id):
        calls["clear_interrupt"].append(thread_id)

    monkeypatch.setattr(rs_mod.sse_emitter_manager, "stop_stream", _stop_stream)
    monkeypatch.setattr(rs_mod.ai_message_repository, "update_status", _update_status)
    monkeypatch.setattr(rs_mod.async_resume, "cancel_task", _cancel_task)
    monkeypatch.setattr(rs_mod.interrupt_handler, "clear_interrupt", _clear_interrupt)

    await rs_mod.reasoning_service.stop(1, 2, "s1")

    assert calls["stop_stream"] == ["s1"]
    assert calls["update_status"] == [(2, 4)]
    assert calls["cancel_task"] == ["s1"]
    assert calls["clear_interrupt"] == ["1:2"]
    # 协作式停止标志已落 Redis：推理循环据此在事件间隙中断收尾
    flag = await mock_redis.get(rs_mod.reasoning_service._stop_key(1, 2))
    assert flag in ("1", b"1")


async def test_cancel_task_clears_mapping_and_cancels():
    task_id = "batch:1:2:42"

    async def _forever():
        await asyncio.sleep(3600)

    task = asyncio.get_running_loop().create_task(_forever())
    async_resume._running_tasks[task_id] = task
    await async_resume._save_task_mapping(task_id, {"thread_id": "1:2"})

    await async_resume.cancel_task(task_id)

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert task_id not in async_resume._running_tasks
    # 反查键先清：任务 finally 的完成回调静默跳过，不会对已取消消息续流
    assert await async_resume._load_task_mapping(task_id) is None


async def test_execute_code_shell_confirm_persists_interrupt(mock_redis, monkeypatch):
    """危险操作确认与算法推荐一致：中断点须落 Redis，否则 resume 端点查不到中断"""
    template = {
        "max_steps": 20,
        "token_budget": 50000,
        "tool_timeout": 60,
        "retry_max": 2,
        "conversation_id": 1,
        "message_id": 2,
        "user_id": 10,
        "stream_session_id": "s1",
        "model_id": "m",
    }
    tools = build_business_tools(template)
    tool = next(t for t in tools if t.name == "execute_code")

    saved = {}
    real_save = interrupt_handler.save_interrupt

    async def _save(thread_id, interrupt_type, data):
        saved["thread_id"] = thread_id
        saved["type"] = interrupt_type
        await real_save(thread_id, interrupt_type, data)

    # langgraph interrupt() 为同步调用（返回拒绝确认载荷）
    def _sync_interrupt(data):
        return {"confirmed": False}

    monkeypatch.setattr(interrupt_handler, "save_interrupt", _save)
    monkeypatch.setattr("app.service.ai.builders.dehaze_tools_builder.interrupt", _sync_interrupt)

    result = await tool.ainvoke({"code": "echo hello", "language": "shell"})

    assert result == "用户拒绝了该 Shell 命令的执行"
    assert saved["thread_id"] == "1:2"
    assert saved["type"] == "confirm"
    stored = await interrupt_handler.get_interrupt("1:2")
    assert stored is not None
    assert stored["type"] == "confirm"
    assert stored["data"]["data"]["action"] == "execute_shell_command"
    # resume 端点依赖 data 中的 stream_session_id 续流
    assert stored["data"]["stream_session_id"] == "s1"


async def test_abefore_agent_run_ctx_wiring():
    """中间件 ctx 属性在 run 外回退模板（直调兼容），run 内指向 run 上下文"""
    template = {"token_budget": 500, "max_steps": 20}
    mw = DehazeHooksMiddleware(template)
    assert mw.ctx is template  # run 外回退模板

    set_run_ctx({"token_budget": 500, "max_steps": 20, "conversation_id": 1})
    assert mw.ctx["conversation_id"] == 1  # run 内指向 run 上下文
