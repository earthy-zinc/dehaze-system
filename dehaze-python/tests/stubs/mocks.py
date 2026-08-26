"""patch 辅助：对外围"外部世界"边界打桩的 monkeypatch 工具。

本模块只放 patch 辅助函数（mocks），内部若依赖假件/工厂，统一从
tests.stubs.fakes / tests.stubs.factories 导入，禁止在此内联重复定义。
"""

from tests.stubs.fakes import FakeGraph, StubInterruptHandler
from tests.stubs.factories import make_conv


def patch_reasoning_boundaries(
    monkeypatch,
    *,
    interrupt=None,
    values=None,
    snapshot=None,
):
    """只替身推理链的"外部世界"边界，内部逻辑与仓储全部走真实代码：

    - Agent 数据/配置加载（_load_agent_anchor/_load_snapshot）与 LLM 图执行
      （_build_graph → FakeGraph）为外部依赖 → 固定返回值
    - 中断存储（interrupt_handler）、SSE 发射器（sse_emitter_manager）为外部
      基础设施 → 记录型桩
    - ES 记忆检索（search_memories）为外部依赖 → 空结果
    - 后台/外部动作（步骤摘要、建议生成、会话 ES 同步）→ no-op

    会话/消息仓储、context_manager.build_context、_finalize_message、
    _push_end 等内部逻辑保持真实执行（配合 db fixture 落库，断言业务结果）。

    Args:
        interrupt: StubInterruptHandler.get_interrupt 的返回值（None 即无挂起）。
        values: FakeGraph 最终 state.values（final_response/stop_reason/usage）。

    Returns:
        (reasoning_service, emitter, interrupt_handler)
        - emitter.released: release_lock(conv_id) 调用列表（挂起让渡并发锁）
        - emitter.events: [(event_type, data), ...] 按序 SSE 事件
    """
    from app.service.ai.builders.deep_agent_builder import DeepAgentBuilder
    from app.service.ai.service.reasoning_service import reasoning_service

    class _Emitter:
        def __init__(self):
            self.events = []
            self.released = []

        async def send_event(self, stream_session_id, event_type, data):
            self.events.append((event_type, data))

        async def release_lock(self, conv_id):
            self.released.append(conv_id)

    default_values = {
        "final_response": "ok",
        "stop_reason": "stop",
        "usage": {"input_tokens": 5, "output_tokens": 3, "cached_input_tokens": 0},
    }
    graph = FakeGraph(values if values is not None else default_values)

    async def _load_agent_anchor(db, conv):
        return 1, 1

    async def _load_snapshot(db, redis, agent_id, version_no):
        return snapshot if snapshot is not None else {"reasoning_mode": "react", "config": {"max_steps": 10}}

    async def _build_graph(db, redis, agent_id, version_no, model_id=None):
        return graph

    async def _resolve(snapshot, messages, model_id):
        return "react", 10

    monkeypatch.setattr(reasoning_service, "_load_agent_anchor", _load_agent_anchor)
    monkeypatch.setattr(reasoning_service, "_load_snapshot", _load_snapshot)
    monkeypatch.setattr(reasoning_service, "_build_graph", _build_graph)
    monkeypatch.setattr(DeepAgentBuilder, "resolve_reasoning_mode", _resolve)

    handler = StubInterruptHandler(interrupt)
    monkeypatch.setattr("app.service.ai.service.reasoning_service.interrupt_handler", handler)

    emitter = _Emitter()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.sse_emitter_manager", emitter)

    async def _empty_search(*a, **k):
        return []

    monkeypatch.setattr("app.service.ai.service.memory_injection.search_memories", _empty_search)

    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service.schedule_step_summaries", lambda *a, **k: None
    )
    # _trigger_suggestions 为实例静态方法，patch 到单例实例属性
    monkeypatch.setattr(reasoning_service, "_trigger_suggestions", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.service.ai.service.reasoning_service._schedule_conversation_sync", lambda *a, **k: None
    )
    return reasoning_service, emitter, handler
