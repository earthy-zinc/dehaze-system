from app.service.ai import deep_agent_builder
from app.service.ai.deep_agent_builder import DeepAgentBuilder


def _snapshot(mode, **config):
    base = {
        "model_id": "gpt-4o-mini",
        "reasoning_mode": mode,
        "config": {
            "max_steps": None,
            "max_steps_react": 10,
            "max_steps_plan": 20,
            "max_steps_reflexion": 30,
            "max_iterations_reflexion": 3,
            "reflexion_threshold": 0.8,
            "token_budget": 10000,
            **config,
        },
    }
    return base


def _mock_eval(monkeypatch, mode):
    async def _eval(state):
        return {"complexity": mode, "reasoning_mode": mode, "usage": {}}

    monkeypatch.setattr(deep_agent_builder, "evaluate_complexity", _eval)


async def test_fixed_react_uses_max_steps_react():
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("react"), [{"role": "user", "content": "去雾这张图"}], "gpt-4o-mini"
    )
    assert mode == "react"
    assert max_steps == 10


async def test_fixed_plan_execute_uses_max_steps_plan():
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("plan_execute"), [{"role": "user", "content": "批量处理"}], "gpt-4o-mini"
    )
    assert mode == "plan_execute"
    assert max_steps == 20


async def test_fixed_reflexion_uses_max_steps_reflexion():
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("reflexion"), [{"role": "user", "content": "审查报告"}], "gpt-4o-mini"
    )
    assert mode == "reflexion"
    assert max_steps == 30


async def test_fixed_direct_uses_max_steps_one():
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("direct"), [{"role": "user", "content": "你好"}], "gpt-4o-mini"
    )
    assert mode == "direct"
    assert max_steps == 1


async def test_auto_l2_resolves_to_plan_execute(monkeypatch):
    _mock_eval(monkeypatch, "plan_execute")
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("auto"), [{"role": "user", "content": "批量处理这些图片"}], "gpt-4o-mini"
    )
    assert mode == "plan_execute"
    assert max_steps == 20


async def test_auto_l0_resolves_to_direct(monkeypatch):
    _mock_eval(monkeypatch, "direct")
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("auto"), [{"role": "user", "content": "你好"}], "gpt-4o-mini"
    )
    assert mode == "direct"
    assert max_steps == 1


async def test_auto_l3_resolves_to_reflexion(monkeypatch):
    _mock_eval(monkeypatch, "reflexion")
    mode, max_steps = await DeepAgentBuilder.resolve_reasoning_mode(
        _snapshot("auto"), [{"role": "user", "content": "确保输出符合规范"}], "gpt-4o-mini"
    )
    assert mode == "reflexion"
    assert max_steps == 30


async def test_direct_skips_graph_build(monkeypatch):
    from contextlib import asynccontextmanager

    from app.service.ai import reasoning_service

    class _Conv:
        system_prompt = None

    @asynccontextmanager
    async def _fake_session():
        yield object()

    async def _get_conv(db, cid, uid):
        return _Conv()

    async def _compress(db, conv, model):
        return None

    async def _build_ctx(db, conv, model):
        return ([{"role": "user", "content": "你好"}], "sys", [])

    async def _load_anchor(self, db, conv):
        return (1, 1)

    async def _load_snap(self, db, redis, a, v):
        return _snapshot("direct")

    async def _resolve(snapshot, messages, model_id):
        return "direct", 1

    state = {"direct_called": False, "graph_called": False}

    async def _run_direct(
        self, conv_id, user_id, msg_id, model_id, stream_session_id, messages, system_prompt
    ):
        state["direct_called"] = True
        return {"final_response": "hi", "stop_reason": "stop", "usage": {}}

    def _build_graph(self, db, redis, agent_id, version_no, model_id=None):
        state["graph_called"] = True
        raise AssertionError("direct 路径不应构建图")

    monkeypatch.setattr(reasoning_service, "get_db_session", _fake_session)
    monkeypatch.setattr(reasoning_service.ai_conversation_repository, "get_by_id_and_user", _get_conv)
    monkeypatch.setattr(reasoning_service.summary_service, "maybe_compress", _compress)
    monkeypatch.setattr(reasoning_service.context_manager, "build_context", _build_ctx)
    monkeypatch.setattr(reasoning_service.ReasoningService, "_load_agent_anchor", _load_anchor)
    monkeypatch.setattr(reasoning_service.ReasoningService, "_load_snapshot", _load_snap)
    monkeypatch.setattr(DeepAgentBuilder, "resolve_reasoning_mode", _resolve)
    monkeypatch.setattr(reasoning_service.ReasoningService, "_run_direct", _run_direct)
    monkeypatch.setattr(reasoning_service.ReasoningService, "_build_graph", _build_graph)

    result = await reasoning_service.reasoning_service.run(
        conv_id=1, user_id=2, msg_id=3, model_id="gpt-4o-mini", stream_session_id="s1"
    )
    assert state["direct_called"]
    assert not state["graph_called"]
    assert result["stop_reason"] == "stop"
