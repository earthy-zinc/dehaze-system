from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.core.exceptions import BusinessException
from app.infrastructure.llm.call import llm_client as llm_client_mod
from app.service.ai.builders.deep_agent_builder import (
    DeepAgentBuilder,
    _build_agent_core,
    _build_subagents,
    _make_ctx,
)


def _minimal_snapshot():
    return {
        "name": "去雾助手",
        "description": "",
        "system_prompt": "你是图像去雾助手，请直接回答。",
        "model_id": "test-model",
        "reasoning_mode": "direct",
        "is_subagent": 0,
        "is_team": 0,
        "is_exposed": 1,
        "mcp_namespaces": [],
        "skills": [],
        "subagents": [],
        "config": {
            "max_steps": 3,
            "max_steps_react": 3,
            "max_steps_plan": 3,
            "max_steps_reflexion": 3,
            "token_budget": 50000,
            "guardrails": {"prompt_injection": {"enabled": False}},
        },
    }


@pytest.fixture(autouse=True)
def _mock_llm_stream(monkeypatch):
    async def fake_stream_chat(db, model_id, messages, **kw):
        yield SimpleNamespace(type="text_delta", content="去雾完成，图像已清晰。")
        yield SimpleNamespace(type="done", usage={"input_tokens": 10, "output_tokens": 5})

    monkeypatch.setattr(llm_client_mod.llm_client, "stream_chat", fake_stream_chat)

    @asynccontextmanager
    async def _session():
        yield object()

    monkeypatch.setattr("app.infrastructure.llm.client.dehaze_chat_model.get_db_session", _session)


class TestMakeCtx:
    def test_ctx_uses_react_default_and_budget(self):
        ctx = _make_ctx(_minimal_snapshot(), _minimal_snapshot()["config"])
        assert ctx["max_steps"] == 3
        assert ctx["token_budget"] == 50000
        assert ctx["step_count"] == 0
        assert ctx["_model_id"] == "test-model"


class TestBuildSubagents:
    async def test_subagent_failure_raises_not_silently_skips(self, monkeypatch):
        """子 Agent 配置错误不得静默跳过，须聚合所有失败项一并抛错。"""
        snapshot = {
            "subagents": [
                {"agent_id": 1001},
                {"agent_id": 1002},
            ]
        }

        async def _no_snapshot(db, agent_id, version_no=None):
            return None

        monkeypatch.setattr(
            "app.service.ai.builders.deep_agent_builder.ai_agent_version_repository",
            SimpleNamespace(get_published_snapshot=_no_snapshot, resolve_snapshot=lambda s: dict(s)),
        )

        with pytest.raises(BusinessException) as ei:
            await _build_subagents(object(), object(), snapshot, {})

        assert "子 Agent 1001" in ei.value.message
        assert "子 Agent 1002" in ei.value.message


class TestBuildAgentCore:
    def test_core_builds_model_tools_middleware(self):
        ctx = {
            "max_steps": 3,
            "token_budget": 50000,
            "token_used": 0,
            "step_count": 0,
            "task_type": "",
            "task_algorithm": "",
            "task_params": {},
            "task_status": "",
            "task_id": "",
            "task_artifacts": [],
            "_model_id": "test-model",
        }
        core = _build_agent_core(_minimal_snapshot(), ctx)
        assert "model" in core and "tools" in core and "middleware" in core
        assert isinstance(core["tools"], list)


class TestE2E:
    async def test_build_and_invoke_produces_final_response(self):
        snapshot = _minimal_snapshot()

        graph = await DeepAgentBuilder.build_from_snapshot(
            object(), object(), snapshot, checkpointer=None
        )

        initial_state = {
            "messages": [{"role": "user", "content": "请处理这张图"}],
            "user_id": None,
            "conversation_id": 0,
            "message_id": 0,
            "model_id": snapshot["model_id"],
            "system_prompt": snapshot["system_prompt"],
            "stream_session_id": "e2e:1",
            "step_count": 0,
            "token_used": 0,
            "token_budget": 50000,
            "thoughts": [],
            "isolated_token_pool": True,
        }
        result = await graph.ainvoke(
            initial_state, config={"configurable": {"thread_id": "e2e:t1"}}
        )

        assert result.get("final_response"), "final_response 缺失"
        assert "去雾完成" in result["final_response"]

    async def test_extract_tool_sequence_from_graph_output(self):
        from app.service.ai.service.eval_runner import _extract_tool_sequence

        with_tools = {
            "messages": [],
            "thoughts": [
                {"tool_name": "recommend_algorithm"},
                {"name": "process_batch"},
            ],
        }
        assert _extract_tool_sequence(with_tools) == ["recommend_algorithm", "process_batch"]

        msg = SimpleNamespace(
            tool_calls=[{"name": "lookup_tool", "args": {}, "id": "c1"}]
        )
        without_thoughts = {"messages": [msg], "thoughts": []}
        assert _extract_tool_sequence(without_thoughts) == ["lookup_tool"]
