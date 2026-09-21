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
from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.strategies.agent_config_resolver import REASONING_DEFAULTS


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
            **REASONING_DEFAULTS,
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
            SimpleNamespace(
                get_published_snapshot=_no_snapshot, resolve_snapshot=lambda s: dict(s)
            ),
        )

        with pytest.raises(BusinessException) as ei:
            await _build_subagents(object(), object(), snapshot)

        assert "子 Agent 1001" in ei.value.message
        assert "子 Agent 1002" in ei.value.message


def _sub_snapshot(name: str = "子去雾助手") -> dict:
    return {**_minimal_snapshot(), "name": name, "is_subagent": 1}


class TestLocalSubagentPath:
    """本地子 Agent（endpoint_id 为假）路径回归：曾因 _build_agent_core 缺 subagent
    形参而在 _build_subagents 抛 TypeError，且即便透传形参也须真正到达中间件 holder。"""

    def _patch_repo(self, monkeypatch, sub_snapshot: dict):
        async def _published(db, agent_id, version_no=None):
            return SimpleNamespace(snapshot=sub_snapshot)

        monkeypatch.setattr(
            "app.service.ai.builders.deep_agent_builder.ai_agent_version_repository",
            SimpleNamespace(
                get_published_snapshot=_published,
                resolve_snapshot=lambda s: dict(s),
            ),
        )

    async def test_local_subagent_core_built_without_typeerror(self, monkeypatch):
        """endpoint_id 为假 + 有效已发布快照 ⇒ 不再抛 TypeError 且构建出子 Agent core。"""
        self._patch_repo(monkeypatch, _sub_snapshot())
        snapshot = {"subagents": [{"agent_id": 2001, "priority": 0}]}

        subagents, remote_tools = await _build_subagents(object(), object(), snapshot)

        assert remote_tools == []
        assert len(subagents) == 1
        sub = subagents[0]
        assert sub["name"] == "子去雾助手"
        # SubAgent 的 model 为 NotRequired 键，先收窄再取值
        assert "model" in sub
        assert sub["model"] is not None

    async def test_subagent_identity_reaches_middleware_holder(self, monkeypatch):
        """子 Agent 名字/优先级须真正到达中间件 holder（写冲突仲裁的身份来源）。"""
        self._patch_repo(monkeypatch, _sub_snapshot("仲裁子Agent"))
        snapshot = {"subagents": [{"agent_id": 2002, "priority": 7}]}

        subagents, _ = await _build_subagents(object(), object(), snapshot)

        sub = subagents[0]
        # SubAgent 的 middleware 为 NotRequired 键，先收窄再遍历
        assert "middleware" in sub
        hooks_mw = next(mw for mw in sub["middleware"] if isinstance(mw, DehazeHooksMiddleware))
        assert hooks_mw._holder["name"] == "仲裁子Agent"
        assert hooks_mw._holder["priority"] == 7

    async def test_build_from_snapshot_with_local_subagent(self, monkeypatch):
        """端到端：带本地子 Agent 的快照可正常构图（不再因 TypeError 崩）。"""
        self._patch_repo(monkeypatch, _sub_snapshot())
        snapshot = {**_minimal_snapshot(), "subagents": [{"agent_id": 2003, "priority": 2}]}

        graph = await DeepAgentBuilder.build_from_snapshot(
            object(), object(), snapshot, checkpointer=None
        )

        assert graph is not None


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
        assert "model" in core
        assert "tools" in core
        assert "middleware" in core
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

        msg = SimpleNamespace(tool_calls=[{"name": "lookup_tool", "args": {}, "id": "c1"}])
        without_thoughts = {"messages": [msg], "thoughts": []}
        assert _extract_tool_sequence(without_thoughts) == ["lookup_tool"]
