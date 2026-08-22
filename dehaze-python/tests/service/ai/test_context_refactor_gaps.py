from datetime import datetime, timedelta
from types import SimpleNamespace

from langchain_core.messages import AIMessage, SystemMessage

from app.service.ai import memory_injection as injection
from app.service.ai.reasoning_service import reasoning_service
from tests.stubs import (
    MemberBenefitRepo,
    NullDBSession,
    install_reasoning_chain_mocks,
    make_benefit,
    make_member,
    make_orm_mem,
    repo_returns,
)


class TestAlwaysOnLimit:
    async def test_inject_memories_passes_repository_preferences_through(self, monkeypatch):
        prefs = [
            make_orm_mem(
                i + 1, "semantic", f"偏好{i}", importance=100 - i, metadata={"is_preference": 1}
            )
            for i in range(25)
        ]

        async def _list_preferences(db, user_id, limit=20):
            return prefs

        async def _empty(*a, **k):
            return []

        async def _touch(db, mid):
            return None

        monkeypatch.setattr(injection.ai_memory_repository, "list_preferences", _list_preferences)
        monkeypatch.setattr(injection.ai_memory_repository, "list_by_skill", _empty)
        monkeypatch.setattr(injection.ai_memory_repository, "list_skills", _empty)
        monkeypatch.setattr(injection.ai_memory_repository, "search_by_keyword", _empty)
        monkeypatch.setattr(injection.ai_memory_repository, "touch", _touch)
        monkeypatch.setattr(injection, "search_memories", _empty)

        text, injected = await injection.inject_memories(object(), 1, "处理雾图")
        prefs_injected = [i for i in injected if i["source"] == "preference"]
        assert len(prefs_injected) == 25
        assert {i["memory_id"] for i in prefs_injected} == set(range(1, 26))
        assert text is not None
        assert "偏好0" in text and "偏好24" in text


class TestLayerFilterCompleteness:
    async def test_preference_query_filters_deleted_status_archived(self):
        from app.repository.ai_memory_repository import ai_memory_repository

        captured = {}

        class _Rows:
            def scalars(self):
                return self

            def all(self):
                return []

        class _DB:
            async def execute(self, stmt):
                captured["stmt"] = stmt
                return _Rows()

        await ai_memory_repository.list_preferences(_DB(), 1, limit=20)
        sql = str(captured["stmt"].compile(compile_kwargs={"literal_binds": True}))
        assert "deleted = 0" in sql
        assert "status = 1" in sql
        assert "archived = 0" in sql


class TestRecencyNumeric:
    def test_recency_decay_7days(self):
        t = datetime.now() - timedelta(days=7)
        score = injection._recency_score(t, t)
        assert abs(score - 0.7916) < 0.01

    def test_recency_decay_30days(self):
        t = datetime.now() - timedelta(days=30)
        score = injection._recency_score(t, t)
        assert abs(score - 0.3679) < 0.01


class TestMultimodalAccumulation:
    async def test_quota_accumulates_across_conversations(self, monkeypatch, mock_redis):
        import app.service.ai_artifact_service as mod

        monkeypatch.setattr(mod, "member_repository", MemberBenefitRepo(member=make_member("level_1")))
        monkeypatch.setattr(
            mod, "member_benefit_repository", MemberBenefitRepo(benefit=make_benefit(multimodal_limit=10))
        )

        key = "ai:multimodal:1:20260101"
        monkeypatch.setattr(
            mod.ai_artifact_service, "_visual_quota_key", staticmethod(lambda uid: key)
        )
        for _ in range(3):
            ok = await mod.ai_artifact_service._consume_visual_quota(mock_redis, 1, limit=10)
            assert ok is True

        used, limit = await mod.ai_artifact_service.check_visual_quota(None, mock_redis, 1)
        assert used == 3
        assert limit == 10


class TestSummaryNoReSummarize:
    async def test_watermark_advances_without_recompressing_old(self, monkeypatch):
        from app.service.ai.summary_service import summary_service

        all_rows = [
            SimpleNamespace(id=i, role="user" if i % 2 else "assistant", content=f"c{i}")
            for i in range(1, 61)
        ]

        async def _list_for_summary(db, conv_id, watermark):
            return [r for r in all_rows if r.id > watermark][::-1]

        monkeypatch.setattr(
            "app.service.ai.summary_service.ai_message_repository.list_for_summary",
            _list_for_summary,
        )

        first = await summary_service._load_messages_to_summarize(
            None, SimpleNamespace(id=1, summary_upto_message_id=0)
        )
        assert first[0]["id"] == 1 and first[-1]["id"] == 40

        second = await summary_service._load_messages_to_summarize(
            None, SimpleNamespace(id=1, summary_upto_message_id=40)
        )
        assert second == []


class TestUsedMemoryIdsE2E:
    async def test_run_extracts_injected_memory_ids_and_finalizes(self, monkeypatch):
        injected_list = [
            {"memory_id": 1, "memory_type": "semantic", "content": "偏好", "source": "preference"},
            {"memory_id": 2, "memory_type": "procedural", "content": "习惯", "source": "skill"},
            {"memory_id": None, "memory_type": "episodic", "content": "无id", "source": "retrieval"},
        ]
        service, recorder = install_reasoning_chain_mocks(monkeypatch, injected_list=injected_list)
        await service.run(conv_id=1, user_id=1, msg_id=1, model_id="m1", stream_session_id="s1")
        finalized = recorder["finalized"][0]
        assert (finalized[0], finalized[2], finalized[3]) == (1, "m1", [1, 2])

    async def test_finalize_writes_used_memory_ids_to_message(self, monkeypatch):
        class _Msg:
            used_memory_ids = None

        msg = _Msg()

        async def _calc(db, model_id, it, ot, cit):
            return 1

        monkeypatch.setattr("app.service.ai.reasoning_service.ai_message_repository", repo_returns(msg))
        monkeypatch.setattr("app.service.ai.reasoning_service.get_db_session", NullDBSession)
        monkeypatch.setattr("app.service.ai.reasoning_service.calculate_credits", _calc)

        result = {
            "final_response": "ok",
            "stop_reason": "stop",
            "usage": {"input_tokens": 5, "output_tokens": 3, "cached_input_tokens": 0},
        }
        await reasoning_service._finalize_message(1, result, "gpt", used_memory_ids=[1, 2])
        assert msg.used_memory_ids == [1, 2]


class TestScenePromptOnConversationCreate:
    def _make_form(self, scene=None, system_prompt=None):
        return SimpleNamespace(
            scene=scene,
            systemPrompt=system_prompt,
            agentCode=None,
            title="新对话",
            model=None,
            modelConfig=None,
            apiKeyId=None,
        )

    async def _create_and_capture(self, monkeypatch, form):
        from app.service.ai_conversation_service import ai_conversation_service

        captured = {}

        async def _resolve(db, agent_code):
            return "default", None

        class _Repo:
            async def create(self, db, conv):
                conv.id = 1
                conv.message_count = 0
                conv.pinned = 0
                conv.title_source = "auto"
                captured["system_prompt"] = conv.system_prompt
                captured["title"] = conv.title
                return conv

        monkeypatch.setattr(ai_conversation_service, "_resolve_agent_anchor", staticmethod(_resolve))
        monkeypatch.setattr(
            "app.service.ai_conversation_service.ai_conversation_repository", _Repo()
        )
        result = await ai_conversation_service.create_conversation(object(), 1, form)
        return result, captured

    async def test_create_conversation_writes_scene_prompt(self, monkeypatch):
        from app.service.ai.scene_templates import SCENE_IMAGE_DISPATCH

        form = self._make_form(scene="image_dispatch")
        _result, captured = await self._create_and_capture(monkeypatch, form)
        assert captured["system_prompt"] == SCENE_IMAGE_DISPATCH

    async def test_create_conversation_explicit_system_prompt_priority(self, monkeypatch):
        from app.service.ai.scene_templates import SCENE_MULTI_STEP

        form = self._make_form(scene="multi_step", system_prompt="自定义人设")
        _result, captured = await self._create_and_capture(monkeypatch, form)
        assert captured["system_prompt"] == "自定义人设"
        assert captured["system_prompt"] != SCENE_MULTI_STEP

    async def test_create_conversation_unknown_scene_falls_back_general(self, monkeypatch):
        from app.service.ai.scene_templates import SCENE_GENERAL

        form = self._make_form(scene="bogus_scene")
        _result, captured = await self._create_and_capture(monkeypatch, form)
        assert captured["system_prompt"] == SCENE_GENERAL


class TestConversationPromptInjection:
    async def test_conversation_prompt_injected_at_runtime_not_graph_key(self, monkeypatch):
        import app.service.ai.deep_agent_builder as builder_mod
        from app.service.ai.deep_agent_builder import DeepAgentBuilder
        from app.service.ai.prompt_composer import STABLE_SYSTEM_PROMPT

        captured = {}

        def _fake_create_deep_agent(**kwargs):
            captured["system_prompt"] = kwargs["system_prompt"]
            return object()

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)

        snapshot = {
            "system_prompt": "Agent 人设",
            "config": {
                "guardrails": {},
                "mcp_namespaces": [],
                "token_budget": 100,
                "max_steps": 1,
                "max_steps_react": 5,
            },
            "model_id": "m1",
            "name": "a1",
            "subagents": [],
        }
        await DeepAgentBuilder.build_from_snapshot(None, None, snapshot)
        built = captured["system_prompt"]
        assert built.startswith(STABLE_SYSTEM_PROMPT)
        assert "Agent 人设" in built
        assert "会话层提示词" not in built

    async def test_conversation_prompt_merged_into_system_message(self, monkeypatch):
        from app.service.ai.dehaze_hooks_middleware import DehazeHooksMiddleware

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
            "conversation_prompt": "请用 RIDCP 算法处理图像",
        }
        mw = DehazeHooksMiddleware(ctx)

        class _Hooks:
            async def run_hooks(self, point, state):
                return None

        monkeypatch.setattr("app.service.ai.dehaze_hooks_middleware.agent_hooks", _Hooks())

        class _Request:
            state = {"messages": []}
            system_message = SystemMessage(content="你是 dehaze 助手")

            def override(self, system_message=None):
                self.system_message = system_message
                return self

        class _Resp:
            def __init__(self):
                self.result = [AIMessage(content="ok")]

        seen = {}

        async def _handler(request):
            seen["system"] = request.system_message.content
            return _Resp()

        await mw.awrap_model_call(request=_Request(), handler=_handler)
        assert seen["system"] == "你是 dehaze 助手\n\n请用 RIDCP 算法处理图像"
        assert seen["system"].count("RIDCP") == 1


class TestBuildContextCallCount:
    async def test_run_builds_context_once(self, monkeypatch):
        service, recorder = install_reasoning_chain_mocks(monkeypatch)
        await service.run(conv_id=1, user_id=1, msg_id=1, model_id="m1", stream_session_id="s1")
        assert recorder["ctx_calls"] == 1
