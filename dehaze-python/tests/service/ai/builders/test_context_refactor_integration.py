from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_memory import SysAiMemory
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.entity.sys_ai_model import SysAiModel
from app.models.entity.sys_ai_provider import SysAiProvider
from app.repository.member_benefit_repository import MemberBenefitRepository
from app.repository.member_repository import MemberRepository
from app.service.ai.builders.context_manager import context_manager
from app.service.ai.service import memory_injection as injection
from app.service.ai.service import trace_collector
from app.service.ai.service.reasoning_service import reasoning_service
from app.service.ai.service.summary_service import summary_service
from app.service.ai.strategies.agent_config_resolver import REASONING_DEFAULTS
from tests.stubs.factories import make_benefit, make_member, make_orm_mem
from tests.stubs.fakes import LLMChunk, MemberBenefitRepo
from tests.stubs.mocks import patch_reasoning_boundaries

pytestmark = pytest.mark.requires_db


class _FakeGraph:
    """create_deep_agent 的替身：真实产物 CompiledStateGraph 可被弱引用，故用可弱引用对象。"""


@pytest.fixture(autouse=True)
def _reset_trace_collector():
    """窗口测试会启动采集器记录截断事件，结束后重置防泄漏到后续用例"""
    yield
    trace_collector._current_collector.set(None)


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

        text, injected = await injection.inject_memories(
            AsyncMock(spec=AsyncSession), 1, "处理雾图"
        )
        prefs_injected = [i for i in injected if i["source"] == "preference"]
        assert len(prefs_injected) == 25
        assert {i["memory_id"] for i in prefs_injected} == set(range(1, 26))
        assert text is not None
        assert "偏好0" in text
        assert "偏好24" in text


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

        await ai_memory_repository.list_preferences(
            cast(AsyncSession, _DB()),  # 替身：_DB 仅实现 execute，无法子类化 AsyncSession
            1,
            limit=20,
        )
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
    async def test_quota_accumulates_across_conversations(self, mock_redis):
        import app.service.ai_artifact_service as mod

        quota = MemberBenefitRepo(
            member=make_member("level_1"), benefit=make_benefit(multimodal_limit=10)
        )
        # 测试替身：同一对象仅实现 get_by_user_id / get_by_level_code，兼满足会员与权益仓储契约
        svc = mod.AiArtifactService(
            member_repository=cast(MemberRepository, quota),  # 替身：tests/stubs 的结构型桩
            member_benefit_repository=cast(MemberBenefitRepository, quota),  # 替身：同上
        )

        key = "ai:multimodal:1:20260101"
        svc._visual_quota_key = staticmethod(lambda user_id: key)
        for _ in range(3):
            ok = await svc._consume_visual_quota(mock_redis, 1, limit=10)
            assert ok is True

        used, limit = await svc.check_visual_quota(AsyncMock(spec=AsyncSession), mock_redis, 1)
        assert used == 3
        assert limit == 10


class TestSummaryNoReSummarize:
    async def test_watermark_advances_without_recompressing_old(self, monkeypatch):
        from app.service.ai.service.summary_service import summary_service

        all_rows = [
            SimpleNamespace(id=i, role="user" if i % 2 else "assistant", content=f"c{i}")
            for i in range(1, 61)
        ]

        async def _get_chain(db, conv_id, start_id, limit=None, max_hops=200):
            return all_rows

        monkeypatch.setattr(
            "app.service.ai.service.summary_service.ai_message_repository.get_chain_by_id",
            _get_chain,
        )

        # 窗口起点 41（40 条已被 token 预算裁剪出窗口）：水位 0 → 摘要 1..40
        first = await summary_service._load_messages_to_summarize(
            AsyncMock(spec=AsyncSession),
            SimpleNamespace(id=1, summary_upto_message_id=0, current_branch_message_id=60),
            window_start_id=41,
        )
        assert first[0]["id"] == 1
        assert first[-1]["id"] == 40

        # 水位推进到 40 后：窗口外无未摘要消息 → 不再重复压缩
        second = await summary_service._load_messages_to_summarize(
            AsyncMock(spec=AsyncSession),
            SimpleNamespace(id=1, summary_upto_message_id=40, current_branch_message_id=60),
            window_start_id=41,
        )
        assert second == []

        # 窗口覆盖全链（起点为链首消息）：无窗口外消息，不触发压缩
        third = await summary_service._load_messages_to_summarize(
            AsyncMock(spec=AsyncSession),
            SimpleNamespace(id=1, summary_upto_message_id=0, current_branch_message_id=60),
            window_start_id=1,
        )
        assert third == []


class TestUsedMemoryIdsE2E:
    @staticmethod
    async def _seed_run_ctx(db):
        """落库真实会话 + 用户消息 + 待生成的 assistant 消息（status=1 生成中）"""
        conv = SysAiConversation(user_id=1, model="m1")
        db.add(conv)
        await db.flush()
        user_msg = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=None,
            role="user",
            content="处理雾图",
            status=2,
        )
        db.add(user_msg)
        await db.flush()
        asst_msg = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=user_msg.id,
            role="assistant",
            content="",
            status=1,
        )
        db.add(asst_msg)
        await db.flush()
        conv.current_branch_message_id = asst_msg.id
        await db.flush()
        return conv, asst_msg

    async def test_run_extracts_injected_memory_ids_and_finalizes(self, db, monkeypatch):
        conv, asst_msg = await self._seed_run_ctx(db)
        # 常驻偏好记忆（is_preference=1，按重要性倒序注入）
        prefs = [
            SysAiMemory(
                user_id=1,
                memory_type="semantic",
                content="偏好",
                metadata_={"is_preference": 1},
                importance=90,
            ),
            SysAiMemory(
                user_id=1,
                memory_type="semantic",
                content="习惯",
                metadata_={"is_preference": 1},
                importance=70,
            ),
        ]
        db.add_all(prefs)
        await db.flush()

        service, _, _ = patch_reasoning_boundaries(monkeypatch)
        await service.run(
            conv_id=conv.id, user_id=1, msg_id=asst_msg.id, model_id="m1", stream_session_id="s1"
        )

        # 业务结果：build_context 真实注入的记忆可见性落库（含真实记忆 ID），消息完成
        await db.refresh(asst_msg)
        assert asst_msg.used_memory_ids == [prefs[0].id, prefs[1].id]
        assert asst_msg.status == 2

    async def test_finalize_writes_used_memory_ids_to_message(self, db):
        msg = SysAiMessage(conversation_id=1, role="assistant", content="", status=1)
        db.add(msg)
        await db.flush()

        result = {
            "final_response": "ok",
            "stop_reason": "stop",
            "usage": {"input_tokens": 5, "output_tokens": 3, "cached_input_tokens": 0},
        }
        await reasoning_service._finalize_message(msg.id, result, "gpt", used_memory_ids=[1, 2])
        await db.refresh(msg)
        assert msg.used_memory_ids == [1, 2]
        assert msg.status == 2
        assert msg.content == "ok"
        assert msg.input_tokens == 5
        assert msg.output_tokens == 3


class TestScenePromptOnConversationCreate:
    def _make_form(self, scene=None, system_prompt=None):
        return SimpleNamespace(
            scene=scene,
            systemPrompt=system_prompt,
            agentCode=None,
            title="新对话",
            model=None,
            modelConfig=None,
            suggestionsEnabled=True,
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

        monkeypatch.setattr(
            ai_conversation_service, "_resolve_agent_anchor", staticmethod(_resolve)
        )
        monkeypatch.setattr(ai_conversation_service, "ai_conversation_repository", _Repo())
        result = await ai_conversation_service.create_conversation(
            AsyncMock(spec=AsyncSession), 1, form
        )
        return result, captured

    async def test_create_conversation_writes_scene_prompt(self, monkeypatch):
        from app.service.ai.strategies.scene_templates import SCENE_IMAGE_DISPATCH

        form = self._make_form(scene="image_dispatch")
        _result, captured = await self._create_and_capture(monkeypatch, form)
        assert captured["system_prompt"] == SCENE_IMAGE_DISPATCH

    async def test_create_conversation_explicit_system_prompt_priority(self, monkeypatch):
        from app.service.ai.strategies.scene_templates import SCENE_MULTI_STEP

        form = self._make_form(scene="multi_step", system_prompt="自定义人设")
        _result, captured = await self._create_and_capture(monkeypatch, form)
        assert captured["system_prompt"] == "自定义人设"
        assert captured["system_prompt"] != SCENE_MULTI_STEP

    async def test_create_conversation_unknown_scene_falls_back_general(self, monkeypatch):
        from app.service.ai.strategies.scene_templates import SCENE_GENERAL

        form = self._make_form(scene="bogus_scene")
        _result, captured = await self._create_and_capture(monkeypatch, form)
        assert captured["system_prompt"] == SCENE_GENERAL


class TestConversationPromptInjection:
    async def test_conversation_prompt_injected_at_runtime_not_graph_key(self, monkeypatch):
        import app.service.ai.builders.deep_agent_builder as builder_mod
        from app.service.ai.builders.deep_agent_builder import DeepAgentBuilder
        from app.service.ai.strategies.prompt_composer import STABLE_SYSTEM_PROMPT

        captured = {}

        def _fake_create_deep_agent(**kwargs):
            captured["system_prompt"] = kwargs["system_prompt"]
            return _FakeGraph()

        monkeypatch.setattr(builder_mod, "create_deep_agent", _fake_create_deep_agent)

        snapshot = {
            "system_prompt": "Agent 人设",
            "config": {
                **REASONING_DEFAULTS,
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
        from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware

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

        monkeypatch.setattr(
            "app.service.ai.middleware.dehaze_hooks_middleware.agent_hooks", _Hooks()
        )

        request = ModelRequest(
            model=FakeListChatModel(responses=["ok"]),
            messages=[],
            system_message=SystemMessage(content="你是 dehaze 助手"),
        )
        seen: dict[str, str] = {}

        async def _handler(req):
            seen["system"] = req.system_message.content
            return ModelResponse(result=[AIMessage(content="ok")])

        await mw.awrap_model_call(request=request, handler=_handler)
        assert seen["system"] == "你是 dehaze 助手\n\n请用 RIDCP 算法处理图像"
        assert seen["system"].count("RIDCP") == 1


class TestBuildContextCallCount:
    async def test_run_builds_context_once(self, db, monkeypatch):
        conv, asst_msg = await TestUsedMemoryIdsE2E._seed_run_ctx(db)
        # 单方法 spy：真实执行 build_context，仅计数（验证记忆注入 touch 无重复副作用）
        calls = []
        real_build = context_manager.build_context

        async def _spy(dbs, c, model_id):
            calls.append(1)
            return await real_build(dbs, c, model_id)

        monkeypatch.setattr(context_manager, "build_context", _spy)
        service, _, _ = patch_reasoning_boundaries(monkeypatch)
        await service.run(
            conv_id=conv.id, user_id=1, msg_id=asst_msg.id, model_id="m1", stream_session_id="s1"
        )
        await db.refresh(asst_msg)
        assert len(calls) == 1  # 单次发送仅执行一次 build_context
        assert asst_msg.status == 2  # 且整条链路真实完成落库


def _llm_stream(*chunks):
    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c

    return _gen


class TestTokenBudgetWindow:
    """上下文窗口按模型 token 预算裁剪（替代旧固定 20 条截断），截断事件写过程链"""

    @staticmethod
    async def _seed_small_model(db) -> str:
        provider = SysAiProvider(
            provider_code=f"winprov-{uuid4().hex[:8]}",
            display_name="窗口测试供应商",
            api_base_url="http://localhost:9/v1",
            protocol_type="openai_compat",
            auth_type="bearer",
            status=1,
        )
        db.add(provider)
        await db.flush()
        model = SysAiModel(
            provider_id=provider.id,
            model_id="win-m1",
            display_name="窗口测试模型",
            max_context_tokens=4000,
            max_output_tokens=500,
            status=1,
        )
        db.add(model)
        await db.flush()
        return model.model_id

    @staticmethod
    async def _seed_chain(db, conv_id: int, count: int, content: str) -> list[SysAiMessage]:
        """落库 count 条首尾相接的长消息（每条 1000 token）"""
        msgs = []
        parent = None
        for i in range(count):
            msg = SysAiMessage(
                conversation_id=conv_id,
                parent_message_id=parent,
                role="user" if i % 2 == 0 else "assistant",
                content=content,
                model="win-m1",
                status=2,
            )
            db.add(msg)
            await db.flush()
            msgs.append(msg)
            parent = msg.id
        return msgs

    async def test_window_trims_oldest_within_budget_and_records_event(self, db, monkeypatch):
        model_id = await self._seed_small_model(db)
        conv = SysAiConversation(user_id=1, model=model_id)
        db.add(conv)
        await db.flush()
        # 6 条 × 4000 字符（约 1000 token）：总 6000 token 超出预算（≈1300），从最早裁剪
        msgs = await self._seed_chain(db, conv.id, 6, "x" * 4000)
        conv.current_branch_message_id = msgs[-1].id
        await db.flush()

        monkeypatch.setattr(
            "app.service.ai.builders.context_manager.inject_memories",
            AsyncMock(return_value=(None, [])),
        )
        collector = trace_collector.start(
            conversation_id=conv.id, message_id=None, user_id=1, agent_code=None, model_id=model_id
        )
        messages, _system_prompt, _injected = await context_manager.build_context(
            db, conv, model_id
        )

        # 最早消息被裁出窗口，最新消息保留（本轮输入不空）
        assert len(messages) < 6
        assert messages[-1]["id"] == msgs[-1].id
        assert messages[0]["id"] > msgs[0].id
        # 截断事件写过程链（消费可观测性：前后 token 与裁剪条数）
        truncate_events = [e for e in collector.context_events if e.get("event") == "truncate"]
        assert len(truncate_events) == 1
        assert truncate_events[0]["count"] == 6 - len(messages)
        assert truncate_events[0]["before_tokens"] > truncate_events[0]["after_tokens"]

    async def test_window_keeps_all_messages_when_within_budget(self, db, monkeypatch):
        model_id = await self._seed_small_model(db)
        conv = SysAiConversation(user_id=1, model=model_id)
        db.add(conv)
        await db.flush()
        msgs = await self._seed_chain(db, conv.id, 3, "y" * 100)  # 总量远小于预算
        conv.current_branch_message_id = msgs[-1].id
        await db.flush()

        monkeypatch.setattr(
            "app.service.ai.builders.context_manager.inject_memories",
            AsyncMock(return_value=(None, [])),
        )
        trace_collector.start(
            conversation_id=conv.id, message_id=None, user_id=1, agent_code=None, model_id=model_id
        )
        messages, _system_prompt, _injected = await context_manager.build_context(
            db, conv, model_id
        )
        assert [m["id"] for m in messages] == [m.id for m in msgs]


class TestSummaryOverflowFallback:
    """摘要保底（§4.1）：窗口外存在未摘要消息即触发压缩，不再依赖 token 阈值"""

    @staticmethod
    async def _seed_conv_with_chain(db) -> tuple[SysAiConversation, list[SysAiMessage]]:
        provider = SysAiProvider(
            provider_code=f"sumprov-{uuid4().hex[:8]}",
            display_name="摘要测试供应商",
            api_base_url="http://localhost:9/v1",
            protocol_type="openai_compat",
            auth_type="bearer",
            status=1,
        )
        db.add(provider)
        await db.flush()
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id="sum-m1",
                display_name="摘要测试模型",
                max_context_tokens=128000,
                max_output_tokens=4096,
                status=1,
            )
        )
        conv = SysAiConversation(user_id=1, model="sum-m1")
        db.add(conv)
        await db.flush()
        parent = None
        msgs = []
        for i in range(5):
            msg = SysAiMessage(
                conversation_id=conv.id,
                parent_message_id=parent,
                role="user" if i % 2 == 0 else "assistant",
                content=f"消息{i}",
                model="sum-m1",
                status=2,
            )
            db.add(msg)
            await db.flush()
            msgs.append(msg)
            parent = msg.id
        conv.current_branch_message_id = msgs[-1].id
        await db.flush()
        return conv, msgs

    async def test_overflow_messages_summarized_beyond_watermark(self, db, monkeypatch):
        """窗口只保留最后 2 条：前 3 条（水位后、窗口外）被增量摘要，水位推进"""
        conv, msgs = await self._seed_conv_with_chain(db)
        window_messages = [
            {"id": msgs[-2].id, "role": "user", "content": "消息3"},
            {"id": msgs[-1].id, "role": "assistant", "content": "消息4"},
        ]
        with patch(
            "app.service.ai.service.summary_service.llm_client.stream_chat",
            side_effect=_llm_stream(LLMChunk("text_delta", "这是压缩后的摘要"), LLMChunk("done")),
        ):
            await summary_service.maybe_compress(db, conv, "sum-m1", window_messages)

        assert conv.summary == "这是压缩后的摘要"
        assert conv.summary_upto_message_id == msgs[-3].id

    async def test_no_summary_when_window_covers_all(self, db, monkeypatch):
        """窗口覆盖全链（无裁剪）：不触发压缩，LLM 不被调用"""
        conv, msgs = await self._seed_conv_with_chain(db)
        window_messages = [{"id": m.id, "role": m.role, "content": m.content} for m in msgs]
        with patch("app.service.ai.service.summary_service.llm_client.stream_chat") as stream:
            await summary_service.maybe_compress(db, conv, "sum-m1", window_messages)
        stream.assert_not_called()
        assert conv.summary is None
