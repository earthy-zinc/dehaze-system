"""子智能体粒度用量（契约 B）：运行期归属 + additive 下发。

归属：子 Agent 自身 DehazeHooksMiddleware 在模型调用期经 subagent_scope 绑定
subagent_code（ContextVar）；采集层（trace_collector）按该上下文内存聚合 per-subagent
用量，推理收尾（reasoning_service）在 message.end 的 usage 中 additive 追加 subAgents。
覆盖：多子智能体分类 / 无子智能体整键省略 / 部分调用失败 / agentCode 缺失 /
credits 复用既有计价路径（同价口径）。
"""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from app.models.schema.ai_model_price import (
    ModelPriceCreateRequest,
    ModelPriceDetailForm,
)
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.service.ai.middleware.dehaze_hooks_middleware import DehazeHooksMiddleware
from app.service.ai.middleware.run_context import current_subagent_code, subagent_scope
from app.service.ai.service import trace_collector
from app.service.ai.service.credits_service import calculate_credits
from app.service.ai.service.reasoning_service import reasoning_service
from app.service.ai_model_price_service import AiModelPriceService
from tests.stubs.fakes import RecorderEmitter

pytestmark = pytest.mark.requires_db


@pytest.fixture(autouse=True)
def _reset_collector():
    """清空采集器 ContextVar，避免跨用例残留"""
    trace_collector._current_collector.set(None)
    yield
    trace_collector._current_collector.set(None)


def _chunk(type: str, content: str = "", usage: dict | None = None):
    return SimpleNamespace(type=type, content=content, usage=usage, tool_call_name="")


def _collector():
    return trace_collector.start(
        conversation_id=1, message_id=11, user_id=42, agent_code="default", model_id="sub-model"
    )


async def _seed_price(db, model_id: str) -> None:
    """种子用户售价（peak/idle 同价，保证任意时刻换算确定）"""
    svc = AiModelPriceService(price_repository=ai_model_price_repository)
    await svc.create_price(
        db,
        ModelPriceCreateRequest(
            model_id=model_id,
            provider_id=1,
            effective_from=datetime(2026, 1, 1),
            details=[
                ModelPriceDetailForm(token_type=t, time_slot=s, unit_price=Decimal(p))
                for t, p in (("input", "1000000"), ("cached", "500000"), ("output", "4000000"))
                for s in ("idle", "peak")
            ],
        ),
    )


class TestSubagentScope:
    async def test_scope_binds_and_restores(self):
        assert current_subagent_code() is None
        with subagent_scope("agentA"):
            assert current_subagent_code() == "agentA"
        assert current_subagent_code() is None

    async def test_scope_nested_restores_layer_by_layer(self):
        with subagent_scope("outer"):
            with subagent_scope("inner"):
                assert current_subagent_code() == "inner"
            assert current_subagent_code() == "outer"
        assert current_subagent_code() is None

    async def test_scope_none_is_noop(self):
        with subagent_scope(None):
            assert current_subagent_code() is None


def _request() -> ModelRequest:
    """真实 ModelRequest（middleware 仅用 state.system_message，模型实例不参与调用）。"""
    return ModelRequest(model=FakeListChatModel(responses=["ok"]), messages=[])


def _response() -> ModelResponse:
    return ModelResponse(result=[AIMessage(content="ok")])


def _ctx():
    return {"max_steps": 20, "token_budget": 50000, "tool_timeout": 60, "step_count": 0}


@pytest.fixture(autouse=True)
def _no_hooks(monkeypatch):
    class _H:
        async def run_hooks(self, point, state):
            return None

    monkeypatch.setattr("app.service.ai.middleware.dehaze_hooks_middleware.agent_hooks", _H())


class TestMiddlewareAttribution:
    async def test_subagent_middleware_binds_code_during_model_call(self):
        mw = DehazeHooksMiddleware(_ctx(), subagent={"name": "agentA", "priority": 1})
        seen = {}

        async def handler(request):
            seen["code"] = current_subagent_code()
            return _response()

        await mw.awrap_model_call(_request(), handler)
        assert seen["code"] == "agentA"
        assert current_subagent_code() is None  # 作用域退出还原

    async def test_main_middleware_has_no_code(self):
        mw = DehazeHooksMiddleware(_ctx())
        seen = {}

        async def handler(request):
            seen["code"] = current_subagent_code()
            return _response()

        await mw.awrap_model_call(_request(), handler)
        assert seen["code"] is None


class TestCollectorAggregation:
    async def test_multiple_subagents_grouped_separately(self, db):
        await _seed_price(db, "sub-model")
        collector = _collector()

        with subagent_scope("agentA"):
            call = collector.begin_llm_call("sub-model", [], None, None)
            call.observe_chunk(
                _chunk(
                    "done",
                    usage={"prompt_tokens": 1000, "completion_tokens": 300, "cached_tokens": 200},
                )
            )
            await call.finish(completed=True)
        with subagent_scope("agentB"):
            call = collector.begin_llm_call("sub-model", [], None, None)
            call.observe_chunk(
                _chunk("done", usage={"prompt_tokens": 500, "completion_tokens": 100})
            )
            await call.finish(completed=True)
        await trace_collector.drain()

        expected_a = await calculate_credits(db, "sub-model", 1000, 300, 200)
        expected_b = await calculate_credits(db, "sub-model", 500, 100, 0)
        assert expected_a > 0
        assert collector.subagent_usage["agentA"] == {
            "agentCode": "agentA",
            "inputTokens": 1000,
            "outputTokens": 300,
            "cachedInputTokens": 200,
            "credits": expected_a,
        }
        assert collector.subagent_usage["agentB"] == {
            "agentCode": "agentB",
            "inputTokens": 500,
            "outputTokens": 100,
            "cachedInputTokens": 0,
            "credits": expected_b,
        }

    async def test_no_subagent_call_leaves_empty(self, db):
        collector = _collector()
        call = collector.begin_llm_call("sub-model", [], None, None)
        call.observe_chunk(_chunk("done", usage={"prompt_tokens": 1000, "completion_tokens": 300}))
        await call.finish(completed=True)
        await trace_collector.drain()

        assert collector.subagent_usage == {}
        assert collector.subagent_usage_payload() == []

    async def test_partial_failure_still_aggregates(self, db):
        """同一子 Agent 一次成功 + 一次失败：失败调用按可用用量计入，不丢成功部分"""
        await _seed_price(db, "sub-model")
        collector = _collector()

        with subagent_scope("agentA"):
            ok = collector.begin_llm_call("sub-model", [], None, None)
            ok.observe_chunk(
                _chunk(
                    "done",
                    usage={"prompt_tokens": 1000, "completion_tokens": 300, "cached_tokens": 200},
                )
            )
            await ok.finish(completed=True)
        # 失败且无用量：贡献 0，不破坏已聚合量
        with subagent_scope("agentA"):
            bad = collector.begin_llm_call("sub-model", [], None, None)
            await bad.finish(completed=False, error_type="5xx")
        # 失败但已有部分用量：如实计入
        with subagent_scope("agentA"):
            bad2 = collector.begin_llm_call("sub-model", [], None, None)
            bad2.observe_chunk(
                _chunk("done", usage={"prompt_tokens": 100, "completion_tokens": 50})
            )
            await bad2.finish(completed=False, error_type="timeout")
        await trace_collector.drain()

        entry = collector.subagent_usage["agentA"]
        assert entry["inputTokens"] == 1100
        assert entry["outputTokens"] == 350
        assert entry["cachedInputTokens"] == 200


class TestPushEndAdditive:
    def _patch_emitter(self, monkeypatch):
        emitter = RecorderEmitter()
        monkeypatch.setattr("app.service.ai.service.reasoning_service.sse_emitter_manager", emitter)
        return emitter

    async def test_omits_subagents_key_when_none(self, db, monkeypatch):
        emitter = self._patch_emitter(monkeypatch)
        _collector()  # 无子 Agent 调用的采集器

        await reasoning_service._push_end(
            "s1",
            {
                "stop_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2},
            },
            credits=7,
        )

        event_type, data = emitter.events[-1]
        assert event_type == "message.end"
        assert data == {
            "stopReason": "stop",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cachedInputTokens": 2,
                "credits": 7,
            },
        }
        assert "subAgents" not in data["usage"]

    async def test_adds_subagents_key_when_present(self, db, monkeypatch):
        emitter = self._patch_emitter(monkeypatch)
        await _seed_price(db, "sub-model")
        collector = _collector()
        await collector.record_subagent_usage("agentA", "sub-model", 1000, 300, 200)

        await reasoning_service._push_end(
            "s1",
            {
                "stop_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2},
            },
            credits=7,
        )

        usage = emitter.events[-1][1]["usage"]
        # 主用量字段口径不变
        assert usage["inputTokens"] == 10
        assert usage["outputTokens"] == 5
        assert usage["cachedInputTokens"] == 2
        assert usage["credits"] == 7
        expected = await calculate_credits(db, "sub-model", 1000, 300, 200)
        assert usage["subAgents"] == [
            {
                "agentCode": "agentA",
                "inputTokens": 1000,
                "outputTokens": 300,
                "cachedInputTokens": 200,
                "credits": expected,
            }
        ]
