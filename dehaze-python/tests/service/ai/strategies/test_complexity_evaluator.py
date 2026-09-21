"""复杂度评估器测试：规则预判优先，LLM 兜底失败降级 L0 但必须可观测。"""

from contextlib import asynccontextmanager

import pytest

from app.service.ai.service.agent_state import AgentState
from app.service.ai.strategies import complexity_evaluator as m
from app.service.ai.strategies.complexity_evaluator import evaluate_complexity
from tests.stubs.fakes import LLMChunk

# 不含任何规则关键词且长度 ≥ 50：规则无法判定，必须走 LLM 兜底
_UNDETERMINED = (
    "能不能帮我梳理一下这份材料的要点？我想知道其中有哪些关键结论，"
    "以及它们分别对应哪些依据，最好再给出简单的说明。"
)


def _state(content: str) -> AgentState:
    # 按真实契约（AgentState TypedDict）构造，而非宽化的 dict
    return {"messages": [{"role": "user", "content": content}], "model_id": "m1", "user_id": 1}


@asynccontextmanager
async def _null_session():
    yield None


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    monkeypatch.setattr(m, "get_db_session", _null_session)


class TestRuleBasedEval:
    @pytest.mark.parametrize(
        ("content", "expected_mode"),
        [
            ("你好", "direct"),
            ("帮我去雾这张图", "react"),
            ("批量处理这些图片", "plan_execute"),
            ("生成一份审查报告", "reflexion"),
        ],
    )
    async def test_rule_eval_short_circuits_llm(self, content, expected_mode, monkeypatch):
        async def _boom(*a, **k):
            raise AssertionError("规则可判定时不得调用 LLM")

        monkeypatch.setattr(m.llm_client, "stream_chat", _boom)
        result = await evaluate_complexity(_state(content))
        assert result["reasoning_mode"] == expected_mode
        assert result["usage"] == {}


class TestLlmFallbackObservability:
    async def test_llm_failure_degrades_to_l0_with_log(self, monkeypatch, caplog):
        """降级行为保留（评估失败降 L0），但必须留下可排查日志，禁止静默吞错。"""
        caplog.set_level("WARNING")

        async def _boom(*a, **k):
            raise RuntimeError("LLM 不可用")
            yield  # pragma: no cover

        monkeypatch.setattr(m.llm_client, "stream_chat", _boom)
        result = await evaluate_complexity(_state(_UNDETERMINED))
        assert result["complexity"] == "L0"
        assert result["reasoning_mode"] == "direct"
        assert "复杂度评估失败" in caplog.text
        assert "LLM 不可用" in caplog.text

    async def test_unrecognized_llm_output_logs_and_degrades(self, monkeypatch, caplog):
        caplog.set_level("WARNING")

        async def _stream(*a, **k):
            yield LLMChunk(content="无法确定")
            yield LLMChunk(type="done", usage={"total_tokens": 5})

        monkeypatch.setattr(m.llm_client, "stream_chat", _stream)
        result = await evaluate_complexity(_state(_UNDETERMINED))
        assert result["complexity"] == "L0"
        assert "无法识别的等级" in caplog.text

    async def test_llm_level_and_usage_returned(self, monkeypatch):
        async def _stream(*a, **k):
            yield LLMChunk(content="等级为 L2")
            yield LLMChunk(type="done", usage={"total_tokens": 7})

        monkeypatch.setattr(m.llm_client, "stream_chat", _stream)
        result = await evaluate_complexity(_state(_UNDETERMINED))
        assert result["complexity"] == "L2"
        assert result["reasoning_mode"] == "plan_execute"
        assert result["usage"] == {"total_tokens": 7}
