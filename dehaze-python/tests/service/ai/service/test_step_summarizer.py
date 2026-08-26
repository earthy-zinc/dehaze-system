import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.service.ai.service.step_summarizer import _generate_summaries, summarize_steps

pytestmark = pytest.mark.requires_db
from tests.stubs.fakes import LLMChunk


def _llm_stream(*chunks):
    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c

    return _gen


def _repo(thoughts, updated):
    class _Repo:
        async def list_by_message(self, db, message_id):
            return thoughts

        async def update(self, db, thought, data):
            updated.append((id(thought), data.get("summary")))

    return _Repo()


async def test_generate_summaries_batch(mock_redis):
    with patch(
        "app.service.ai.service.step_summarizer.llm_client.stream_chat",
        side_effect=_llm_stream(
            LLMChunk("text_delta", '["步骤1：分析图像", "步骤2：执行去雾"]'), LLMChunk("done")
        ),
    ):
        steps = [
            {"thought": "先分析", "tool": "image_analysis", "observation": "雾图"},
            {"thought": "再处理", "tool": "dehaze", "observation": "完成"},
        ]
        summaries = await _generate_summaries(None, "model-x", steps)
    assert summaries == ["步骤1：分析图像", "步骤2：执行去雾"]


async def test_generate_summaries_non_json_returns_none(mock_redis):
    with patch(
        "app.service.ai.service.step_summarizer.llm_client.stream_chat",
        side_effect=_llm_stream(LLMChunk("text_delta", "抱歉，我无法概括"), LLMChunk("done")),
    ):
        result = await _generate_summaries(None, "model-x", [{"thought": "t", "tool": None}])
    assert result is None


async def test_generate_summaries_json_block(mock_redis):
    with patch(
        "app.service.ai.service.step_summarizer.llm_client.stream_chat",
        side_effect=_llm_stream(LLMChunk("text_delta", '```json\n["概括"]\n```'), LLMChunk("done")),
    ):
        result = await _generate_summaries(None, "model-x", [{"thought": "t", "tool": None}])
    assert result == ["概括"]


async def test_summarize_steps_updates_thoughts(db, mock_redis):
    thoughts = [
        SimpleNamespace(thought="t1", tool="tool1", observation="o1"),
        SimpleNamespace(thought="t2", tool="tool2", observation="o2"),
    ]
    updated = []

    with (
        patch("app.service.ai.service.step_summarizer.ai_agent_thought_repository", _repo(thoughts, updated)),
        patch(
            "app.service.ai.service.step_summarizer.llm_client.stream_chat",
            side_effect=_llm_stream(LLMChunk("text_delta", json.dumps(["s1", "s2"])), LLMChunk("done")),
        ),
    ):
        await summarize_steps(1, "model-x")
    assert sorted(u[1] for u in updated) == ["s1", "s2"]


async def test_summarize_steps_llm_failure_silent(db, mock_redis):
    thoughts = [SimpleNamespace(thought="t1", tool="t1", observation="o1")]
    updated = []

    with (
        patch("app.service.ai.service.step_summarizer.ai_agent_thought_repository", _repo(thoughts, updated)),
        patch(
            "app.service.ai.service.step_summarizer.llm_client.stream_chat",
            AsyncMock(side_effect=RuntimeError("llm down")),
        ),
    ):
        await summarize_steps(1, "model-x")
    assert updated == []


async def test_summarize_steps_no_thoughts_no_call(db, mock_redis):
    with (
        patch(
            "app.service.ai.service.step_summarizer.ai_agent_thought_repository",
            SimpleNamespace(list_by_message=AsyncMock(return_value=[])),
        ),
        patch("app.service.ai.service.step_summarizer.llm_client.stream_chat") as stream,
    ):
        await summarize_steps(1, "model-x")
    stream.assert_not_called()
