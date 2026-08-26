import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.infrastructure.sse.sse_event_converter import SseEventConverter
from tests.stubs.fakes import RecorderEmitter


class _Repo:

    def __init__(self):
        self.thoughts = []

    async def create_thought(self, db, **kwargs):
        self.thoughts.append(kwargs)
        return None


@pytest.fixture
def conv(monkeypatch):
    ctx = {"stream_session_id": "s1", "message_id": 1, "conversation_id": 100}
    emitter = RecorderEmitter()
    repo = _Repo()
    monkeypatch.setattr("app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter)
    monkeypatch.setattr("app.infrastructure.sse.sse_event_converter.ai_agent_thought_repository", repo)
    return SseEventConverter(ctx), emitter, repo


def _thinking_chunk(text):
    return AIMessageChunk(content="", additional_kwargs={"thinking": text})


async def test_thinking_delta_streams_as_separate_block(conv):
    converter, emitter, _ = conv
    await converter.handle({"type": "messages", "ns": [], "data": [_thinking_chunk("思考A")]})
    await converter.handle({"type": "messages", "ns": [], "data": [_thinking_chunk("思考B")]})
    assert emitter.events[0] == ("content_block.start", {"index": 1, "type": "thinking"})
    deltas = [e[1] for e in emitter.events if e[0] == "content_block.delta"]
    assert deltas == [
        {"index": 1, "delta": {"type": "thinking_delta", "thinking": "思考A"}},
        {"index": 1, "delta": {"type": "thinking_delta", "thinking": "思考B"}},
    ]
    assert not [
        e for e in emitter.events if e[0] == "content_block.start" and e[1].get("type") == "text"
    ]


async def test_text_closes_thinking_block_and_records_thinking_thought(conv):
    converter, emitter, repo = conv
    await converter.handle({"type": "messages", "ns": [], "data": [_thinking_chunk("先思考")]})
    await converter.handle({"type": "messages", "ns": [], "data": [AIMessageChunk(content="正文")]})
    stops = [e[1] for e in emitter.events if e[0] == "content_block.stop"]
    assert {"index": 1} in stops
    text_starts = [
        e for e in emitter.events if e[0] == "content_block.start" and e[1].get("type") == "text"
    ]
    assert text_starts
    await converter.finish()
    thinking_thoughts = [t for t in repo.thoughts if t["tool"] is None]
    assert thinking_thoughts
    assert thinking_thoughts[0]["thought"] == "先思考"
    assert thinking_thoughts[0]["status"] == 1


async def test_thinking_thought_before_tool_thought(conv):
    converter, emitter, repo = conv
    await converter.handle({"type": "messages", "ns": [], "data": [_thinking_chunk("为何搜索")]})
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "search", "args": '{"q":"x"}', "id": "c1", "index": None}],
    )
    await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
    await converter.handle(
        {
            "type": "updates",
            "ns": [],
            "data": {
                "tool_node": {
                    "messages": [
                        ToolMessage(content="ok", tool_call_id="c1", name="search"),
                    ]
                }
            },
        }
    )
    thoughts = repo.thoughts
    assert len(thoughts) == 2
    assert thoughts[0]["tool"] is None
    assert thoughts[0]["thought"] == "为何搜索"
    assert thoughts[1]["tool"] == "search"
    assert thoughts[1]["status"] == 1
    thought_events = [e[1] for e in emitter.events if e[0] == "thought"]
    assert thought_events[0]["tool"] is None
    assert thought_events[1]["tool"] == "search"


async def test_tool_status_2_failure_persisted_and_emitted(conv):
    converter, emitter, repo = conv
    chunk = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "dehaze", "args": "{}", "id": "c2", "index": None}]
    )
    await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
    await converter.handle(
        {
            "type": "updates",
            "ns": [],
            "data": {
                "tool_node": {
                    "messages": [
                        ToolMessage(
                            content="工具调用失败：超时",
                            tool_call_id="c2",
                            name="dehaze",
                            status="error",
                            additional_kwargs={
                                "_dehaze_status": 2,
                                "_dehaze_error": "工具调用失败：超时",
                            },
                        ),
                    ]
                }
            },
        }
    )
    thought = repo.thoughts[0]
    assert thought["status"] == 2
    assert "超时" in thought["error"]
    event = [e[1] for e in emitter.events if e[0] == "thought"][0]
    assert event["status"] == 2
    assert "超时" in event["error"]


async def test_tool_status_3_skipped_persisted_and_emitted(conv):
    converter, emitter, repo = conv
    chunk = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "mcp_tool", "args": "{}", "id": "c3", "index": None}]
    )
    await converter.handle({"type": "messages", "ns": [], "data": [chunk]})
    await converter.handle(
        {
            "type": "updates",
            "ns": [],
            "data": {
                "tool_node": {
                    "messages": [
                        ToolMessage(
                            content="该步骤已跳过：服务不可用",
                            tool_call_id="c3",
                            name="mcp_tool",
                            status="error",
                            additional_kwargs={"_dehaze_status": 3, "_dehaze_error": "服务不可用"},
                        ),
                    ]
                }
            },
        }
    )
    thought = repo.thoughts[0]
    assert thought["status"] == 3
    assert thought["error"] == "服务不可用"
    event = [e[1] for e in emitter.events if e[0] == "thought"][0]
    assert event["status"] == 3
    assert event["error"] == "服务不可用"


async def test_finish_flushes_residual_thinking(conv):
    converter, emitter, repo = conv
    await converter.handle({"type": "messages", "ns": [], "data": [_thinking_chunk("只思考")]})
    await converter.finish()
    assert ("content_block.stop", {"index": 1}) in emitter.events
    thinking_thoughts = [t for t in repo.thoughts if t["tool"] is None]
    assert thinking_thoughts and thinking_thoughts[0]["thought"] == "只思考"
