import asyncio

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.infrastructure.sse.sse_event_converter import SseEventConverter
from tests.stubs.fakes import RecorderEmitter


def _make_converter(emitter, monkeypatch):
    ctx = {"stream_session_id": "s1", "message_id": 7, "conversation_id": 100}
    monkeypatch.setattr("app.infrastructure.sse.sse_event_converter.sse_emitter_manager", emitter)

    async def _noop_create(db, **kw):
        return None

    monkeypatch.setattr(
        "app.infrastructure.sse.sse_event_converter.ai_agent_thought_repository",
        type("R", (), {"create_thought": staticmethod(_noop_create)})(),
    )
    return SseEventConverter(ctx)


class TestFullSequence:
    async def test_text_reply_full_sequence(self, monkeypatch):
        emitter = RecorderEmitter()
        converter = _make_converter(emitter, monkeypatch)

        await emitter.send_event(
            "s1", "message.start", {"messageId": 7, "conversationId": 100, "model": "gpt"}
        )

        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="你好")]}
        )
        await converter.handle(
            {"type": "messages", "ns": [], "data": [AIMessageChunk(content="世界")]}
        )
        await converter.finish()
        await emitter.send_event(
            "s1",
            "message.end",
            {
                "stopReason": "stop",
                "usage": {
                    "inputTokens": 5,
                    "outputTokens": 10,
                    "cachedInputTokens": 2,
                    "credits": 3,
                },
            },
        )

        types = [e[0] for e in emitter.events]
        assert types == [
            "message.start",
            "content_block.start",
            "content_block.delta",
            "content_block.delta",
            "content_block.stop",
            "message.end",
        ]
        assert emitter.events[0][1] == {"messageId": 7, "conversationId": 100, "model": "gpt"}
        assert emitter.events[1][1] == {"index": 0, "type": "text"}
        assert emitter.events[2][1]["delta"] == {"type": "text_delta", "text": "你好"}
        assert emitter.events[4][1] == {"index": 0}
        assert emitter.events[5][1]["stopReason"] == "stop"
        assert emitter.events[5][1]["usage"] == {
            "inputTokens": 5,
            "outputTokens": 10,
            "cachedInputTokens": 2,
            "credits": 3,
        }

    async def test_tool_reply_sequence_with_thought(self, monkeypatch):
        emitter = RecorderEmitter()
        converter = _make_converter(emitter, monkeypatch)

        chunk1 = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "search", "args": '{"q":', "id": "c1", "index": 0}],
        )
        chunk2 = AIMessageChunk(
            content="",
            # 续片片段只带增量 args，name 与首片一致地为 None（ToolCallChunk.name 可空）
            tool_call_chunks=[{"name": None, "args": '"雾"}', "id": "c1", "index": 0}],
        )
        await converter.handle({"type": "messages", "ns": [], "data": [chunk1]})
        await converter.handle({"type": "messages", "ns": [], "data": [chunk2]})
        await converter.handle(
            {
                "type": "updates",
                "ns": ["sub"],
                "data": {
                    "tool_node": {
                        "messages": [ToolMessage(content="ok", tool_call_id="c1", name="search")]
                    }
                },
            }
        )
        await converter.finish()
        await emitter.send_event("s1", "message.end", {"stopReason": "tool_calls", "usage": {}})

        types = [e[0] for e in emitter.events]
        assert "content_block.start" not in types
        assert types.index("thought") < types.index("message.end")
        delta = next(e for e in emitter.events if e[0] == "content_block.delta")
        assert delta[1]["delta"]["type"] == "input_json_delta"
        assert emitter.events[-1][1]["stopReason"] == "tool_calls"

    async def test_interrupt_before_message_end(self, monkeypatch):
        emitter = RecorderEmitter()
        converter = _make_converter(emitter, monkeypatch)
        from langgraph.types import Interrupt

        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {
                    "__interrupt__": [
                        Interrupt(
                            value={
                                "type": "confirm",
                                "data": {"artifactId": 5, "recommendation": "x"},
                            }
                        )
                    ]
                },
            }
        )
        await emitter.send_event("s1", "message.end", {"stopReason": "stop", "usage": {}})
        types = [e[0] for e in emitter.events]
        assert "interrupt" in types
        assert types.index("interrupt") < types.index("message.end")
        assert emitter.events[0][1] == {
            "type": "confirm",
            "data": {"artifactId": 5, "recommendation": "x"},
        }

    async def test_interrupt_defaults_type_confirm(self, monkeypatch):
        emitter = RecorderEmitter()
        converter = _make_converter(emitter, monkeypatch)
        from langgraph.types import Interrupt

        await converter.handle(
            {
                "type": "updates",
                "ns": [],
                "data": {"__interrupt__": [Interrupt(value={"artifactId": 7})]},
            }
        )
        assert emitter.events[0][1] == {
            "type": "confirm",
            "data": {"artifactId": 7},
        }

    async def test_no_content_no_content_block(self, monkeypatch):
        emitter = RecorderEmitter()
        converter = _make_converter(emitter, monkeypatch)
        await converter.finish()
        await emitter.send_event("s1", "message.end", {"stopReason": "stop", "usage": {}})
        types = [e[0] for e in emitter.events]
        assert types == ["message.end"]
        assert "content_block.start" not in types
        assert "content_block.stop" not in types


class TestStreamTimeoutStateMachine:
    @staticmethod
    def _make_manager(monkeypatch, mock_redis, heartbeat, timeout):
        import app.infrastructure.sse.sse_emitter_manager as m

        manager = m.SseEmitterManager()
        monkeypatch.setattr(m.settings, "AI_MESSAGE_HEARTBEAT_INTERVAL", heartbeat)
        monkeypatch.setattr(m.settings, "AI_MESSAGE_STREAM_TIMEOUT", timeout)

        async def _get_redis():
            return mock_redis

        monkeypatch.setattr(m, "get_redis_client", _get_redis)
        return manager

    async def test_idle_timeout_ends_stream_without_error(self, monkeypatch, mock_redis):
        """空闲超时只结束连接：不推 error（推理仍在跑），也不落终结事件。"""
        import json

        manager = self._make_manager(monkeypatch, mock_redis, heartbeat=0.03, timeout=0.02)
        queue = asyncio.Queue()
        chunks = [chunk async for chunk in manager._stream_from_queue("s-timeout", queue)]
        assert chunks == []

        events = [json.loads(r) for r in await mock_redis.lrange("ai:stream:s-timeout", 0, -1)]
        assert [ev["event"] for ev in events] == []

    async def test_heartbeat_within_timeout_pushes_ping(self, monkeypatch, mock_redis):
        import json

        import app.infrastructure.sse.sse_emitter_manager as m

        manager = self._make_manager(monkeypatch, mock_redis, heartbeat=0.01, timeout=10)
        queue = asyncio.Queue()
        asyncio.get_running_loop().call_later(0.015, lambda: queue.put_nowait(m._STREAM_END))
        chunks = [chunk async for chunk in manager._stream_from_queue("s-ping", queue)]

        assert [
            line for chunk in chunks for line in chunk.splitlines() if line.startswith("event: ")
        ] == ["event: ping"]
        # 心跳不写缓存：重连重放不需要心跳
        assert [json.loads(r) for r in await mock_redis.lrange("ai:stream:s-ping", 0, -1)] == []
