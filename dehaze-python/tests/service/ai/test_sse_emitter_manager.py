import asyncio
import json

import pytest

from app.infrastructure.sse import sse_emitter_manager as m


@pytest.fixture
def emitter(monkeypatch, mock_redis):
    e = m.SseEmitterManager()

    async def _get_redis_client():
        return mock_redis

    monkeypatch.setattr(m, "get_redis_client", _get_redis_client)
    return e, mock_redis


async def test_stop_stream_no_queue_appends_message_end(emitter):
    e, redis = emitter
    await e.stop_stream("s-noqueue")
    key = "ai:stream:s-noqueue"
    raw = await redis.lrange(key, 0, -1)
    events = [json.loads(r) for r in raw]
    assert [ev["event"] for ev in events] == ["message.end"]
    data = events[0]["data"]
    assert data["stopReason"] == "canceled"
    assert data["usage"] == {
        "inputTokens": 0,
        "outputTokens": 0,
        "cachedInputTokens": 0,
        "credits": 0,
    }


async def test_stop_stream_terminal_idempotent(emitter):
    e, redis = emitter
    await e.stop_stream("s-idem")
    await e.stop_stream("s-idem")
    key = "ai:stream:s-idem"
    raw = await redis.lrange(key, 0, -1)
    events = [json.loads(r) for r in raw]
    assert [ev["event"] for ev in events] == ["message.end"]


async def test_stop_stream_with_queue_puts_end_and_terminates(emitter):
    e, redis = emitter
    queue = asyncio.Queue()
    e._queues["s-q"] = queue
    await e.stop_stream("s-q")
    assert queue.qsize() == 1
    key = "ai:stream:s-q"
    raw = await redis.lrange(key, 0, -1)
    events = [json.loads(r) for r in raw]
    assert [ev["event"] for ev in events] == ["message.end"]


async def test_send_event_and_replay(emitter):
    e, redis = emitter
    await e.send_event("s-replay", "message.start", {"a": 1})
    await e.send_event("s-replay", "content_block.delta", {"b": 2})
    cached = await e._get_cached_events("s-replay", 0)
    assert [ev["event"] for ev in cached] == ["message.start", "content_block.delta"]
    cached2 = await e._get_cached_events("s-replay", 1)
    assert [ev["event"] for ev in cached2] == ["content_block.delta"]
