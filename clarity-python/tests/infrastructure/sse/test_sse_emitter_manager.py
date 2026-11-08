import asyncio
import json

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.sse import sse_emitter_manager as m


@pytest.fixture
def emitter(monkeypatch, mock_redis):
    e = m.SseEmitterManager()

    async def _get_redis_client():
        return mock_redis

    monkeypatch.setattr(m, "get_redis_client", _get_redis_client)
    return e, mock_redis


def _types(chunks: list[str]) -> list[str]:
    """从 SSE 文本块提取事件类型序列"""
    return [
        line.split(": ", 1)[1]
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("event: ")
    ]


async def _cached(e, redis, stream_session_id: str) -> list[dict]:
    raw = await redis.lrange(f"ai:stream:{stream_session_id}", 0, -1)
    return [json.loads(r) for r in raw]


async def test_stop_stream_no_queue_writes_nothing(emitter):
    """停止只结束连接：不补写 canceled 终帧（终帧由收尾方唯一写出）。"""
    e, redis = emitter
    await e.stop_stream("s-noqueue")
    assert await redis.lrange("ai:stream:s-noqueue", 0, -1) == []


async def test_stop_stream_idempotent(emitter):
    e, redis = emitter
    await e.stop_stream("s-idem")
    await e.stop_stream("s-idem")
    assert await redis.lrange("ai:stream:s-idem", 0, -1) == []


async def test_stop_stream_with_queue_puts_end_and_terminates(emitter):
    e, redis = emitter
    queue = asyncio.Queue()
    e._queues["s-q"] = queue
    await e.stop_stream("s-q")
    assert queue.qsize() == 1
    assert await redis.lrange("ai:stream:s-q", 0, -1) == []


async def test_send_message_end_written_once(emitter):
    """终态唯一：四条收尾路径共享同一个幂等键，只有第一条 message.end 生效。"""
    e, redis = emitter
    first = await e.send_message_end("s-end", message_id=7, status="completed")
    second = await e.send_message_end("s-end", message_id=7, status="canceled")
    assert first is True
    assert second is False
    events = await _cached(e, redis, "s-end")
    assert [ev["event"] for ev in events] == ["message.end"]
    assert events[0]["data"] == {"messageId": 7, "status": "completed"}


async def test_send_message_end_carries_error_and_usage(emitter):
    e, redis = emitter
    await e.send_message_end(
        "s-end2",
        message_id=9,
        status="failed",
        error="模型不可用",
        usage={"inputTokens": 1, "outputTokens": 2, "cachedInputTokens": 0, "credits": 0},
    )
    event = (await _cached(e, redis, "s-end2"))[0]
    assert event["data"]["status"] == "failed"
    assert event["data"]["error"] == "模型不可用"
    assert event["data"]["usage"]["outputTokens"] == 2


async def test_send_event_and_replay(emitter):
    e, _redis = emitter
    await e.send_event("s-replay", "message.start", {"a": 1})
    await e.send_event("s-replay", "content_block.delta", {"b": 2})
    cached, has_cache = await e._get_cached_events("s-replay", 0)
    assert [ev["event"] for ev in cached] == ["message.start", "content_block.delta"]
    assert has_cache is True
    cached2, _ = await e._get_cached_events("s-replay", 1)
    assert [ev["event"] for ev in cached2] == ["content_block.delta"]


async def test_get_cached_events_reports_missing_cache(emitter):
    """无缓存（streamSessionId 过期/从未存在）时存在性必须为 False。"""
    e, _redis = emitter
    events, has_cache = await e._get_cached_events("s-missing", 0)
    assert events == []
    assert has_cache is False


async def test_send_error_sanitizes_unknown_exception(emitter):
    e, redis = emitter

    await e.send_error("s-err", RuntimeError("minio 连接失败 secret=abc123"), message_id=3)

    events = await _cached(e, redis, "s-err")
    assert [ev["event"] for ev in events] == ["error", "message.end"]
    assert events[0]["data"] == {
        "code": ResultCode.AI_LLM_CALL_FAILED.code,
        "message": ResultCode.AI_LLM_CALL_FAILED.msg,
        "messageId": 3,
    }
    assert "secret=abc123" not in json.dumps(events, ensure_ascii=False)
    assert events[1]["data"]["status"] == "failed"
    assert events[1]["data"]["messageId"] == 3


async def test_send_error_keeps_business_code(emitter):
    e, redis = emitter

    await e.send_error(
        "s-err2", BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在"), message_id=4
    )

    events = await _cached(e, redis, "s-err2")
    assert events[0]["data"] == {
        "code": ResultCode.RESOURCE_NOT_FOUND.code,
        "message": "会话不存在",
        "messageId": 4,
    }


async def test_send_error_does_not_duplicate_end_after_finalize(emitter):
    """推理侧已写出终态时，异常出口不再补第二条 message.end。"""
    e, redis = emitter
    await e.send_message_end("s-err3", message_id=5, status="completed")
    await e.send_error("s-err3", RuntimeError("late failure"), message_id=5)
    events = await _cached(e, redis, "s-err3")
    assert [ev["event"] for ev in events] == ["message.end", "error"]


async def test_idle_timeout_ends_stream_without_error(emitter, monkeypatch):
    """空闲超时只结束连接：不推 error（推理仍在跑），连接结束后事件仍进缓存。"""
    e, redis = emitter
    # 心跳间隔远大于空闲超时：首轮空闲即判定超时。两者差值须远大于事件循环时钟分辨率
    # （Windows 15.6ms，定时器可提前一个分辨率触发），否则首轮会误判为未超时并推心跳
    monkeypatch.setattr(m.settings, "AI_MESSAGE_HEARTBEAT_INTERVAL", 0.3)
    monkeypatch.setattr(m.settings, "AI_MESSAGE_STREAM_TIMEOUT", 0.01)

    async def _late_event():
        await asyncio.sleep(0.5)
        await e.send_event("s-idle", "block.delta", {"blockId": "b", "delta": "迟到 token"})

    task = asyncio.create_task(_late_event())
    chunks = [chunk async for chunk in e.create_stream(1, "s-idle")]
    assert chunks == []
    assert await _cached(e, redis, "s-idle") == []
    assert "s-idle" not in e._queues

    # 连接结束后后台推理继续推送，事件照常进缓存供重连重放
    await task
    assert [ev["event"] for ev in await _cached(e, redis, "s-idle")] == ["block.delta"]


async def test_heartbeat_reaches_client_without_feeding_queue(emitter, monkeypatch):
    """心跳直连下发：不回流队列（否则刷新活跃时间使空闲超时永不触发），也不写缓存。"""
    e, redis = emitter
    monkeypatch.setattr(m.settings, "AI_MESSAGE_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(m.settings, "AI_MESSAGE_STREAM_TIMEOUT", 10)
    await e.register_stream("s-hb")

    gen = e.create_stream(1, "s-hb")
    chunk = await gen.__anext__()
    assert e._queues["s-hb"].qsize() == 0
    await gen.aclose()

    assert _types([chunk]) == ["ping"]
    assert await _cached(e, redis, "s-hb") == []


async def test_reconnect_registers_queue_before_replay(emitter, monkeypatch):
    """重放前队列必须已注册，否则重放期间到达的事件无处可落。"""
    e, _redis = emitter
    await e.send_event("s-recon", "block.delta", {"blockId": "b", "delta": "a"})
    real_get = e._get_cached_events
    seen = {}

    async def _spy(stream_session_id, last_event_id):
        await asyncio.sleep(0)
        seen["has_queue"] = stream_session_id in e._queues
        return await real_get(stream_session_id, last_event_id)

    monkeypatch.setattr(e, "_get_cached_events", _spy)

    gen = e.reconnect("s-recon", 0)
    first = await gen.__anext__()
    assert _types([first]) == ["block.delta"]
    assert seen["has_queue"] is True
    await gen.aclose()
    assert "s-recon" not in e._queues


async def test_reconnect_delivers_event_produced_during_replay(emitter, monkeypatch):
    e, _redis = emitter
    await e.send_event("s-race", "message.start", {"messageId": 1})
    real_get = e._get_cached_events

    async def _push_during_replay(stream_session_id, last_event_id):
        events, has_cache = await real_get(stream_session_id, last_event_id)
        # 重放 await 期间推理推来的新事件：队列已注册则应送达重连客户端
        await e.send_event("s-race", "block.delta", {"blockId": "b", "delta": "b"})
        return events, has_cache

    monkeypatch.setattr(e, "_get_cached_events", _push_during_replay)

    gen = e.reconnect("s-race", 0)
    chunks = [await gen.__anext__(), await gen.__anext__()]
    await gen.aclose()

    assert _types(chunks) == ["message.start", "block.delta"]


async def test_reconnect_dedups_event_racing_replay_read(emitter, monkeypatch):
    """register 与读缓存之间写入的事件同时在重放列表与活跃队列，只能下发一次。"""
    e, _redis = emitter
    monkeypatch.setattr(m.settings, "AI_MESSAGE_HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr(m.settings, "AI_MESSAGE_STREAM_TIMEOUT", 0.2)
    await e.send_event("s-dup", "message.start", {"messageId": 1})
    real_get = e._get_cached_events

    async def _push_before_read(stream_session_id, last_event_id):
        # 读缓存前到达：事件写入缓存的同时也进入已注册的活跃队列
        await e.send_event(stream_session_id, "block.delta", {"blockId": "b", "delta": "x"})
        return await real_get(stream_session_id, last_event_id)

    monkeypatch.setattr(e, "_get_cached_events", _push_before_read)

    chunks = [chunk async for chunk in e.reconnect("s-dup", 0)]

    types = [t for t in _types(chunks) if t != "ping"]
    # 重放已下发两条；队列里的同一条 delta 必须按 watermark 丢弃，不能再拼一次
    assert types == ["message.start", "block.delta"]


async def test_reconnect_stops_after_replaying_terminal_event(emitter):
    """流已终结（缓存含 message.end）时重放完即结束，不残留队列空等。"""
    e, _redis = emitter
    await e.send_event("s-done", "block.delta", {"blockId": "b", "delta": "a"})
    await e.send_message_end("s-done", message_id=1, status="completed")
    await e.stop_stream("s-done")

    chunks = [chunk async for chunk in e.reconnect("s-done", 0)]

    assert _types(chunks) == ["block.delta", "message.end"]
    assert "s-done" not in e._queues


async def test_reconnect_expired_session_ends_immediately(emitter, monkeypatch):
    """streamSessionId 已过期（无缓存、无活跃流）时重连立即结束。

    否则会挂队列空等到空闲超时，客户端只拿到超时错误（SDK 该边界用例即此现象）。
    """
    e, _redis = emitter
    monkeypatch.setattr(m.settings, "AI_MESSAGE_STREAM_TIMEOUT", 5)

    chunks = [chunk async for chunk in e.reconnect("s-gone", 0)]

    assert chunks == []
    assert "s-gone" not in e._queues
