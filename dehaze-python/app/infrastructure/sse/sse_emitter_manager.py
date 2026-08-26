"""SSE 流式输出管理器

管理 AI 对话的 SSE 连接与事件推送：
- 连接管理：维护会话与 SSE 连接的映射，同会话同一时间只允许一个活跃连接
- 事件推送：message.start / content_block.start|delta|stop / thought / interrupt / ping /
  error / message.end
- 心跳保活：每 15 秒推送 ping 事件，防止代理超时断连
- token 缓存：每个事件写入 Redis list（ai:stream:{streamSessionId}，TTL 5 分钟），支撑断线重连
- 断线重连：客户端携带 Last-Event-ID 重连时，从 Redis 缓存重放断点之后的事件
- 并发锁：Redis 分布式锁（ai:streaming:{conversationId}，TTL 120 秒），
  同会话同一时间只允许一个流式输出
- 流被停止/取消且无活跃连接时，向缓存追加标准 message.end 终结事件，重连客户端重放后正确结束

部署约束：多实例部署需 sticky session（同一 stream_session 的活跃队列仅存在于单实例内存），
断线重连仅重放 Redis 缓存，续流能力不做 Redis pub/sub 跨实例同步。

SSE 事件格式（每个事件携带递增 id）：
    id: {eventId}
    event: {eventType}
    data: {json_data}
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.redis import get_redis_client

logger = logging.getLogger(__name__)

_STREAM_CACHE_PREFIX = "ai:stream:"
_STREAM_LOCK_PREFIX = "ai:streaming:"
_STREAM_CACHE_TTL = 300
_STREAM_LOCK_TTL = 120
_STREAM_END = None


class SseEmitterManager:
    """SSE 流式输出管理器（单例）"""

    def __init__(self) -> None:
        # stream_session_id -> asyncio.Queue，保存待推送事件（含 id/event/data）
        self._queues: dict[str, asyncio.Queue] = {}

    # ── 并发锁 ──────────────────────────────────────────

    async def acquire_lock(self, conversation_id) -> bool:
        """获取流式并发锁，成功返回 True，已被占用返回 False"""
        redis = await get_redis_client()
        key = f"{_STREAM_LOCK_PREFIX}{conversation_id}"
        return bool(await redis.set(key, "1", nx=True, ex=_STREAM_LOCK_TTL))

    async def release_lock(self, conversation_id) -> None:
        """释放流式并发锁"""
        redis = await get_redis_client()
        await redis.delete(f"{_STREAM_LOCK_PREFIX}{conversation_id}")

    # ── 事件缓存（Redis）────────────────────────────────

    async def _next_event_id(self, stream_session_id: str) -> int:
        """生成递增事件 ID（Redis 原子自增，TTL 与缓存一致）"""
        redis = await get_redis_client()
        key = f"{_STREAM_CACHE_PREFIX}{stream_session_id}:counter"
        event_id = await redis.incr(key)
        await redis.expire(key, _STREAM_CACHE_TTL)
        return event_id

    async def _cache_event(self, stream_session_id: str, event: dict) -> None:
        """将事件写入 Redis list 并刷新 TTL"""
        redis = await get_redis_client()
        key = f"{_STREAM_CACHE_PREFIX}{stream_session_id}"
        await redis.rpush(key, json.dumps(event, ensure_ascii=False))  # type: ignore
        await redis.expire(key, _STREAM_CACHE_TTL)

    async def _get_cached_events(self, stream_session_id: str, last_event_id: int) -> list[dict]:
        """从 Redis 读取断点（last_event_id）之后的事件"""
        redis = await get_redis_client()
        key = f"{_STREAM_CACHE_PREFIX}{stream_session_id}"
        raw_events = await redis.lrange(key, 0, -1)  # type: ignore
        events = []
        for raw in raw_events:
            try:
                event = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if event.get("id", 0) > last_event_id:
                events.append(event)
        return events

    # ── 事件推送 ────────────────────────────────────────

    async def send_event(self, stream_session_id: str, event_type: str, data: dict) -> None:
        """推送事件并缓存到 Redis，同时写入活跃连接队列"""
        event_id = await self._next_event_id(stream_session_id)
        event = {"id": event_id, "event": event_type, "data": data}
        await self._cache_event(stream_session_id, event)
        queue = self._queues.get(stream_session_id)
        if queue is not None:
            await queue.put(event)

    async def _cache_terminal(self, stream_session_id: str) -> None:
        """在 Redis 缓存追加标准终结事件 message.end（stopReason=canceled），
        供重连客户端重放后走与正常完成一致的收尾逻辑而非挂起。
        仅写缓存，不进入活跃队列（活跃流由结束哨兵收尾）。

        幂等标记存 Redis（SETNX + TTL 与流缓存一致）：跨实例一致、
        进程重启不失效，且不随会话数无界增长（替代进程内集合）。
        """
        redis = await get_redis_client()
        first = await redis.set(
            f"{_STREAM_CACHE_PREFIX}{stream_session_id}:terminated",
            "1",
            nx=True,
            ex=_STREAM_CACHE_TTL,
        )
        if not first:
            return  # 已终结过（重复 stop / 多实例并发），不重复追加
        event_id = await self._next_event_id(stream_session_id)
        await self._cache_event(
            stream_session_id,
            {
                "id": event_id,
                "event": "message.end",
                "data": {
                    "stopReason": "canceled",
                    "usage": {
                        "inputTokens": 0,
                        "outputTokens": 0,
                        "cachedInputTokens": 0,
                        "credits": 0,
                    },
                },
            },
        )

    async def stop_stream(self, stream_session_id: str) -> None:
        """结束流式输出：向队列放入结束哨兵；无活跃队列时在缓存追加终结事件。

        客户端取消/断连后重连需能收到 message.end 终结事件走统一收尾，故无论
        是否有活跃队列都确保缓存中存在终结标记（已有则幂等跳过）。
        """
        queue = self._queues.get(stream_session_id)
        if queue is not None:
            await queue.put(_STREAM_END)
        await self._cache_terminal(stream_session_id)

    # ── 流生成 ──────────────────────────────────────────

    @staticmethod
    def _format_event(event: dict) -> str:
        """将事件格式化为 SSE 文本"""
        data = json.dumps(event["data"], ensure_ascii=False)
        return f"id: {event['id']}\nevent: {event['event']}\ndata: {data}\n\n"

    async def _stream_from_queue(
        self, stream_session_id: str, queue: asyncio.Queue
    ) -> AsyncGenerator[str, None]:
        """从队列读取事件并推送，空闲时发送心跳，超时则推送 error 并结束"""
        heartbeat = settings.AI_MESSAGE_HEARTBEAT_INTERVAL
        timeout = settings.AI_MESSAGE_STREAM_TIMEOUT
        loop = asyncio.get_running_loop()
        last_activity = loop.time()
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except TimeoutError:
                if loop.time() - last_activity >= timeout:
                    await self.send_event(
                        stream_session_id,
                        "error",
                        {
                            "code": ResultCode.SYSTEM_EXECUTION_TIMEOUT.code,
                            "message": ResultCode.SYSTEM_EXECUTION_TIMEOUT.msg,
                        },
                    )
                    # error 后补 message.end 收尾，保证客户端总能走到统一完成处理
                    await self.send_event(
                        stream_session_id,
                        "message.end",
                        {
                            "stopReason": "error",
                            "usage": {
                                "inputTokens": 0,
                                "outputTokens": 0,
                                "cachedInputTokens": 0,
                                "credits": 0,
                            },
                        },
                    )
                    break
                await self.send_event(stream_session_id, "ping", {})
                continue
            if event is _STREAM_END:
                break
            last_activity = loop.time()
            yield self._format_event(event)

    async def register_stream(self, stream_session_id: str) -> None:
        """预注册流式会话并创建活跃事件队列。

        供发送端在推送 message.start 等前置事件前调用，使这些事件能进入活跃
        队列送达客户端（否则队列未建立时 send_event 仅写入 Redis 缓存而丢失）。
        create_stream 会复用该队列；未预注册时 create_stream 自行创建。
        """
        if stream_session_id not in self._queues:
            self._queues[stream_session_id] = asyncio.Queue()

    async def create_stream(self, conversation_id, stream_session_id) -> AsyncGenerator[str, None]:
        """创建 SSE 流生成器。

        调用方需先通过 acquire_lock 获取并发锁（失败则拒绝请求），
        流结束或客户端断连时自动释放锁。若已由 register_stream 预注册事件队列
        （用于 message.start 前置事件），则复用该队列保证事件顺序。
        """
        queue = self._queues.get(stream_session_id)
        if queue is None:
            queue = asyncio.Queue()
            self._queues[stream_session_id] = queue
        try:
            async for chunk in self._stream_from_queue(stream_session_id, queue):
                yield chunk
        finally:
            self._queues.pop(stream_session_id, None)
            await self.release_lock(conversation_id)

    async def reconnect(
        self, stream_session_id: str, last_event_id: int
    ) -> AsyncGenerator[str, None]:
        """断线重连：先重放 Redis 缓存中断点之后的事件，再继续拉取新事件"""
        for event in await self._get_cached_events(stream_session_id, last_event_id):
            yield self._format_event(event)
        queue = self._queues.get(stream_session_id)
        if queue is not None:
            async for chunk in self._stream_from_queue(stream_session_id, queue):
                yield chunk


sse_emitter_manager = SseEmitterManager()
