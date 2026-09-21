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
from app.core.exceptions import public_error
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

    async def _get_cached_events(
        self, stream_session_id: str, last_event_id: int
    ) -> tuple[list[dict], bool]:
        """从 Redis 读取断点（last_event_id）之后的事件，并返回缓存是否存在。

        why: 重连需区分"流在跑但暂无新事件"（继续挂队列续流）与"streamSessionId
        已过期/从未存在"（应立即结束）；仅看"断点后事件数"两者都是空列表，
        后者会挂队列等到空闲超时，客户端只拿到超时错误。
        """
        redis = await get_redis_client()
        key = f"{_STREAM_CACHE_PREFIX}{stream_session_id}"
        raw_events = await redis.lrange(key, 0, -1)  # type: ignore
        events = []
        for raw in raw_events:
            try:
                event = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as e:
                # 单条历史事件损坏不应中断整段重放（契约：跳过），但必须留痕：
                # 静默跳过后客户端重连将莫名丢事件，且无从排查
                logger.warning(
                    "SSE 历史事件解析失败，跳过 [stream_session_id=%s]: %s",
                    stream_session_id,
                    e,
                    exc_info=True,
                )
                continue
            if event.get("id", 0) > last_event_id:
                events.append(event)
        return events, bool(raw_events)

    # ── 事件推送 ────────────────────────────────────────

    async def send_event(self, stream_session_id: str, event_type: str, data: dict) -> None:
        """推送事件并缓存到 Redis，同时写入活跃连接队列"""
        event_id = await self._next_event_id(stream_session_id)
        event = {"id": event_id, "event": event_type, "data": data}
        await self._cache_event(stream_session_id, event)
        queue = self._queues.get(stream_session_id)
        if queue is not None:
            await queue.put(event)

    async def send_error(self, stream_session_id: str, error: Exception) -> None:
        """推送 error 事件并补 message.end(error) 收尾（异常→客户端载荷的唯一出口）。

        补 message.end 保证客户端总能走统一的完成处理逻辑，不因缺终结事件而挂起。
        """
        await self.send_event(stream_session_id, "error", public_error(error))
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
        """从队列读取事件并推送，空闲时发送心跳，空闲超时则结束本条连接"""
        heartbeat = settings.AI_MESSAGE_HEARTBEAT_INTERVAL
        timeout = settings.AI_MESSAGE_STREAM_TIMEOUT
        loop = asyncio.get_running_loop()
        last_activity = loop.time()
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except TimeoutError:
                if loop.time() - last_activity >= timeout:
                    # 空闲超时只结束本条连接：执行与连接解耦，后台推理继续跑完并正常
                    # 落库计费，用户重连（Last-Event-ID 重放）或刷新消息历史即可看到
                    # 完整回复。不推 error 事件——连接空闲不代表推理失败，推 error 会让
                    # 客户端把仍在生成的消息误置为失败态。
                    logger.warning("SSE 流空闲超时，结束连接: stream=%s", stream_session_id)
                    break
                # 心跳直连下发，不走 send_event：否则 ping 会回流进本队列，既被当成
                # 业务事件重复推送，又不断刷新 last_activity 使空闲超时永不触发。
                # 心跳无需进缓存（重连重放不要心跳）。
                event_id = await self._next_event_id(stream_session_id)
                yield self._format_event({"id": event_id, "event": "ping", "data": {}})
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
        """断线重连：先注册队列再重放 Redis 缓存中断点之后的事件，然后续流。

        顺序不可颠倒：重放是多次 await，期间推理可能已推送新事件，队列未建立时
        这些事件只进缓存而无活跃连接消费，重连会直接结束并漏掉这段事件。
        """
        created = stream_session_id not in self._queues
        await self.register_stream(stream_session_id)
        queue = self._queues[stream_session_id]
        try:
            events, has_cache = await self._get_cached_events(stream_session_id, last_event_id)
            for event in events:
                yield self._format_event(event)
            # 重放内容已含终结事件 → 流已结束（含停止时追加的终结事件），无需再挂队列
            if any(event.get("event") == "message.end" for event in events):
                return
            # 无缓存 + 本次新建队列（无活跃流）→ streamSessionId 已过期或从未存在：
            # 立即结束连接，否则挂队列空等到空闲超时，客户端只能拿到超时错误
            if created and not has_cache:
                return
            async for chunk in self._stream_from_queue(stream_session_id, queue):
                yield chunk
        finally:
            if created:
                self._queues.pop(stream_session_id, None)


sse_emitter_manager = SseEmitterManager()
