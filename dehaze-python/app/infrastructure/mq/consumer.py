from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Callable, Coroutine, Optional

import aio_pika
from aio_pika.abc import (AbstractChannel, AbstractConnection,
                          AbstractIncomingMessage, AbstractQueue)
from app.config import settings

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class Consumer:
    """
    RabbitMQ 消费者

    - 连接断开时自动指数退避重连
    - 支持注册多个队列的 handler
    - nack 时不 requeue，通过重试队列分级延迟重投
    - prefetch 限流
    """

    def __init__(self) -> None:
        self._conn: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._closed = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._handlers: dict[str, Handler] = {}
        self._queues: dict[str, AbstractQueue] = {}

    @property
    def is_connected(self) -> bool:
        return (
            self._conn is not None
            and not self._conn.is_closed
            and self._channel is not None
            and not self._channel.is_closed
        )

    async def connect(self) -> None:
        if self._closed:
            raise RuntimeError("Consumer already closed")

        try:
            conn = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                reconnect_interval=settings.RABBITMQ_RECONNECT_INITIAL_INTERVAL,
            )
            conn.close_callbacks.add(self._on_connection_lost)

            channel = await conn.channel()
            await channel.set_qos(prefetch_count=settings.RABBITMQ_PREFETCH_COUNT)

            self._conn = conn
            self._channel = channel

            logger.info("RabbitMQ Consumer 已连接")

            if self._handlers:
                await self._resubscribe_all()

        except Exception as e:
            logger.error(f"RabbitMQ Consumer 连接失败: {e}")
            raise

    def _on_connection_lost(self, *args: Any) -> None:
        if self._closed:
            return
        logger.warning("RabbitMQ Consumer 连接断开，启动自动重连")
        self._queues.clear()
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """指数退避重连循环"""
        attempt = 0
        max_retries = settings.RABBITMQ_RECONNECT_MAX_RETRIES
        initial = settings.RABBITMQ_RECONNECT_INITIAL_INTERVAL
        max_interval = settings.RABBITMQ_RECONNECT_MAX_INTERVAL

        while not self._closed:
            if max_retries > 0 and attempt >= max_retries:
                logger.error(
                    f"RabbitMQ Consumer 重连已达最大重试次数({max_retries})，放弃重连"
                )
                return

            interval = min(initial * math.pow(2, attempt), max_interval)
            attempt += 1
            logger.info(
                f"RabbitMQ Consumer 重连等待: attempt={attempt}, interval={interval:.1f}s"
            )
            await asyncio.sleep(interval)

            try:
                await self.connect()
                logger.info(f"RabbitMQ Consumer 重连成功: attempt={attempt}")
                return
            except Exception as e:
                logger.warning(
                    f"RabbitMQ Consumer 重连失败: attempt={attempt}, {e}")

    async def register(self, queue_name: str, handler: Handler) -> None:
        """
        注册队列消费 handler

        Args:
            queue_name: 队列名称（如 "task.execute"）
            handler: 消息处理函数
        """
        self._handlers[queue_name] = handler

        if self.is_connected:
            await self._subscribe(queue_name, handler)

    async def _subscribe(self, queue_name: str, handler: Handler) -> None:
        """订阅单个队列，同时声明重试队列和死信队列"""
        if self._channel is None:
            raise RuntimeError("Channel not connected")

        exchange = await self._channel.declare_exchange(
            settings.RABBITMQ_EXCHANGE,
            aio_pika.ExchangeType(settings.RABBITMQ_EXCHANGE_TYPE),
            durable=True,
        )

        # 声明死信队列
        dlx_queue_name = f"{queue_name}.dlx"
        dlx_queue = await self._channel.declare_queue(
            dlx_queue_name,
            durable=True,
        )
        dlx_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{dlx_queue_name}"
        await dlx_queue.bind(exchange, routing_key=dlx_routing_key)

        # 声明重试队列（带 TTL，过期后重新路由到主队列）
        retry_delays = settings.RABBITMQ_RETRY_DELAYS
        for i, delay_ms in enumerate(retry_delays):
            retry_queue_name = f"{queue_name}.retry.{i}"
            main_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{queue_name}"
            await self._channel.declare_queue(
                retry_queue_name,
                durable=True,
                arguments={
                    "x-message-ttl": delay_ms,
                    "x-dead-letter-exchange": settings.RABBITMQ_EXCHANGE,
                    "x-dead-letter-routing-key": main_routing_key,
                },
            )
            retry_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{retry_queue_name}"
            # 绑定重试队列到 exchange
            retry_queue_obj = await self._channel.declare_queue(
                retry_queue_name, durable=True, passive=True,
            )
            await retry_queue_obj.bind(exchange, routing_key=retry_routing_key)

        # 声明主队列
        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            arguments={"x-message-ttl": 86400000},  # 24h TTL
        )

        routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{queue_name}"
        await queue.bind(exchange, routing_key=routing_key)

        await queue.consume(self._make_callback(queue_name, handler))
        self._queues[queue_name] = queue

        logger.info(
            f"Consumer 已订阅队列: queue={queue_name}, routing_key={routing_key}, "
            f"retry_levels={len(retry_delays)}, dlx={dlx_queue_name}"
        )

    async def _resubscribe_all(self) -> None:
        self._queues.clear()
        for queue_name, handler in self._handlers.items():
            try:
                await self._subscribe(queue_name, handler)
            except Exception as e:
                logger.error(f"重新订阅队列失败: queue={queue_name}, {e}")

    def _make_callback(
        self, queue_name: str, handler: Handler
    ) -> Callable[[AbstractIncomingMessage], Coroutine[Any, Any, None]]:
        """为 handler 包装 ack/nack + 分级重试逻辑"""

        async def callback(message: AbstractIncomingMessage) -> None:
            try:
                body = json.loads(message.body.decode())
                await handler(body)
                await message.ack()
            except json.JSONDecodeError:
                # 无法解析的消息，直接丢弃
                logger.error(
                    f"消息反序列化失败（已丢弃）: queue={queue_name}, "
                    f"body={message.body[:200]}"
                )
                await message.reject(requeue=False)
            except Exception as e:
                logger.error(
                    f"消息处理失败: queue={queue_name}, error={e}",
                    exc_info=True,
                )
                # 分级重试：从 headers 读取 retry count
                await self._retry_or_dlx(message, queue_name)

        return callback

    async def _retry_or_dlx(
        self, message: AbstractIncomingMessage, queue_name: str
    ) -> None:
        """根据重试次数决定投递到重试队列还是死信队列"""
        if self._channel is None:
            logger.error("Channel not connected, cannot retry message")
            await message.reject(requeue=False)
            return

        headers: dict[str, Any] = dict(
            message.headers) if message.headers else {}
        retry_count = int(headers.get("x-retry-count", 0))
        retry_delays = settings.RABBITMQ_RETRY_DELAYS

        if retry_count < len(retry_delays):
            # 投递到对应级别的重试队列
            retry_queue = f"{queue_name}.retry.{retry_count}"
            retry_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{retry_queue}"

            exchange = await self._channel.declare_exchange(
                settings.RABBITMQ_EXCHANGE,
                aio_pika.ExchangeType(settings.RABBITMQ_EXCHANGE_TYPE),
                durable=True,
            )

            new_headers = {**headers, "x-retry-count": str(retry_count + 1)}
            retry_message = aio_pika.Message(
                body=message.body,
                content_type=message.content_type,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=new_headers,
            )
            await exchange.publish(retry_message, routing_key=retry_routing_key)
            await message.ack()

            logger.info(
                f"消息已投递到重试队列: queue={retry_queue}, "
                f"retryCount={retry_count + 1}, "
                f"delay={retry_delays[retry_count]}ms"
            )
        else:
            # 超过最大重试次数，投递到死信队列
            dlx_queue = f"{queue_name}.dlx"
            dlx_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{dlx_queue}"

            exchange = await self._channel.declare_exchange(
                settings.RABBITMQ_EXCHANGE,
                aio_pika.ExchangeType(settings.RABBITMQ_EXCHANGE_TYPE),
                durable=True,
            )

            dlx_headers = {**headers, "x-retry-count": str(retry_count)}
            dlx_message = aio_pika.Message(
                body=message.body,
                content_type=message.content_type,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=dlx_headers,
            )
            await exchange.publish(dlx_message, routing_key=dlx_routing_key)
            await message.ack()

            logger.warning(
                f"消息已投递到死信队列（重试耗尽）: queue={dlx_queue}, "
                f"totalRetries={retry_count}"
            )

    async def close(self) -> None:
        self._closed = True

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        for queue_name, queue in self._queues.items():
            try:
                await queue.cancel(queue_name)
            except Exception:
                pass
        self._queues.clear()

        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._conn and not self._conn.is_closed:
            await self._conn.close()

        logger.info("RabbitMQ Consumer 已关闭")
