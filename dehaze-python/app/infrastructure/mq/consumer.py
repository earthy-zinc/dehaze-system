from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine

import aio_pika
from aio_pika.abc import (AbstractChannel, AbstractIncomingMessage,
                          AbstractQueue)
from app.config import settings

from app.infrastructure.mq.base import BaseRabbitMQClient

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any], dict[str, Any]], Coroutine[Any, Any, None]]


class Consumer(BaseRabbitMQClient):
    """
    RabbitMQ 消费者

    - 连接断开时自动指数退避重连（由基类提供）
    - 支持注册多个队列的 handler
    - nack 时不 requeue，通过重试队列分级延迟重投
    - prefetch 限流
    """

    _name = "RabbitMQ Consumer"

    def __init__(self) -> None:
        super().__init__()
        self._handlers: dict[str, Handler] = {}
        self._dlq_handlers: set[str] = set()
        self._queues: dict[str, AbstractQueue] = {}

    async def _on_connected(self, channel: AbstractChannel) -> None:
        await channel.set_qos(prefetch_count=settings.RABBITMQ_PREFETCH_COUNT)
        if self._handlers:
            await self._resubscribe_all()

    def _on_disconnect(self) -> None:
        self._queues.clear()

    async def _on_closing(self) -> None:
        for queue_name, queue in self._queues.items():
            try:
                await queue.cancel(queue_name)
            except Exception:
                pass
        self._queues.clear()

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

    async def register_dlq(self, queue_name: str, handler: Handler) -> None:
        """
        注册死信队列消费 handler（不创建 DLX/重试队列，避免递归声明）

        Args:
            queue_name: 死信队列名称（如 "task.execute.dlx"）
            handler: 消息处理函数
        """
        self._handlers[queue_name] = handler
        self._dlq_handlers.add(queue_name)

        if self.is_connected:
            await self._subscribe_dlq(queue_name, handler)

    async def _subscribe_dlq(self, queue_name: str, handler: Handler) -> None:
        """订阅死信队列（简单声明并消费，不创建 DLX/重试队列）"""
        if self._channel is None:
            raise RuntimeError("Channel not connected")

        exchange = await self._channel.declare_exchange(
            settings.RABBITMQ_EXCHANGE,
            aio_pika.ExchangeType(settings.RABBITMQ_EXCHANGE_TYPE),
            durable=True,
        )

        queue = await self._channel.declare_queue(queue_name, durable=True)
        routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{queue_name}"
        await queue.bind(exchange, routing_key=routing_key)
        await queue.consume(self._make_callback(queue_name, handler))
        self._queues[queue_name] = queue

        logger.info(
            f"Consumer 已订阅死信队列: queue={queue_name}, routing_key={routing_key}"
        )

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
        main_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{queue_name}"
        for i, delay_ms in enumerate(retry_delays):
            retry_queue_name = f"{queue_name}.retry.{i}"
            retry_queue_obj = await self._channel.declare_queue(
                retry_queue_name,
                durable=True,
                arguments={
                    "x-message-ttl": delay_ms,
                    "x-dead-letter-exchange": settings.RABBITMQ_EXCHANGE,
                    "x-dead-letter-routing-key": main_routing_key,
                },
            )
            retry_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{retry_queue_name}"
            await retry_queue_obj.bind(exchange, routing_key=retry_routing_key)

        # 声明主队列（配置 DLX：过期或 reject 的消息进入死信队列）
        queue = await self._channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-message-ttl": 86400000,  # 24h TTL
                "x-dead-letter-exchange": settings.RABBITMQ_EXCHANGE,
                "x-dead-letter-routing-key": dlx_routing_key,
            },
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
                if queue_name in self._dlq_handlers:
                    await self._subscribe_dlq(queue_name, handler)
                else:
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
                headers: dict[str, Any] = dict(message.headers) if message.headers else {}
                await handler(body, headers)
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
