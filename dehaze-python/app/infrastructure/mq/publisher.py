from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange
from app.config import settings

from app.infrastructure.mq.base import BaseRabbitMQClient

logger = logging.getLogger(__name__)


class Publisher(BaseRabbitMQClient):
    """
    RabbitMQ 发布器

    - 连接断开时自动指数退避重连（由基类提供）
    - 发布时若 channel 不可用则等待重连完成后重试
    - 消息持久化（delivery_mode=PERSISTENT）
    """

    _name = "RabbitMQ Publisher"

    def __init__(self) -> None:
        super().__init__()
        self._exchange: Optional[AbstractExchange] = None
        self._connected_event = asyncio.Event()

    async def _on_connected(self, channel: AbstractChannel) -> None:
        exchange = await channel.declare_exchange(
            settings.RABBITMQ_EXCHANGE,
            aio_pika.ExchangeType(settings.RABBITMQ_EXCHANGE_TYPE),
            durable=True,
        )
        self._exchange = exchange
        self._connected_event.set()

    async def _on_connect_failed(self) -> None:
        self._connected_event.clear()

    def _on_disconnect(self) -> None:
        self._connected_event.clear()
        self._exchange = None

    async def _on_closing(self) -> None:
        # 释放等待中的 publish 调用
        self._connected_event.set()

    async def publish(
        self,
        routing_key: str,
        body: dict[str, Any],
        *,
        headers: Optional[dict[str, Any]] = None,
        timeout: float = 5.0,
    ) -> None:
        """
        发布消息

        Args:
            routing_key: 路由键（不含前缀，如 "export"）
            body: 消息体（自动 JSON 序列化）
            headers: 附加 AMQP Headers
            timeout: 等待连接可用的超时时间
        """
        if self._closed:
            raise RuntimeError("Publisher already closed")

        # 等待连接就绪
        if not self._connected_event.is_set():
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"RabbitMQ Publisher 连接不可用（等待 {timeout}s 超时）"
                )

        full_routing_key = f"{settings.RABBITMQ_ROUTING_KEY_PREFIX}.{routing_key}"

        message = aio_pika.Message(
            body=json.dumps(body, default=str).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers=headers or {},
        )

        assert self._exchange is not None
        await self._exchange.publish(message, routing_key=full_routing_key)
        logger.debug(f"消息已发布: routing_key={full_routing_key}")
