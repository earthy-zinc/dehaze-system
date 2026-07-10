from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Optional, Union

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractConnection, AbstractExchange
from app.config import settings

logger = logging.getLogger(__name__)


class Publisher:
    """
    RabbitMQ 发布器

    - 连接断开时自动指数退避重连
    - 发布时若 channel 不可用则等待重连完成后重试
    - 消息持久化（delivery_mode=PERSISTENT）
    """

    def __init__(self) -> None:
        self._conn: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._exchange: Optional[AbstractExchange] = None
        self._closed = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._connected_event = asyncio.Event()

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
            raise RuntimeError("Publisher already closed")

        try:
            conn = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                reconnect_interval=settings.RABBITMQ_RECONNECT_INITIAL_INTERVAL,
            )
            # 监听连接关闭事件
            conn.close_callbacks.add(self._on_connection_lost)

            channel = await conn.channel()
            exchange = await channel.declare_exchange(
                settings.RABBITMQ_EXCHANGE,
                aio_pika.ExchangeType(settings.RABBITMQ_EXCHANGE_TYPE),
                durable=True,
            )

            self._conn = conn
            self._channel = channel
            self._exchange = exchange

            self._connected_event.set()
            logger.info(
                f"RabbitMQ Publisher 已连接: exchange={settings.RABBITMQ_EXCHANGE}"
            )
        except Exception as e:
            logger.error(f"RabbitMQ Publisher 连接失败: {e}")
            self._connected_event.clear()
            raise

    def _on_connection_lost(self, *args: Any) -> None:
        """连接断开回调，触发重连"""
        if self._closed:
            return
        logger.warning("RabbitMQ Publisher 连接断开，启动自动重连")
        self._connected_event.clear()
        self._exchange = None
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
                    f"RabbitMQ Publisher 重连已达最大重试次数({max_retries})，放弃重连"
                )
                return

            interval = min(initial * math.pow(2, attempt), max_interval)
            attempt += 1
            logger.info(
                f"RabbitMQ Publisher 重连等待: attempt={attempt}, interval={interval:.1f}s"
            )
            await asyncio.sleep(interval)

            try:
                await self.connect()
                logger.info(f"RabbitMQ Publisher 重连成功: attempt={attempt}")
                return
            except Exception as e:
                logger.warning(
                    f"RabbitMQ Publisher 重连失败: attempt={attempt}, {e}")

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

    async def close(self) -> None:
        self._closed = True
        self._connected_event.set()  # 释放等待中的 publish 调用

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._conn and not self._conn.is_closed:
            await self._conn.close()

        logger.info("RabbitMQ Publisher 已关闭")
