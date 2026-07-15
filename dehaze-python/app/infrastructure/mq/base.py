"""
RabbitMQ 客户端基类

封装 Publisher / Consumer 共享的连接生命周期管理：
- 自动重连（指数退避）
- 连接断开回调
- 优雅关闭

子类通过覆盖钩子方法实现差异化行为：
- _on_connected:    连接成功后初始化（声明 exchange / 设置 qos / 重新订阅）
- _on_connect_failed: 连接失败后状态清理
- _on_disconnect:    连接断开后状态清理
- _on_closing:       关闭前资源清理
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Optional

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractConnection
from app.config import settings

logger = logging.getLogger(__name__)


class BaseRabbitMQClient:
    """RabbitMQ 客户端基类，管理连接生命周期与自动重连"""

    # 子类覆盖：用于日志标识
    _name: str = "RabbitMQ"

    def __init__(self) -> None:
        self._conn: Optional[AbstractConnection] = None
        self._channel: Optional[AbstractChannel] = None
        self._closed = False
        self._reconnect_task: Optional[asyncio.Task] = None

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
            raise RuntimeError(f"{self._name} already closed")

        try:
            conn = await aio_pika.connect_robust(
                settings.RABBITMQ_URL,
                reconnect_interval=settings.RABBITMQ_RECONNECT_INITIAL_INTERVAL,
            )
            conn.close_callbacks.add(self._on_connection_lost)

            channel = await conn.channel()

            self._conn = conn
            self._channel = channel

            # 子类钩子：声明 exchange / 设置 qos / 重新订阅
            await self._on_connected(channel)

            logger.info(f"{self._name} 已连接")
        except Exception as e:
            logger.error(f"{self._name} 连接失败: {e}")
            await self._on_connect_failed()
            raise

    # ===== 子类钩子 =====

    async def _on_connected(self, channel: AbstractChannel) -> None:
        """子类覆盖：连接成功后的初始化"""

    async def _on_connect_failed(self) -> None:
        """子类覆盖：连接失败后的状态清理"""

    def _on_disconnect(self) -> None:
        """子类覆盖：连接断开后的状态清理"""

    async def _on_closing(self) -> None:
        """子类覆盖：关闭前的资源清理"""

    # ===== 连接断开与重连 =====

    def _on_connection_lost(self, *args: Any) -> None:
        """连接断开回调，触发重连"""
        if self._closed:
            return
        logger.warning(f"{self._name} 连接断开，启动自动重连")
        self._on_disconnect()
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
                    f"{self._name} 重连已达最大重试次数({max_retries})，放弃重连"
                )
                return

            interval = min(initial * math.pow(2, attempt), max_interval)
            attempt += 1
            logger.info(
                f"{self._name} 重连等待: attempt={attempt}, interval={interval:.1f}s"
            )
            await asyncio.sleep(interval)

            try:
                await self.connect()
                logger.info(f"{self._name} 重连成功: attempt={attempt}")
                return
            except Exception as e:
                logger.warning(
                    f"{self._name} 重连失败: attempt={attempt}, {e}")

    # ===== 关闭 =====

    async def close(self) -> None:
        self._closed = True
        await self._on_closing()

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

        logger.info(f"{self._name} 已关闭")
