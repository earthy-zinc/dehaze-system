"""
RabbitMQ 连接管理

全局单例管理 Publisher / Consumer 生命周期，供 Lifespan 调用。
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.infrastructure.mq.consumer import Consumer
from app.infrastructure.mq.publisher import Publisher

logger = logging.getLogger(__name__)

_publisher: Optional[Publisher] = None
_consumer: Optional[Consumer] = None


def get_publisher() -> Optional[Publisher]:
    """获取全局 Publisher 实例（未启用 RabbitMQ 时返回 None）"""
    return _publisher


def get_consumer() -> Optional[Consumer]:
    """获取全局 Consumer 实例（未启用 RabbitMQ 时返回 None）"""
    return _consumer


async def init_mq() -> tuple[Optional[Publisher], Optional[Consumer]]:
    """
    初始化 RabbitMQ 连接（在 Lifespan 启动阶段调用）

    Returns:
        (publisher, consumer) 元组，未启用时均为 None
    """
    global _publisher, _consumer

    if not settings.RABBITMQ_ENABLED:
        logger.info("RabbitMQ 未启用，跳过初始化")
        return None, None

    # 初始化 Publisher
    _publisher = Publisher()
    try:
        await _publisher.connect()
    except Exception as e:
        logger.error(f"RabbitMQ Publisher 初始化失败（服务继续启动）: {e}")
        _publisher = None

    # 初始化 Consumer
    _consumer = Consumer()
    try:
        await _consumer.connect()
    except Exception as e:
        logger.error(f"RabbitMQ Consumer 初始化失败（服务继续启动）: {e}")
        _consumer = None

    # 注册消费者 handler
    if _consumer is not None:
        await _register_handlers(_consumer)

    return _publisher, _consumer


async def _register_handlers(consumer: Consumer) -> None:
    """注册所有队列的消费者 handler"""
    from app.infrastructure.mq.handlers import handle_dlq_message, handle_export_task

    await consumer.register("task.execute", handle_export_task)
    logger.info("已注册消费者 handler: task.execute")

    await consumer.register_dlq("task.execute.dlx", handle_dlq_message)
    logger.info("已注册死信队列 handler: task.execute.dlx")


async def close_mq() -> None:
    """关闭 RabbitMQ 连接（在 Lifespan 关闭阶段调用）"""
    global _publisher, _consumer

    if _consumer is not None:
        await _consumer.close()
        _consumer = None

    if _publisher is not None:
        await _publisher.close()
        _publisher = None

    logger.info("RabbitMQ 连接已关闭")
