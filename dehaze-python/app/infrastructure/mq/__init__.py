"""
RabbitMQ 消息队列模块

提供 Publisher / Consumer / 连接管理，与 Go/Java 端共享交换机和队列设计。
基于 aio-pika（asyncio 原生 AMQP 客户端），支持自动重连、指数退避。
"""

from app.infrastructure.mq.connection import (close_mq, get_consumer,
                                              get_publisher, init_mq)
from app.infrastructure.mq.consumer import Consumer
from app.infrastructure.mq.publisher import Publisher

__all__ = [
    "Publisher",
    "Consumer",
    "init_mq",
    "close_mq",
    "get_publisher",
    "get_consumer",
]
