"""
RabbitMQ 消息队列模块

提供 Publisher / Consumer / 连接管理，与 Go/Java 端共享交换机和队列设计。
基于 aio-pika（asyncio 原生 AMQP 客户端），支持自动重连、指数退避。

显式导入（不做包级 re-export）：
    from app.infrastructure.mq.connection import init_mq, get_publisher
"""
