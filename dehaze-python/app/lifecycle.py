"""
应用生命周期管理
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.config import settings
from app.database import close_db, init_db
from app.dependencies.redis import (check_redis_health, close_redis,
                                    get_redis_client)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""

    # 初始化日志系统（必须最先执行，确保后续所有日志输出格式一致）
    from app.infrastructure.logging import setup_logging
    setup_logging(use_json_format=settings.LOG_FORMAT_JSON)

    logger.info(f"启动 {settings.APP_NAME} v{settings.APP_VERSION}")

    # 初始化数据库
    await init_db()

    # 初始化 Redis 连接并进行健康检查
    redis = await get_redis_client()
    app.state.redis = redis
    await check_redis_health()

    # 初始化任务追踪器
    from app.service.task_tracker import init_task_tracker
    task_tracker = init_task_tracker(
        shutdown_timeout=settings.GRACEFUL_SHUTDOWN_TIMEOUT
    )
    # 启动 Redis 背景状态同步（跨 Worker 全局视图）
    await task_tracker.start(redis)
    app.state.task_tracker = task_tracker

    # 初始化 WebSocket 跨 Worker 通信（Redis Pub/Sub）
    from app.service.websocket_service import init_websocket_manager
    await init_websocket_manager()

    # 检查/创建 MinIO Bucket（仅 MinIO 模式）
    from app.service.file_service import FileService
    await FileService.ensure_bucket_exists()

    # 初始化 RabbitMQ（如果启用）
    from app.infrastructure.mq.connection import init_mq
    publisher, consumer = await init_mq()
    app.state.mq_publisher = publisher
    app.state.mq_consumer = consumer

    # 初始化 XXL-Job 执行器（如果启用）
    from app.infrastructure.job.executor import init_xxljob
    xxljob_runner = await init_xxljob()
    app.state.xxljob_runner = xxljob_runner

    # 启动 GPU 指标采集器（如果启用）
    from app.infrastructure.metrics import collect_gpu_metrics
    gpu_collector = await collect_gpu_metrics(
        collect_interval=settings.PROMETHEUS_GPU_COLLECT_INTERVAL
    )
    app.state.gpu_collector = gpu_collector

    logger.info(f"✅ {settings.APP_NAME} v{settings.APP_VERSION} 启动成功")

    yield

    await _graceful_shutdown(app)


async def _graceful_shutdown(app: FastAPI) -> None:
    """
    优雅关闭流程

    uvicorn 收到 SIGTERM/SIGINT 后自动触发 lifespan 关闭，
    无需自定义信号处理器。
    """
    logger.info("=" * 50)
    logger.info("开始优雅关闭...")

    # 1. 通知任务追踪器进入关闭模式（拒绝新任务）
    from app.service.task_tracker import TaskTracker
    task_tracker: TaskTracker | None = getattr(app.state, "task_tracker", None)
    if task_tracker:
        await task_tracker.initiate_shutdown()

    # 2. 通知 WebSocket 客户端（跨 Worker 广播）
    try:
        from app.service.websocket_service import WebSocketService
        await WebSocketService.broadcast_shutdown_notification()
        logger.info("已通知 WebSocket 客户端")
    except Exception as e:
        logger.warning(f"通知 WebSocket 客户端失败: {e}")

    # 3. 等待进行中的任务完成
    if task_tracker:
        running_count = task_tracker.running_count
        if running_count > 0:
            logger.info(f"等待 {running_count} 个任务完成...")
            completed, cancelled = await task_tracker.wait_for_completion()
            logger.info(
                f"任务等待完成: completed={completed}, cancelled={cancelled}")
        else:
            logger.info("没有运行中的任务")

    # 3.5 停止 TaskTracker Redis 状态同步
    if task_tracker:
        await task_tracker.stop()

    # 3.6 关闭 WebSocket 跨 Worker 通信
    try:
        from app.service.websocket_service import close_websocket_manager
        await close_websocket_manager()
        logger.info("WebSocket 跨 Worker 通信已关闭")
    except Exception as e:
        logger.warning(f"关闭 WebSocket 跨 Worker 通信失败: {e}")

    # 4. 关闭 XXL-Job 执行器
    from app.infrastructure.job.executor import close_xxljob
    await close_xxljob()

    # 5. 关闭 RabbitMQ 连接
    from app.infrastructure.mq.connection import close_mq
    await close_mq()

    # 6. 停止 GPU 指标采集器
    from app.infrastructure.metrics import GPUMetricsCollector
    gpu_collector: GPUMetricsCollector | None = getattr(
        app.state, "gpu_collector", None)
    if gpu_collector:
        await gpu_collector.stop()
        logger.info("GPU 指标采集器已停止")

    # 7. 关闭 Redis 连接
    await close_redis()
    logger.info("Redis 连接已关闭")

    # 8. 关闭数据库连接
    await close_db()
    logger.info("数据库连接已关闭")

    logger.info("=" * 50)
    logger.info("👋 服务已优雅关闭")
