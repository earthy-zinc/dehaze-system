"""
应用生命周期管理
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI

from app.config import settings
from app.database import close_db, init_db
from app.dependencies.redis import (check_redis_health, close_redis,
                                    get_redis_client)

logger = logging.getLogger(__name__)

# 主 Worker 文件锁句柄（保持打开以持有锁，进程退出时自动释放）
_main_worker_lock_file = None


def _try_become_main_worker() -> bool:
    """尝试成为主 Worker（通过文件锁互斥）。

    在多 Worker 部署（uvicorn --workers N）下，某些资源只需启动一次：
    - XXL-Job executor daemon（绑定端口，多实例会冲突）
    - GPU 指标采集器（重复采集同一块 GPU 无意义）

    Linux/Mac 使用 fcntl 文件锁，第一个获取锁的 Worker 为主 Worker。
    Windows 总是返回 True（开发环境通常使用 --workers 1）。
    主 Worker 崩溃后锁自动释放，但其他 Worker 不会自动接管（需重启）。
    """
    global _main_worker_lock_file

    if sys.platform == "win32":
        return True

    try:
        import fcntl
    except ImportError:
        return True

    lock_path = os.path.join(settings.LOG_DIR, "main_worker.lock")
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)

    try:
        _main_worker_lock_file = open(lock_path, "w")
        fcntl.flock(_main_worker_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _main_worker_lock_file.write(str(os.getpid()))
        _main_worker_lock_file.flush()
        return True
    except (IOError, OSError):
        if _main_worker_lock_file:
            _main_worker_lock_file.close()
            _main_worker_lock_file = None
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""

    # 初始化日志系统（必须最先执行，确保后续所有日志输出格式一致）
    from app.infrastructure.logging import setup_logging
    setup_logging(use_json_format=settings.LOG_FORMAT_JSON)

    logger.info("启动 %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # 多 Worker 守卫：判断当前 Worker 是否为主 Worker
    # XXL-Job daemon 和 GPU 指标采集器只需在主 Worker 中启动
    is_main_worker = _try_become_main_worker()
    if is_main_worker:
        logger.info("当前 Worker 为主 Worker (pid=%d)，将启动独占资源", os.getpid())
    else:
        logger.info("当前 Worker 为从 Worker (pid=%d)，跳过独占资源启动", os.getpid())

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

    # 初始化 XXL-Job 执行器（仅在主 Worker 启动，避免端口冲突）
    xxljob_runner: Optional[object] = None
    if is_main_worker:
        from app.infrastructure.job.executor import init_xxljob
        xxljob_runner = await init_xxljob()
    else:
        if settings.XXLJOB_ENABLED:
            logger.info("从 Worker 跳过 XXL-Job 启动（由主 Worker 负责）")
    app.state.xxljob_runner = xxljob_runner

    # 启动 GPU 指标采集器（仅在主 Worker 启动，避免重复采集）
    gpu_collector = None
    if is_main_worker:
        from app.infrastructure.metrics import collect_gpu_metrics
        gpu_collector = await collect_gpu_metrics(
            collect_interval=settings.PROMETHEUS_GPU_COLLECT_INTERVAL
        )
    else:
        logger.info("从 Worker 跳过 GPU 指标采集器启动（由主 Worker 负责）")
    app.state.gpu_collector = gpu_collector

    logger.info("✅ %s v%s 启动成功", settings.APP_NAME, settings.APP_VERSION)

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
        logger.warning("通知 WebSocket 客户端失败: %s", e)

    # 3. 等待进行中的任务完成
    if task_tracker:
        running_count = task_tracker.running_count
        if running_count > 0:
            logger.info("等待 %s 个任务完成...", running_count)
            completed, cancelled = await task_tracker.wait_for_completion()
            logger.info(
                "任务等待完成: completed=%s, cancelled=%s", completed, cancelled)
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
        logger.warning("关闭 WebSocket 跨 Worker 通信失败: %s", e)

    # 4. 关闭 XXL-Job 执行器（仅在主 Worker 中启动了才需关闭）
    if getattr(app.state, "xxljob_runner", None) is not None:
        from app.infrastructure.job.executor import close_xxljob
        await close_xxljob()

    # 5. 关闭 RabbitMQ 连接
    from app.infrastructure.mq.connection import close_mq
    await close_mq()

    # 6. 停止 GPU 指标采集器（仅在主 Worker 中启动了才需停止）
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
