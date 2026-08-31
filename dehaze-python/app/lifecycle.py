"""
应用生命周期管理
"""

import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.database import close_db, init_db
from app.dependencies.mongo import close_mongo, init_mongo_indexes
from app.dependencies.redis import check_redis_health, close_redis, get_redis_client

logger = logging.getLogger(__name__)

# 主 Worker 文件锁句柄（保持打开以持有锁，进程退出时自动释放）
_main_worker_lock_file = None


def _register_soft_delete_filter() -> None:
    """注册全局逻辑删除过滤器。

    对所有继承 SoftDeleteMixin 的实体的 ORM SELECT 查询自动追加 deleted=0 条件，
    等价于 Java MyBatis-Plus 的全局逻辑删除。
    需要查已删除数据时，使用 execution_options(include_deleted=True) 绕过。
    """
    from sqlalchemy import event
    from sqlalchemy.orm import Session, with_loader_criteria

    from app.models.base import SoftDeleteMixin

    def _soft_delete_criteria(execute_state):
        if not execute_state.is_select:
            return
        if execute_state.execution_options.get("include_deleted"):
            return
        execute_state.statement = execute_state.statement.options(
            with_loader_criteria(
                SoftDeleteMixin,
                lambda cls: cls.deleted == 0,
                include_aliases=True,
            )
        )

    event.listen(Session, "do_orm_execute", _soft_delete_criteria)


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
    except OSError:
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

    await init_db()

    # 注册全局逻辑删除过滤器（等价于 Java MyBatis-Plus @TableLogic / Go GORM 软删除）
    _register_soft_delete_filter()

    from app.service.preset_service import preset_service

    await preset_service.seed_system_presets()

    # 内置 Skill 播种 + 预热 SkillManager 内存索引
    from app.database import get_db_session
    from app.service.ai.service.skill_manager import skill_manager
    from app.service.ai_skill_service import skill_manage_service

    async with get_db_session() as db:
        await skill_manage_service.ensure_builtin_skills(db)
        await skill_manager.refresh_index(db)

    # 内置本地模型：幂等播种 local provider/Key/LLM+Embedding 模型（默认模型路由目标）
    from app.infrastructure.llm.local.model_seeder import ensure_local_models

    async with get_db_session() as db:
        await ensure_local_models(db)

    # 内置本地语音引擎：幂等播种 local asr/tts provider 与默认模型/音色（语音引擎注册表）
    try:
        from app.infrastructure.voice.seeder import ensure_local_engines

        async with get_db_session() as db:
            await ensure_local_engines(db)
    except Exception as exc:  # noqa: BLE001 播种失败仅告警，不影响服务启动
        logger.warning("内置本地语音引擎播种失败（不影响启动）: %s", exc)

    # 主 Worker 后台预下载模型文件（不阻塞启动；首次对话时 ensure_running 兜底）
    if is_main_worker:
        import threading

        from app.infrastructure.llm.local.local_llm_model import (
            ensure_embedding_model,
            ensure_model,
            is_downloaded,
            is_embedding_downloaded,
        )

        def _prefetch_local_model() -> None:
            try:
                if not is_downloaded():
                    logger.info("后台预下载内置本地对话模型（Qwen3-0.6B，约 378MB）")
                ensure_model()
                if not is_embedding_downloaded():
                    logger.info("后台预下载内置本地向量模型（Qwen3-Embedding-0.6B，约 610MB）")
                ensure_embedding_model()
            except Exception as exc:  # noqa: BLE001 预下载失败不影响启动，首次调用时会重试
                logger.warning("本地模型预下载失败（首次调用时将重试）: %s", exc)

        threading.Thread(target=_prefetch_local_model, name="local-llm-prefetch", daemon=True).start()

    redis = await get_redis_client()
    app.state.redis = redis
    await check_redis_health()

    # 幂等补齐系统预置字典默认项（ai_guardrail_defaults / ai_provider_health / ai_embedding
    # / member_growth_rules / favorite_capacity）
    # 必须在 ensure_default_agent 之前执行：默认 Agent 初始发布快照的 resolved_config
    # 依赖这些 sys_dict 默认参数，缺失会导致推理配置缺键而快速失败。
    from app.database import get_db_session
    from app.service.dict_service import ensure_system_dict_defaults

    async with get_db_session() as db:
        await ensure_system_dict_defaults(db, redis)

    # 确保默认 Agent 存在（agent_code='default'，不可删除）
    from app.database import get_db_session
    from app.service.ai_agent_service import agent_service

    async with get_db_session() as db:
        await agent_service.ensure_default_agent(db, redis)

    # 启动缓存失效广播订阅（多实例 L1 缓存一致性）
    from app.infrastructure.cache.cache import start_cache_invalidation_listener

    await start_cache_invalidation_listener()

    await init_mongo_indexes()

    # 初始化 ES 索引（记忆向量 / 会话全文，未启用时静默跳过）
    from app.infrastructure.es.ai_conversation_index import ensure_conversation_index
    from app.service.ai.service.memory_es_service import ensure_memory_index

    await ensure_memory_index()
    await ensure_conversation_index()

    from app.service.task_tracker import init_task_tracker

    task_tracker = init_task_tracker(shutdown_timeout=settings.GRACEFUL_SHUTDOWN_TIMEOUT)
    # 启动 Redis 背景状态同步（跨 Worker 全局视图）
    await task_tracker.start(redis)
    app.state.task_tracker = task_tracker

    # 初始化 WebSocket 跨 Worker 通信（Redis Pub/Sub）
    from app.service.websocket_service import init_websocket_manager

    await init_websocket_manager()

    from app.service.file_service import file_service

    await file_service.ensure_bucket_exists()

    from app.infrastructure.mq.connection import init_mq

    publisher, consumer = await init_mq()
    app.state.mq_publisher = publisher
    app.state.mq_consumer = consumer

    # 初始化 XXL-Job 执行器（仅在主 Worker 启动，避免端口冲突）
    xxljob_runner: object | None = None
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
        from app.service.websocket_service import websocket_service

        await websocket_service.broadcast_shutdown_notification()
        logger.info("已通知 WebSocket 客户端")
    except Exception as e:
        logger.warning("通知 WebSocket 客户端失败: %s", e)

    # 3. 等待进行中的任务完成
    if task_tracker:
        running_count = task_tracker.running_count
        if running_count > 0:
            logger.info("等待 %s 个任务完成...", running_count)
            completed, cancelled = await task_tracker.wait_for_completion()
            logger.info("任务等待完成: completed=%s, cancelled=%s", completed, cancelled)
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

    gpu_collector: GPUMetricsCollector | None = getattr(app.state, "gpu_collector", None)
    if gpu_collector:
        await gpu_collector.stop()
        logger.info("GPU 指标采集器已停止")

    # 7. 关闭 Redis 连接
    # 7.1 停止缓存失效广播订阅
    try:
        from app.infrastructure.cache.cache import stop_cache_invalidation_listener

        await stop_cache_invalidation_listener()
        logger.info("缓存失效广播订阅已停止")
    except Exception as e:
        logger.warning("停止缓存失效广播订阅失败: %s", e)

    # 7.2 关闭 Redis 连接
    await close_redis()
    logger.info("Redis 连接已关闭")

    # 8. 关闭 MongoDB 连接
    await close_mongo()
    logger.info("MongoDB 连接已关闭")

    # 9. 关闭数据库连接
    await close_db()
    logger.info("数据库连接已关闭")

    # 10. 回收本地 LLM 子进程（对话与 embedding 推理同进程；TTS 为库内推理无子进程）
    from app.infrastructure.llm.local.local_llm_manager import shutdown as shutdown_local_llm

    shutdown_local_llm()

    logger.info("=" * 50)
    logger.info("👋 服务已优雅关闭")
