"""
应用生命周期管理
"""

import errno
import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import settings
from app.database import close_db, init_db
from app.dependencies.mongo import close_mongo, init_mongo_indexes
from app.dependencies.redis import check_redis_health, close_redis, get_redis_client

logger = logging.getLogger(__name__)

# 主 Worker 文件锁句柄（保持打开以持有锁，进程退出时自动释放）
_main_worker_lock_file = None


_soft_delete_filter_registered = False


def _register_soft_delete_filter() -> None:
    """注册全局逻辑删除过滤器。

    对所有继承 SoftDeleteMixin 的实体的 ORM SELECT 查询自动追加 deleted=0 条件，
    等价于 Java MyBatis-Plus 的全局逻辑删除。
    需要查已删除数据时，使用 execution_options(include_deleted=True) 绕过。
    """
    global _soft_delete_filter_registered
    if _soft_delete_filter_registered:
        return

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
    _soft_delete_filter_registered = True


def _try_become_main_worker() -> bool:
    """尝试成为主 Worker（通过文件锁互斥）。

    在多 Worker 部署（uvicorn --workers N）下，某些资源只需启动一次：
    - XXL-Job executor daemon（绑定端口，多实例会冲突）

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

    lock_path = str(Path(settings.LOG_DIR) / "main_worker.lock")
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)

    fd = None
    try:
        # 用 os.open 持有文件描述符：flock 需要进程存活期间保持 fd 打开
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, str(os.getpid()).encode())
        _main_worker_lock_file = fd
        return True
    except OSError as e:
        if fd is not None:
            os.close(fd)
        _main_worker_lock_file = None
        # flock LOCK_NB 抢锁失败（EACCES/EAGAIN/EWOULDBLOCK）＝"已有主 Worker"，属正常；
        # 其余 OSError（日志目录不可写/磁盘满等）会让本进程静默降级为非主 Worker，
        # 导致 XXL-Job 等独占资源无人启动，必须暴露
        if e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            logger.debug("未获取主 Worker 锁（已有主 Worker 持有）: %s", e)
        else:
            logger.error("主 Worker 锁文件操作失败，本进程降级为非主 Worker: %s", e, exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""

    # 初始化日志系统（必须最先执行，确保后续所有日志输出格式一致）
    from app.infrastructure.logging import setup_logging

    setup_logging(use_json_format=settings.LOG_FORMAT_JSON)

    logger.info("启动 %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # 多 Worker 守卫：判断当前 Worker 是否为主 Worker
    # XXL-Job daemon 只需在主 Worker 中启动
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
        # 历史 name-key 资源对象迁到 skill_id-key（幂等，迁移后为空操作）
        try:
            await skill_manage_service.migrate_legacy_object_keys(db)
        except Exception as exc:
            logger.warning("SKILL 资源对象 key 迁移失败（不影响启动）: %s", exc)
        await skill_manager.refresh_index(db)

    # 内置本地模型/语音/图像供应商的初始化不在启动期执行（配置权威源是数据库与
    # config/sql/data/ 种子数据，见 dehaze-doc）；需要初始化或轮换内置服务密钥时显式执行：
    #   dehaze-python/.venv/bin/python -m app.cli init-builtin

    redis = await get_redis_client()
    app.state.redis = redis
    await check_redis_health()

    # 供应商健康开关缓存预热：DB 是权威源、Redis 仅缓存（缓存缺省语义为"开启"），
    # 启动同步一次，避免迁移直改 DB / Redis 清空后运行行为与配置漂移
    try:
        from app.infrastructure.provider.provider_health_service import (
            warmup_provider_health_cache,
        )

        await warmup_provider_health_cache(redis)
    except Exception as exc:
        logger.warning("供应商健康开关缓存预热失败（不影响启动）: %s", exc)

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

    # 启动 MCP Server 健康巡检（仅主 Worker，避免多实例重复探测同一 Server）
    if is_main_worker:
        from app.service.ai_mcp.mcp_health_checker import mcp_health_checker

        mcp_health_checker.start()
        app.state.mcp_health_checker = mcp_health_checker

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

    # 3.7 等待在途观测埋点落盘（trace/llm_call 后台写库），避免连接池回收后写入报错
    try:
        from app.service.ai.service.trace_collector import drain as drain_trace_writes

        await drain_trace_writes()
        logger.info("观测埋点在途落盘已刷新")
    except Exception as e:
        logger.warning("等待观测埋点落盘失败: %s", e)

    # 4. 关闭 XXL-Job 执行器（仅在主 Worker 中启动了才需关闭）
    if getattr(app.state, "xxljob_runner", None) is not None:
        from app.infrastructure.job.executor import close_xxljob

        await close_xxljob()

    # 5. 关闭 RabbitMQ 连接
    from app.infrastructure.mq.connection import close_mq

    await close_mq()

    # 6. 停止 MCP Server 健康巡检
    from app.service.ai_mcp.mcp_health_checker import mcp_health_checker

    if getattr(app.state, "mcp_health_checker", None) is not None:
        await mcp_health_checker.stop()
        logger.info("MCP Server 健康巡检已停止")

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

    # 推理服务（dehaze-llm / voice / modal）由 scripts/run.py 独立管理生命周期，
    # 业务进程退出不回收：reload / 重启不影响模型服务可用性

    logger.info("=" * 50)
    logger.info("👋 服务已优雅关闭")
