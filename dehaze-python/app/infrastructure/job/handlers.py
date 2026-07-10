"""
XXL-Job 定时任务 Handler

定义所有注册到 XXL-Job 调度中心的定时任务。
任务在 XXL-Job Admin 控制台中配置 CRON 表达式和参数。

任务清单：
- cleanupExpiredTasks: 清理过期任务（每天凌晨 2 点）
- cleanupStuckTasks:   回收僵死任务（每小时）
- modelHealthCheck:    模型健康检查（每 30 分钟）
- cleanupOrphanFiles:  孤儿文件清理（每天凌晨 4 点）
- cleanupTempFiles:    临时文件清理（每 6 小时）
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from pyxxl import JobHandler
from sqlalchemy import and_, delete, update

from app.database import get_db_session
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus
from app.repository.task_repository import task_repository

logger = logging.getLogger(__name__)

# 全局 handler 注册器（由 executor.py 导入并绑定到 PyxxlRunner）
xxl_handler = JobHandler()

# 任务缓存 Redis Key 前缀（与 TaskServiceAsync 保持一致）
_TASK_CACHE_PREFIX = "task:cache:"
_TASK_PROGRESS_PREFIX = "task:progress:"
_TASK_CANCEL_PREFIX = "task:cancel:"


@xxl_handler.register(name="cleanupExpiredTasks")
async def cleanup_expired_tasks() -> str:
    """
    清理过期任务

    - 删除 7 天前已完成/已取消的任务
    - 删除 30 天前所有已终止的任务（排除 pending/processing）
    - 精准清理对应的 Redis 缓存（仅删除已被数据库删除的任务 Key）

    CRON 建议: 0 0 2 * * ? （每天凌晨 2 点）
    """
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    async with get_db_session() as db:
        # 先收集要删除的 task_id（用于后续 Redis 精准清理）
        terminated_task_ids = await task_repository.get_terminated_task_ids(db, seven_days_ago)

        # 删除 7 天前已完成/已取消的任务
        stmt_completed = delete(SysTask).where(
            and_(
                SysTask.status.in_([TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]),
                SysTask.created_at < seven_days_ago,
            )
        )
        result_completed = await db.execute(stmt_completed)

        # 删除 30 天前已终止的任务（排除 pending/processing，防止误删正在执行的任务）
        stmt_old = delete(SysTask).where(
            and_(
                SysTask.status.not_in([TaskStatus.PENDING.value, TaskStatus.PROCESSING.value]),
                SysTask.created_at < thirty_days_ago,
            )
        )
        result_old = await db.execute(stmt_old)

    deleted_completed = result_completed.rowcount
    deleted_old = result_old.rowcount
    total = deleted_completed + deleted_old

    # 精准清理 Redis 缓存（仅删除已在 DB 中被删除的任务 Key）
    redis_deleted = await _cleanup_task_redis_keys(terminated_task_ids)

    msg = (
        f"过期任务清理完成: "
        f"7天前已完成/取消={deleted_completed}, "
        f"30天前已终止={deleted_old}, "
        f"总计删除={total}, "
        f"Redis缓存清理={redis_deleted}"
    )
    logger.info(msg)
    return msg


@xxl_handler.register(name="cleanupStuckTasks")
async def cleanup_stuck_tasks() -> str:
    """
    回收僵死任务

    将超过 30 分钟 updated_at 未更新且处于 processing 状态的任务标记为 failed。
    将超过 24 小时仍处于 pending 状态的任务标记为 failed。
    这些任务可能由于进程崩溃、网络中断等原因未正常完成。

    CRON 建议: 0 0 * * * ? （每小时）
    """
    now = datetime.now()
    processing_threshold = now - timedelta(minutes=30)
    pending_threshold = now - timedelta(hours=24)

    async with get_db_session() as db:
        # 回收 30 分钟未更新的 processing 任务（基于 updated_at）
        stmt_processing = (
            update(SysTask)
            .where(
                and_(
                    SysTask.status == TaskStatus.PROCESSING.value,
                    SysTask.updated_at < processing_threshold,
                )
            )
            .values(
                status=TaskStatus.FAILED.value,
                error_message="任务超时（30分钟无进度更新），已被系统自动回收",
                completed_at=now,
                updated_at=now,
            )
        )
        result_processing = await db.execute(stmt_processing)

        # 回收 24 小时未启动的 pending 任务
        stmt_pending = (
            update(SysTask)
            .where(
                and_(
                    SysTask.status == TaskStatus.PENDING.value,
                    SysTask.created_at < pending_threshold,
                )
            )
            .values(
                status=TaskStatus.FAILED.value,
                error_message="任务超时（24h未启动），已被系统自动回收",
                completed_at=now,
                updated_at=now,
            )
        )
        result_pending = await db.execute(stmt_pending)

    recovered_processing = result_processing.rowcount
    recovered_pending = result_pending.rowcount
    total = recovered_processing + recovered_pending

    msg = (
        f"僵死任务回收完成: "
        f"processing超时={recovered_processing}(>30min), "
        f"pending超时={recovered_pending}(>24h), "
        f"总计回收={total}"
    )
    logger.info(msg)
    return msg


@xxl_handler.register(name="modelHealthCheck")
async def model_health_check() -> str:
    """
    模型健康检查

    检查 GPU 设备可用性和已加载模型的状态，
    异常时记录告警日志（后续可对接告警通知）。

    CRON 建议: 0 */30 * * * ? （每 30 分钟）
    """
    issues: list[str] = []

    # 检查 GPU 可用性
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA 不可用")
        else:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                used_pct = (total_mem - free_mem) / total_mem * 100
                if used_pct > 95:
                    issues.append(
                        f"GPU:{i} 显存使用率过高: {used_pct:.1f}%"
                    )
    except Exception as e:
        issues.append(f"GPU 检查异常: {e}")

    # 检查数据库连接
    try:
        from sqlalchemy import text
        async with get_db_session() as db:
            await db.execute(text("SELECT 1"))
    except Exception as e:
        issues.append(f"数据库连接异常: {e}")

    # 检查 Redis 连接
    try:
        from app.dependencies.redis import check_redis_health
        await check_redis_health()
    except Exception as e:
        issues.append(f"Redis 检查异常: {e}")

    if issues:
        msg = f"健康检查发现 {len(issues)} 个问题: {'; '.join(issues)}"
        logger.warning(msg)
    else:
        msg = "健康检查通过: GPU/DB/Redis 均正常"
        logger.info(msg)

    return msg


async def _cleanup_task_redis_keys(task_ids: list[str]) -> int:
    """
    精准清理 Redis 中指定任务的缓存 Key

    仅删除已经从数据库中被删除的任务对应的 Key，不影响正在执行的任务。

    Args:
        task_ids: 需要清理的任务 UUID 列表

    Returns:
        删除的 Key 数量
    """
    if not task_ids:
        return 0

    try:
        from app.dependencies.redis import get_redis_client

        redis = await get_redis_client()
        if redis is None:
            return 0

        # 精准删除：根据 task_id 构造 Key
        keys_to_delete = []
        for tid in task_ids:
            keys_to_delete.append(f"{_TASK_CACHE_PREFIX}{tid}")
            keys_to_delete.append(f"{_TASK_PROGRESS_PREFIX}{tid}")
            keys_to_delete.append(f"{_TASK_CANCEL_PREFIX}{tid}")

        deleted = 0
        if keys_to_delete:
            deleted = await redis.delete(*keys_to_delete)

        if deleted > 0:
            logger.info(f"已精准清理 {deleted} 个 Redis 任务缓存 Key（涉及 {len(task_ids)} 个任务）")

        return deleted

    except Exception as e:
        logger.warning(f"Redis 任务缓存清理失败（不影响主流程）: {e}")
        return 0


# ==================== 文件清理定时任务 ====================


@xxl_handler.register(name="cleanupOrphanFiles")
async def cleanup_orphan_files() -> str:
    """
    孤儿文件清理

    清理存储中存在但数据库无对应记录的文件（通常由上传事务回滚导致）。
    仅清理超过阈值时间（默认 48h）的孤儿文件，避免误删正在上传中的文件。

    CRON 建议: 0 0 4 * * ? （每天凌晨 4 点）
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from app.config import settings
    from app.repository.file_repository import file_repository
    from app.service.storage.factory import get_storage_service

    threshold_hours = settings.FILE_ORPHAN_CLEANUP_HOURS
    bucket_name = settings.MINIO_BUCKET_NAME

    async with get_db_session() as db:
        # 获取数据库中所有文件的 object_name
        db_object_names = set(await file_repository.get_all_object_names(db))

    # 从存储中获取所有对象
    storage = get_storage_service()
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="file-cleanup")
    loop = asyncio.get_running_loop()

    def _list_storage_objects():
        return storage.list_objects(bucket_name, prefix="upload/")

    try:
        storage_objects = await loop.run_in_executor(executor, _list_storage_objects)
    except Exception as e:
        msg = f"孤儿文件清理失败: 无法列出存储对象: {e}"
        logger.error(msg)
        return msg

    # 找出孤儿文件（存储中有但 DB 没有的）
    orphan_objects = [obj for obj in storage_objects if obj not in db_object_names]

    if not orphan_objects:
        msg = "孤儿文件清理完成: 未发现孤儿文件"
        logger.info(msg)
        return msg

    # 删除孤儿文件
    deleted = 0
    failed = 0
    for obj_name in orphan_objects:
        def _delete(name=obj_name):
            try:
                storage.delete(bucket_name, name)
                return True
            except Exception as e:
                logger.warning(f"孤儿文件删除失败 [{name}]: {e}")
                return False

        try:
            ok = await loop.run_in_executor(executor, _delete)
            if ok:
                deleted += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    executor.shutdown(wait=False)

    msg = (
        f"孤儿文件清理完成: "
        f"发现={len(orphan_objects)}, "
        f"已删除={deleted}, "
        f"失败={failed}"
    )
    logger.info(msg)
    return msg


@xxl_handler.register(name="cleanupTempFiles")
async def cleanup_temp_files() -> str:
    """
    临时文件清理

    清理超过阈值时间（默认 24h）的临时目录中的文件。
    临时文件由文件处理任务产生，正常情况下任务完成后会清理，
    但异常中断时可能残留。

    CRON 建议: 0 0 */6 * * ? （每 6 小时）
    """
    import os
    import time
    from app.config import settings

    temp_dir = settings.TEMP_DIR_RESOLVED
    threshold_hours = settings.FILE_TEMP_CLEANUP_HOURS
    threshold_seconds = threshold_hours * 3600
    now = time.time()

    if not os.path.exists(temp_dir):
        msg = f"临时文件清理: 目录不存在 {temp_dir}"
        logger.info(msg)
        return msg

    deleted = 0
    failed = 0

    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                mtime = os.path.getmtime(filepath)
                if now - mtime > threshold_seconds:
                    os.unlink(filepath)
                    deleted += 1
            except Exception as e:
                logger.warning(f"临时文件删除失败 [{filepath}]: {e}")
                failed += 1

        # 尝试删除空目录（不删除 temp_dir 本身）
        if root != temp_dir:
            try:
                if not os.listdir(root):
                    os.rmdir(root)
            except Exception:
                pass

    msg = (
        f"临时文件清理完成: "
        f"目录={temp_dir}, "
        f"阈值={threshold_hours}h, "
        f"已删除={deleted}, "
        f"失败={failed}"
    )
    logger.info(msg)
    return msg
