"""任务状态维护：进度更新、终态落库、缓存刷新、取消标记、WebSocket 推送。"""

import json
import logging
import time
from datetime import datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import (
    PROGRESS_MIN_INTERVAL_SECONDS,
    PROGRESS_MIN_PERCENT_DELTA,
    TASK_CACHE_PREFIX,
    TASK_CANCEL_PREFIX,
    TASK_EXPIRE_HOURS,
    TASK_PROGRESS_PREFIX,
)
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus
from app.repository.task_repository import task_repository

logger = logging.getLogger(__name__)


def task_to_dict(task: SysTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status,
        "progress": task.progress,
        "total_files": task.total_files,
        "processed_files": task.processed_files,
        "result": task.result,
        "error_message": task.error_message,
        "create_by": task.create_by,
        "created_at": task.create_time.isoformat() if task.create_time else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "expires_at": task.expires_at.isoformat() if task.expires_at else None,
        "idempotency_key": task.idempotency_key,
        "retry_count": task.retry_count,
        "worker_id": task.worker_id,
    }


async def update_task_completed(
    db: AsyncSession,
    redis: Redis,
    task_id: str,
    result: Any,
    expires_at: datetime | None = None,
) -> None:
    sys_task = await task_repository.get_by_task_id(db, task_id)
    if sys_task is None:
        return
    sys_task.status = TaskStatus.COMPLETED.value
    sys_task.result = result
    sys_task.completed_at = datetime.now()
    if expires_at:
        sys_task.expires_at = expires_at
    await refresh_task_cache(redis, sys_task)
    await push_task_ws_message(
        sys_task,
        {
            "type": "task_status",
            "task_id": sys_task.task_id,
            "status": TaskStatus.COMPLETED.value,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def update_task_failed(
    db: AsyncSession,
    redis: Redis,
    task_id: str,
    error_message: str,
) -> None:
    sys_task = await task_repository.get_by_task_id(db, task_id)
    if sys_task is None:
        return
    sys_task.status = TaskStatus.FAILED.value
    sys_task.error_message = error_message
    sys_task.completed_at = datetime.now()
    await refresh_task_cache(redis, sys_task)
    await push_task_ws_message(
        sys_task,
        {
            "type": "task_status",
            "task_id": sys_task.task_id,
            "status": TaskStatus.FAILED.value,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
        },
    )


async def update_task_progress(
    db: AsyncSession,
    redis: Redis,
    db_task_id: int,
    processed_files: int,
    total_files: int,
    *,
    _last_update: dict[str, Any] | None = None,
) -> None:
    progress = int(processed_files * 100 / total_files) if total_files > 0 else 100

    if _last_update is not None:
        elapsed = time.monotonic() - _last_update.get("time", 0)
        last_progress = _last_update.get("progress", 0)
        if (
            elapsed < PROGRESS_MIN_INTERVAL_SECONDS
            and abs(progress - last_progress) < PROGRESS_MIN_PERCENT_DELTA
            and progress < 100
        ):
            return

    sys_task = await task_repository.get_by_id(db, db_task_id)
    if sys_task:
        await task_repository.update_progress(
            db, sys_task.task_id, progress, processed_files, total_files
        )
        sys_task.progress = progress
        sys_task.processed_files = processed_files

        progress_key = TASK_PROGRESS_PREFIX + sys_task.task_id
        await redis.setex(
            progress_key,
            TASK_EXPIRE_HOURS * 3600,
            json.dumps(
                {"progress": progress, "processed": processed_files, "total": total_files}
            ),
        )

        await refresh_task_cache(redis, sys_task)

        await push_task_ws_message(
            sys_task,
            {
                "type": "task_progress",
                "task_id": sys_task.task_id,
                "progress": progress,
                "status": sys_task.status,
                "processed_files": processed_files,
                "total_files": total_files,
                "timestamp": datetime.now().isoformat(),
            },
        )

        if _last_update is not None:
            _last_update["time"] = time.monotonic()
            _last_update["progress"] = progress


async def refresh_task_cache(redis: Redis, sys_task: SysTask) -> None:
    cache_key = TASK_CACHE_PREFIX + sys_task.task_id
    task_dict = task_to_dict(sys_task)
    await redis.setex(cache_key, TASK_EXPIRE_HOURS * 3600, json.dumps(task_dict, default=str))


async def is_task_cancelled(redis: Redis, task_id: str) -> bool:
    cancel_key = TASK_CANCEL_PREFIX + task_id
    is_cancelled = await redis.get(cancel_key)
    if isinstance(is_cancelled, bytes):
        is_cancelled = is_cancelled.decode("utf-8")
    return is_cancelled == "true"


async def push_task_ws_message(sys_task: SysTask, message: dict[str, Any]) -> None:
    try:
        from app.service.websocket_service import manager as ws_manager

        if sys_task.create_by is not None:
            await ws_manager.send_personal(sys_task.create_by, message)
    except Exception as e:
        logger.debug("WebSocket 推送失败（不影响任务执行）: %s", e)
