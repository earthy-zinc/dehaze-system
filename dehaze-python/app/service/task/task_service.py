"""任务生命周期域：创建/查询/取消/重试与调度分发（幂等、Redis 缓存）。"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.constants import (
    CANCEL_FLAG_TTL_SECONDS,
    IDEMPOTENCY_KEY_PREFIX,
    TASK_CACHE_PREFIX,
    TASK_CANCEL_PREFIX,
    TASK_EXPIRE_HOURS,
)
from app.core.exceptions import BusinessException
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus, TaskType
from app.repository.task_repository import task_repository
from app.service.task.task_state import refresh_task_cache, task_to_dict

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_TTL = 24 * 3600


async def create_task(
    db: AsyncSession,
    redis: Redis,
    task_type: str,
    params_json: str | None,
    user_id: int,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """
    创建任务（统一入口，对齐 Java createTask）
    """
    if user_id is None:
        raise BusinessException("用户未登录")

    valid_types = [t.value for t in TaskType]
    if task_type not in valid_types:
        raise BusinessException(ResultCode.TASK_TYPE_UNSUPPORTED)

    if idempotency_key:
        existing_task = await task_repository.get_by_idempotency_key(db, idempotency_key)
        if existing_task is not None:
            logger.debug(
                "幂等键命中，返回已有任务: taskId=%s, idempotencyKey=%s",
                existing_task.task_id,
                idempotency_key,
            )
            return task_to_dict(existing_task)

    task_id = str(uuid.uuid4())
    now = datetime.now()
    sys_task = SysTask(
        task_id=task_id,
        task_type=task_type,
        status=TaskStatus.PENDING.value,
        progress=0,
        total_files=0,
        processed_files=0,
        params=params_json or "{}",
        create_by=user_id,
        create_time=now,
        expires_at=now + timedelta(hours=TASK_EXPIRE_HOURS),
        idempotency_key=idempotency_key,
    )

    db.add(sys_task)
    try:
        await db.flush()
        await db.refresh(sys_task)
    except IntegrityError:
        await db.rollback()
        if idempotency_key:
            existing_task = await task_repository.get_by_idempotency_key(db, idempotency_key)
            if existing_task is not None:
                logger.debug("并发幂等键命中，返回已有任务: taskId=%s", existing_task.task_id)
                return task_to_dict(existing_task)
        raise

    if idempotency_key:
        idempotency_redis_key = IDEMPOTENCY_KEY_PREFIX + idempotency_key
        await redis.setex(idempotency_redis_key, IDEMPOTENCY_KEY_TTL, task_id)

    cache_key = TASK_CACHE_PREFIX + task_id
    task_dict = task_to_dict(sys_task)
    await redis.setex(cache_key, TASK_EXPIRE_HOURS * 3600, json.dumps(task_dict, default=str))

    await _dispatch_task(
        db_task_id=sys_task.id,
        task_id=task_id,
        task_type=task_type,
        params_json=params_json,
        user_id=user_id,
    )

    logger.debug("创建任务成功: taskId=%s, type=%s, userId=%s", task_id, task_type, user_id)
    return task_dict


async def get_task_status(
    db: AsyncSession,
    redis: Redis,
    task_id: str,
    user_id: int,
) -> dict[str, Any] | None:
    if not task_id:
        raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

    cache_key = TASK_CACHE_PREFIX + task_id
    cached_task = await redis.get(cache_key)
    if cached_task:
        try:
            task_data = json.loads(cached_task)
            if task_data.get("create_by") != user_id:
                raise BusinessException(ResultCode.TASK_UNAUTHORIZED)
            return task_data
        except json.JSONDecodeError as e:
            logger.warning("解析缓存数据失败: %s", e)

    sys_task = await task_repository.get_by_task_id(db, task_id)
    if sys_task is None:
        return None

    if sys_task.create_by != user_id:
        raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

    task_dict = task_to_dict(sys_task)
    await redis.setex(cache_key, TASK_EXPIRE_HOURS * 3600, json.dumps(task_dict, default=str))

    return task_dict


async def list_tasks(
    db: AsyncSession,
    user_id: int,
    status: int | None = None,
    task_type: str | None = None,
    task_category: str | None = None,
    page: int = 1,
    size: int = 10,
) -> dict[str, Any]:
    items, total = await task_repository.get_user_tasks_paginated(
        db,
        user_id,
        status=status,
        task_type=task_type,
        task_category=task_category,
        page=page,
        size=size,
    )
    return {
        "list": [task_to_dict(t) for t in items],
        "total": total,
    }


async def get_export_object_name(
    db: AsyncSession,
    redis: Redis,
    task_id: str,
    user_id: int,
) -> str | None:
    """读取导出文件的对象键（object_name），用于响应层运行时拼接下载 URL。"""
    if not task_id:
        raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

    sys_task = await task_repository.get_by_task_id(db, task_id)
    if sys_task is None:
        raise BusinessException(ResultCode.TASK_NOT_FOUND)

    if sys_task.create_by != user_id:
        raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

    if sys_task.status != TaskStatus.COMPLETED.value:
        raise BusinessException(ResultCode.TASK_STATUS_INVALID, "任务未完成，无法下载")

    if sys_task.expires_at and sys_task.expires_at < datetime.now():
        raise BusinessException(ResultCode.TASK_STATUS_INVALID, "任务已过期，无法下载")

    if not sys_task.result:
        logger.warning("任务结果为空: taskId=%s", task_id)
        return None

    if not isinstance(sys_task.result, str):
        logger.warning("任务结果非对象键: taskId=%s", task_id)
        return None

    return sys_task.result


async def cancel_task(
    db: AsyncSession,
    redis: Redis,
    task_id: str,
    user_id: int,
) -> None:
    if not task_id:
        raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

    sys_task = await task_repository.get_by_task_id(db, task_id)
    if sys_task is None:
        raise BusinessException(ResultCode.TASK_NOT_FOUND)

    if sys_task.create_by != user_id:
        raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

    if sys_task.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
        raise BusinessException(ResultCode.TASK_STATUS_INVALID, "任务已完成或失败，无法取消")

    if sys_task.status == TaskStatus.CANCELLED.value:
        return

    sys_task.status = TaskStatus.CANCELLED.value
    sys_task.completed_at = datetime.now()

    cache_key = TASK_CACHE_PREFIX + task_id
    task_dict = task_to_dict(sys_task)
    await redis.setex(cache_key, TASK_EXPIRE_HOURS * 3600, json.dumps(task_dict, default=str))

    cancel_key = TASK_CANCEL_PREFIX + task_id
    await redis.setex(cancel_key, CANCEL_FLAG_TTL_SECONDS, "true")

    logger.debug("取消任务成功: taskId=%s", task_id)


async def retry_task(
    db: AsyncSession,
    redis: Redis,
    task_id: str,
    user_id: int,
) -> dict[str, Any]:
    if not task_id:
        raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

    sys_task = await task_repository.get_by_task_id(db, task_id)
    if sys_task is None:
        raise BusinessException(ResultCode.TASK_NOT_FOUND)

    if sys_task.create_by != user_id:
        raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

    if sys_task.status != TaskStatus.FAILED.value:
        raise BusinessException(ResultCode.TASK_STATUS_INVALID, "仅失败任务可重试")

    sys_task.status = TaskStatus.PENDING.value
    sys_task.progress = 0
    sys_task.processed_files = 0
    sys_task.error_message = None
    sys_task.started_at = None
    sys_task.completed_at = None
    sys_task.retry_count = (sys_task.retry_count or 0) + 1
    sys_task.worker_id = None
    sys_task.expires_at = datetime.now() + timedelta(hours=TASK_EXPIRE_HOURS)

    await refresh_task_cache(redis, sys_task)

    await _dispatch_task(
        db_task_id=sys_task.id,
        task_id=sys_task.task_id,
        task_type=sys_task.task_type,
        params_json=sys_task.params,
        user_id=user_id,
    )

    logger.debug(
        "重试任务: taskId=%s, userId=%s, retryCount=%s", task_id, user_id, sys_task.retry_count
    )
    return task_to_dict(sys_task)


async def _dispatch_task(
    db_task_id: int,
    task_id: str,
    task_type: str,
    params_json: str | None,
    user_id: int,
) -> None:
    from app.infrastructure.mq.connection import get_publisher
    from app.service.task.task_executor import execute_task_background

    publisher = get_publisher()

    if publisher is not None and publisher.is_connected:
        try:
            await publisher.publish(
                routing_key="task.export",
                body={
                    "db_task_id": db_task_id,
                    "task_id": task_id,
                    "task_type": task_type,
                },
            )
            logger.debug("任务已发布到 RabbitMQ: taskId=%s", task_id)
            return
        except Exception as e:
            logger.warning("RabbitMQ 发布失败，降级为本地执行: %s", e)

    background_task = asyncio.create_task(
        execute_task_background(db_task_id, task_id, task_type, params_json)
    )

    try:
        from app.service.task_tracker import get_task_tracker

        tracker = get_task_tracker()
        await tracker.register(
            task_id=task_id,
            task=background_task,
            task_type=task_type,
            metadata={"db_task_id": db_task_id, "user_id": user_id},
        )
    except Exception as e:
        logger.warning("任务追踪注册失败（不影响执行）: %s", e)
