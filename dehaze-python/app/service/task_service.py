"""
任务服务模块

实现导出任务的异步执行和管理，使用 Redis 存储任务状态。
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.constants import (
    CANCEL_FLAG_TTL_SECONDS,
    IDEMPOTENCY_KEY_PREFIX,
    PROGRESS_MIN_INTERVAL_SECONDS,
    PROGRESS_MIN_PERCENT_DELTA,
    RESULT_FILE_EXPIRE_DAYS,
    SYSTEM_USER_ID,
    TASK_CACHE_PREFIX,
    TASK_CANCEL_PREFIX,
    TASK_EXPIRE_HOURS,
    TASK_PROGRESS_PREFIX,
)
from app.core.exceptions import BusinessException, TaskCancelledException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.metrics.task_metrics import TaskMetricsContext
from app.models.base import set_current_user_id
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus, TaskType
from app.repository.task_repository import task_repository

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
            return _task_to_dict(existing_task)

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
                return _task_to_dict(existing_task)
        raise

    if idempotency_key:
        idempotency_redis_key = IDEMPOTENCY_KEY_PREFIX + idempotency_key
        await redis.setex(idempotency_redis_key, IDEMPOTENCY_KEY_TTL, task_id)

    cache_key = TASK_CACHE_PREFIX + task_id
    task_dict = _task_to_dict(sys_task)
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

    task_dict = _task_to_dict(sys_task)
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
        "list": [_task_to_dict(t) for t in items],
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
    task_dict = _task_to_dict(sys_task)
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
    return _task_to_dict(sys_task)


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


async def _dispatch_task(
    db_task_id: int,
    task_id: str,
    task_type: str,
    params_json: str | None,
    user_id: int,
) -> None:
    from app.infrastructure.mq.connection import get_publisher

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


def _task_to_dict(task: SysTask) -> dict[str, Any]:
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


async def _update_task_progress(
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
    task_dict = _task_to_dict(sys_task)
    await redis.setex(cache_key, TASK_EXPIRE_HOURS * 3600, json.dumps(task_dict, default=str))


async def _is_task_cancelled(redis: Redis, task_id: str) -> bool:
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


async def execute_task_background(
    db_task_id: int,
    task_id: str,
    task_type: str,
    params_json: str | None,
) -> None:
    set_current_user_id(SYSTEM_USER_ID)
    try:
        redis = await get_redis_client()
        metrics_enabled = settings.PROMETHEUS_ENABLED

        try:
            async with get_db_session() as db:
                logger.debug("开始执行任务: taskId=%s, type=%s", task_id, task_type)

                sys_task = await task_repository.get_by_id(db, db_task_id)
                if sys_task is None:
                    logger.error("任务不存在: taskId=%s", task_id)
                    return

                set_current_user_id(sys_task.create_by)

                try:
                    sys_task.status = TaskStatus.PROCESSING.value
                    sys_task.started_at = datetime.now()
                    await refresh_task_cache(redis, sys_task)

                    metrics_cm = (
                        TaskMetricsContext(task_type)
                        if metrics_enabled
                        else contextlib.AsyncExitStack()
                    )
                    async with metrics_cm as metrics_ctx:
                        from app.service.task.factory import TaskStrategyFactory

                        strategy = TaskStrategyFactory.get_strategy(task_type)

                        progress_state = {"time": 0.0, "progress": 0}

                        async def progress_callback(processed: int, total: int) -> None:
                            await _update_task_progress(
                                db,
                                redis,
                                db_task_id,
                                processed,
                                total,
                                _last_update=progress_state,
                            )

                        async def cancel_checker() -> bool:
                            return await _is_task_cancelled(redis, task_id)

                        task_result = await strategy.execute(
                            db=db,
                            sys_task=sys_task,
                            params_json=params_json,
                            progress_callback=progress_callback,
                            cancel_checker=cancel_checker,
                        )

                        if task_result:
                            await update_task_completed(
                                db,
                                redis,
                                task_id,
                                task_result,
                                datetime.now() + timedelta(days=RESULT_FILE_EXPIRE_DAYS),
                            )
                            logger.info("任务完成: taskId=%s, result=%s", task_id, task_result)
                        else:
                            if metrics_enabled and isinstance(metrics_ctx, TaskMetricsContext):
                                metrics_ctx.set_status("failed")
                            await update_task_failed(
                                db, redis, task_id, "任务执行未返回结果"
                            )

                except asyncio.CancelledError:
                    logger.warning("任务被取消（服务关闭）: taskId=%s", task_id)
                    await update_task_failed(
                        db, redis, task_id, "服务关闭，任务中断"
                    )
                    raise

                except TaskCancelledException:
                    logger.warning("任务被取消: taskId=%s", task_id)
                    sys_task.status = TaskStatus.CANCELLED.value
                    sys_task.completed_at = datetime.now()
                    await refresh_task_cache(redis, sys_task)

                except Exception as e:
                    logger.error("任务执行失败: taskId=%s", task_id, exc_info=True)
                    await update_task_failed(db, redis, task_id, str(e))

        except Exception as e:
            logger.error("后台任务执行异常: %s", e, exc_info=True)
    finally:
        set_current_user_id(None)


async def consume_export_message(
    body: dict[str, Any],
    headers: dict[str, Any],
) -> None:
    """
    消费导出任务消息，执行导出逻辑。

    从 RabbitMQ 接收导出任务消息，执行导出逻辑。
    包含幂等检查：仅处理非终态的任务。

    消息格式（最小自描述）:
        {
            "db_task_id": int,
            "task_id": str,
            "task_type": str,
        }

    user_id 和 params_json 从 DB sys_task 记录中读取，不从消息体获取。
    """
    try:
        db_task_id = body.get("db_task_id")
        task_id = body.get("task_id")
        task_type = body.get("task_type")

        if not task_id or not task_type or db_task_id is None:
            logger.error(f"导出任务消息格式无效（缺少必要字段）: {body}")
            return

        logger.debug(f"[MQ] 开始处理导出任务: taskId={task_id}, type={task_type}")

        # 幂等检查：查 DB 确认任务非终态才执行
        async with get_db_session() as db:
            sys_task = await task_repository.get_by_id(db, db_task_id)
            if sys_task is None:
                logger.warning(f"[MQ] 任务不存在，跳过: db_task_id={db_task_id}, taskId={task_id}")
                return

            terminal_states = {
                TaskStatus.COMPLETED.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value,
            }
            if sys_task.status in terminal_states:
                logger.debug(
                    f"[MQ] 任务已为终态，跳过重复消费: taskId={task_id}, status={sys_task.status}"
                )
                return

            # 从 DB 读取 user_id 和 params_json（不信任消息体）
            user_id = sys_task.create_by
            params_json = sys_task.params or "{}"

            set_current_user_id(user_id)

            retry_count_str = headers.get("x-retry-count")
            if retry_count_str is not None:
                try:
                    retry_count = int(retry_count_str)
                    if retry_count > 0:
                        await task_repository.update_retry_count(db, task_id, retry_count)
                        await db.commit()
                except (ValueError, TypeError):
                    logger.warning(f"[MQ] retry_count 解析失败: {retry_count_str}")

        await execute_task_background(db_task_id, task_id, task_type, params_json)

        logger.debug(f"[MQ] 导出任务处理完成: taskId={task_id}")
    finally:
        set_current_user_id(None)


async def consume_dlq_message(
    body: dict[str, Any],
    headers: dict[str, Any],
) -> None:
    """
    消费死信队列消息，将对应任务标记为 FAILED 并记录错误信息。

    消息格式（最小自描述）:
        {
            "db_task_id": int,
            "task_id": str,
            "task_type": str,
        }
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        task_id = body.get("task_id")
        db_task_id = body.get("db_task_id")
        retry_count_str = headers.get("x-retry-count")
        retry_count: int | str = "未知"
        if retry_count_str is not None:
            try:
                retry_count = int(retry_count_str)
            except (ValueError, TypeError):
                retry_count = "未知"

        logger.error(
            f"[DLQ] 收到死信消息: taskId={task_id}, db_task_id={db_task_id}, "
            f"retry_count={retry_count}, body={body}"
        )

        if db_task_id is None:
            logger.error(f"[DLQ] 死信消息缺少 db_task_id，无法更新任务状态: {body}")
            return

        # 将任务标记为 FAILED
        async with get_db_session() as db:
            sys_task = await task_repository.get_by_id(db, db_task_id)
            if sys_task is None:
                logger.warning(f"[DLQ] 任务不存在，无法标记失败: db_task_id={db_task_id}")
                return

            # 仅在非终态时更新（避免覆盖已完成的任务）
            terminal_states = {
                TaskStatus.COMPLETED.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value,
            }
            if sys_task.status in terminal_states:
                logger.debug(
                    f"[DLQ] 任务已为终态，跳过: taskId={task_id}, status={sys_task.status}"
                )
                return

            sys_task.status = TaskStatus.FAILED.value
            sys_task.error_message = f"消息重试耗尽进入死信队列（重试次数: {retry_count}）"
            sys_task.completed_at = datetime.now()
            if isinstance(retry_count, int):
                sys_task.retry_count = retry_count
            await db.commit()

            # 更新缓存并推送 WebSocket 通知
            try:
                redis = await get_redis_client()
                await refresh_task_cache(redis, sys_task)
                await push_task_ws_message(
                    sys_task,
                    {
                        "type": "task_status",
                        "task_id": sys_task.task_id,
                        "status": TaskStatus.FAILED.value,
                        "result": None,
                        "error_message": sys_task.error_message,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"[DLQ] 更新缓存/推送通知失败: {e}")

        logger.warning(f"[DLQ] 死信消息处理完成，任务已标记失败: taskId={task_id}")
    finally:
        set_current_user_id(None)
