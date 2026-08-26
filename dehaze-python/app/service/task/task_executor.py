"""任务执行域：后台执行任务（策略分发 + 进度/终态回写 + 指标）。"""

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta

from app.config import settings
from app.core.constants import RESULT_FILE_EXPIRE_DAYS, SYSTEM_USER_ID
from app.core.exceptions import TaskCancelledException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.metrics.task_metrics import TaskMetricsContext
from app.models.base import set_current_user_id
from app.models.enum.task_enum import TaskStatus
from app.repository.task_repository import task_repository
from app.service.task.task_state import (
    is_task_cancelled,
    refresh_task_cache,
    update_task_completed,
    update_task_failed,
    update_task_progress,
)

logger = logging.getLogger(__name__)


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
                            await update_task_progress(
                                db,
                                redis,
                                db_task_id,
                                processed,
                                total,
                                _last_update=progress_state,
                            )

                        async def cancel_checker() -> bool:
                            return await is_task_cancelled(redis, task_id)

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
