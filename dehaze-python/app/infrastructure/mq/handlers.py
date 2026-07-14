"""
RabbitMQ 消费者 Handler

定义各队列的消息处理函数，由 Consumer 消费时回调。
包含幂等检查和消息校验。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.core.constants import SYSTEM_USER_ID
from app.database import get_db_session
from app.models.base import set_current_user_id
from app.models.enum.task_enum import TaskStatus
from app.repository.task_repository import task_repository

logger = logging.getLogger(__name__)


async def handle_export_task(body: dict[str, Any], headers: dict[str, Any]) -> None:
    """
    导出任务消费者 handler

    从 RabbitMQ 接收导出任务消息，执行导出逻辑。
    包含幂等检查：仅处理非终态的任务。

    消息格式:
        {
            "db_task_id": int,
            "task_id": str,
            "task_type": str,
            "target_id": int | None,
            "target_ids": list[int],
            "options": dict,
            "user_id": int | None,
        }
    """
    user_id = body.get("user_id")
    set_current_user_id(user_id)
    try:
        # 消息字段校验
        db_task_id = body.get("db_task_id")
        task_id = body.get("task_id")
        task_type = body.get("task_type")
        target_id = body.get("target_id")
        target_ids = body.get("target_ids")
        options = body.get("options")

        if not task_id or not task_type or db_task_id is None:
            logger.error(f"导出任务消息格式无效（缺少必要字段）: {body}")
            return

        logger.info(f"[MQ] 开始处理导出任务: taskId={task_id}, type={task_type}")

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
                logger.info(
                    f"[MQ] 任务已为终态，跳过重复消费: taskId={task_id}, status={sys_task.status}"
                )
                return

            # 更新 retry_count（从消息 header 读取）
            retry_count_str = headers.get("x-retry-count")
            if retry_count_str is not None:
                try:
                    retry_count = int(retry_count_str)
                    if retry_count > 0:
                        await task_repository.update_retry_count(db, task_id, retry_count)
                        await db.commit()
                except (ValueError, TypeError):
                    logger.warning(f"[MQ] retry_count 解析失败: {retry_count_str}")

        # 延迟导入避免循环依赖
        from app.service.task_service import TaskServiceAsync

        await TaskServiceAsync._execute_export_task_background(
            db_task_id=db_task_id,
            task_id=task_id,
            task_type=task_type,
            target_id=target_id,
            target_ids=target_ids,
            options=options,
        )

        logger.info(f"[MQ] 导出任务处理完成: taskId={task_id}")
    finally:
        set_current_user_id(None)


async def handle_dlq_message(body: dict[str, Any], headers: dict[str, Any]) -> None:
    """
    死信队列消费者 handler

    处理重试耗尽或过期的消息，将对应任务标记为 FAILED 并记录错误信息。
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
                logger.info(
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
                from app.dependencies.redis import get_redis_client
                from app.service.task_service import TaskServiceAsync
                redis = await get_redis_client()
                await TaskServiceAsync._update_cache(redis, sys_task)
                await TaskServiceAsync._push_task_ws_message(sys_task, {
                    "type": "task_status",
                    "task_id": sys_task.task_id,
                    "status": TaskStatus.FAILED.value,
                    "result": None,
                    "error_message": sys_task.error_message,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.warning(f"[DLQ] 更新缓存/推送通知失败: {e}")

        logger.warning(f"[DLQ] 死信消息处理完成，任务已标记失败: taskId={task_id}")
    finally:
        set_current_user_id(None)
