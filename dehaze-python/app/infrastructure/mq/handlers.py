"""
RabbitMQ 消费者 Handler

定义各队列的消息处理函数，由 Consumer 消费时回调。
包含幂等检查和消息校验。
"""

from __future__ import annotations

import logging
from typing import Any

from app.database import get_db_session
from app.models.enum.task_enum import TaskStatus
from app.repository.task_repository import task_repository

logger = logging.getLogger(__name__)


async def handle_export_task(body: dict[str, Any]) -> None:
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
        }
    """
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
