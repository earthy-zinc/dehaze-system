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

LOW_RATING_URGENT_COUNT = 3
LOW_RATING_SEVERE_RATE = 0.20


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
            "params_json": str,
            "user_id": int | None,
        }
    """
    user_id = body.get("user_id")
    set_current_user_id(user_id)
    try:
        db_task_id = body.get("db_task_id")
        task_id = body.get("task_id")
        task_type = body.get("task_type")
        params_json = body.get("params_json", "{}")

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

            retry_count_str = headers.get("x-retry-count")
            if retry_count_str is not None:
                try:
                    retry_count = int(retry_count_str)
                    if retry_count > 0:
                        await task_repository.update_retry_count(db, task_id, retry_count)
                        await db.commit()
                except (ValueError, TypeError):
                    logger.warning(f"[MQ] retry_count 解析失败: {retry_count_str}")

        from app.service.task_service import TaskServiceAsync

        await TaskServiceAsync._execute_task_background(
            db_task_id, task_id, task_type, params_json
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


async def handle_low_rating_alert(body: dict[str, Any], headers: dict[str, Any]) -> None:
    """
    低分告警消费者 handler

    评价创建成功且 rating ≤ 2 时，由 feedback_service 发布消息触发。
    根据 rating 值和聚合统计决定告警级别：
      - 普通告警：所有低分评价（站内信）
      - 紧急告警：rating=1 且同算法 24h 内低分 ≥3 条（站内信）
      - 严重告警：全局 24h 低分率 >20%（站内信）

    消息格式:
        {
            "ratingId": int,
            "userId": int,
            "algorithmId": int,
            "rating": int,
            "comment": str | None,
            "createTime": str | None,
        }
    """
    set_current_user_id(SYSTEM_USER_ID)
    try:
        rating_id = body.get("ratingId")
        algorithm_id = body.get("algorithmId")
        rating_value = body.get("rating")
        comment = body.get("comment")

        if rating_id is None or rating_value is None:
            logger.error(f"[MQ] 低分告警消息格式无效: {body}")
            return

        logger.info(f"[MQ] 处理低分告警: ratingId={rating_id}, rating={rating_value}")

        async with get_db_session() as db:
            from sqlalchemy import select
            from app.models.entity.sys_user import SysRole, SysUser, SysUserRole
            from app.repository.feedback_repository import rating_repository
            from app.service.message_service import MessageService

            admin_stmt = (
                select(SysUser.id)
                .join(SysUserRole, SysUser.id == SysUserRole.user_id)
                .join(SysRole, SysUserRole.role_id == SysRole.id)
                .where(
                    SysUser.deleted == 0,
                    SysUser.status == 1,
                    SysRole.code.in_(["ROOT", "ADMIN"]),
                    SysRole.deleted == 0,
                )
                .distinct()
            )
            admin_ids = [row[0] for row in (await db.execute(admin_stmt)).fetchall()]
            if not admin_ids:
                logger.warning("[MQ] 无管理员用户，跳过低分告警")
                return

            await MessageService.send(db, {
                "type": "alert",
                "title": "收到低分评价",
                "content": f"评价ID {rating_id}，评分 {rating_value} 星，评论：{comment or '无'}",
                "recipientIds": admin_ids,
                "priority": 3,
                "bizModule": "feedback",
                "bizId": f"rating:low:{rating_id}",
            })

            if rating_value == 1 and algorithm_id:
                low_count = await rating_repository.count_low_ratings_by_algorithm_24h(
                    db, algorithm_id
                )
                if low_count >= LOW_RATING_URGENT_COUNT:
                    await MessageService.send(db, {
                        "type": "critical_alert",
                        "title": "低分评价紧急告警",
                        "content": f"算法ID {algorithm_id} 在24小时内收到 {low_count} 条低分评价，请紧急处理",
                        "recipientIds": admin_ids,
                        "priority": 1,
                        "bizModule": "feedback",
                        "bizId": f"rating:urgent:{algorithm_id}:{rating_id}",
                    })

            stats = await rating_repository.get_low_rating_stats_24h(db)
            if stats["total"] > 0:
                low_rate = stats["lowCount"] / stats["total"]
                if low_rate > LOW_RATING_SEVERE_RATE:
                    await MessageService.send(db, {
                        "type": "critical_alert",
                        "title": "全局低分率严重告警",
                        "content": f"24小时内全局低分率达 {low_rate * 100:.1f}%，超过 {LOW_RATING_SEVERE_RATE * 100:.0f}% 阈值，请立即处理",
                        "recipientIds": admin_ids,
                        "priority": 2,
                        "bizModule": "feedback",
                        "bizId": f"rating:severe:{rating_id}",
                    })

            await db.commit()

        logger.info(f"[MQ] 低分告警处理完成: ratingId={rating_id}")
    finally:
        set_current_user_id(None)

