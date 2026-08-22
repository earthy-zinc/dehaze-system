"""
RabbitMQ 消费者 Handler

定义各队列的消息处理函数，由 Consumer 消费时回调。
handler 为薄壳：解析消息后交由对应 service 消费函数执行业务逻辑。
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.constants import SYSTEM_USER_ID
from app.database import get_db_session
from app.models.base import set_current_user_id
from app.service import task_service

logger = logging.getLogger(__name__)

LOW_RATING_URGENT_COUNT = 3
LOW_RATING_SEVERE_RATE = 0.20


async def handle_export_task(body: dict[str, Any], headers: dict[str, Any]) -> None:
    """
    导出任务消费者 handler

    解析导出任务消息，交由 task_service.consume_export_message 执行。
    """
    await task_service.consume_export_message(body, headers)
    logger.debug(f"[MQ] 导出任务 handler 处理完成: taskId={body.get('task_id')}")


async def handle_dlq_message(body: dict[str, Any], headers: dict[str, Any]) -> None:
    """
    死信队列消费者 handler

    解析死信消息，交由 task_service.consume_dlq_message 执行。
    """
    await task_service.consume_dlq_message(body, headers)
    logger.warning(f"[MQ] 死信 handler 处理完成: taskId={body.get('task_id')}")


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

        logger.debug(f"[MQ] 处理低分告警: ratingId={rating_id}, rating={rating_value}")

        async with get_db_session() as db:
            from app.repository.feedback_repository import rating_repository
            from app.repository.user_repository import user_repository
            from app.service.message_service import message_service

            admin_ids = await user_repository.list_active_admin_ids(db)
            if not admin_ids:
                logger.warning("[MQ] 无管理员用户，跳过低分告警")
                return

            await message_service.send(
                db,
                {
                    "type": "alert",
                    "title": "收到低分评价",
                    "content": (
                        f"评价ID {rating_id}，评分 {rating_value} 星，评论：{comment or '无'}"
                    ),
                    "recipientIds": admin_ids,
                    "priority": 3,
                    "bizModule": "feedback",
                    "bizId": f"rating:low:{rating_id}",
                },
            )

            if rating_value == 1 and algorithm_id:
                low_count = await rating_repository.count_low_ratings_by_algorithm_24h(
                    db, algorithm_id
                )
                if low_count >= LOW_RATING_URGENT_COUNT:
                    await message_service.send(
                        db,
                        {
                            "type": "critical_alert",
                            "title": "低分评价紧急告警",
                            "content": (
                                f"算法ID {algorithm_id} 在24小时内收到 {low_count} 条低分评价，"
                                "请紧急处理"
                            ),
                            "recipientIds": admin_ids,
                            "priority": 1,
                            "bizModule": "feedback",
                            "bizId": f"rating:urgent:{algorithm_id}:{rating_id}",
                        },
                    )

            stats = await rating_repository.get_low_rating_stats_24h(db)
            if stats["total"] > 0:
                low_rate = stats["lowCount"] / stats["total"]
                if low_rate > LOW_RATING_SEVERE_RATE:
                    await message_service.send(
                        db,
                        {
                            "type": "critical_alert",
                            "title": "全局低分率严重告警",
                            "content": (
                                f"24小时内全局低分率达 {low_rate * 100:.1f}%，"
                                f"超过 {LOW_RATING_SEVERE_RATE * 100:.0f}% 阈值，请立即处理"
                            ),
                            "recipientIds": admin_ids,
                            "priority": 2,
                            "bizModule": "feedback",
                            "bizId": f"rating:severe:{rating_id}",
                        },
                    )

            await db.commit()

        logger.debug(f"[MQ] 低分告警处理完成: ratingId={rating_id}")
    finally:
        set_current_user_id(None)
