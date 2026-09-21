"""
图像输入历史记录服务
对齐 dehaze-java SysInputHistoryServiceImpl 逻辑
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_input_history import SysInputHistory
from app.repository.input_history_repository import input_history_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.utils.datetime_utils import format_time

# 无会员档案/权益缺失时的兜底保留条数（对齐 sys_member_benefit.level_0 种子值）
DEFAULT_HISTORY_RETENTION = 100


class InputHistoryService:
    """图像输入历史记录服务"""

    async def list_history(
        self,
        db: AsyncSession,
        user_id: int,
        status: int | None = None,
        input_source: str | None = None,
        keywords: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页查询历史记录"""
        histories, total = await input_history_repository.get_paginated(
            db=db,
            user_id=user_id,
            status=status,
            input_source=input_source,
            keywords=keywords,
            page=page,
            size=size,
        )
        list_vo = [self._to_vo(h) for h in histories]
        return list_vo, total

    async def get_history(
        self, db: AsyncSession, history_id: int, user_id: int
    ) -> dict[str, Any] | None:
        """查询历史记录详情（仅限本人）"""
        history = await input_history_repository.get_by_id(db, history_id)
        if not history:
            return None
        if history.user_id != user_id:
            return None
        return self._to_vo(history)

    async def create_history(self, db: AsyncSession, data: dict[str, Any], user_id: int) -> int:
        """创建历史记录 (对齐 Java SysInputHistoryServiceImpl.createHistory)"""
        # 配额检查：超过会员等级保留条数时自动清理最旧记录（文档 §4.3）
        retention = await self._get_history_retention(db, user_id)
        count = await input_history_repository.count_by_user(db, user_id)
        if count >= retention:
            await input_history_repository.delete_oldest(db, user_id)

        history = await input_history_repository.create_history(
            db=db,
            user_id=user_id,
            original_image_url=data.get("originalImageUrl"),
            original_thumbnail_url=data.get("originalThumbnailUrl"),
            result_image_url=data.get("resultImageUrl"),
            result_thumbnail_url=data.get("resultThumbnailUrl"),
            algorithm_id=data.get("algorithmId"),
            algorithm_name=data.get("algorithmName"),
            algorithm_params=data.get("algorithmParams"),
            processing_time=data.get("processingTime"),
            status=data.get("status", 3),
            input_source=data.get("inputSource", "upload"),
        )
        return history.id

    async def delete_history(self, db: AsyncSession, history_id: int, user_id: int) -> None:
        """删除单条历史记录（幂等，对齐 Java deleteHistory）"""
        await input_history_repository.delete_by_user(db, user_id, history_id)

    async def batch_delete(self, db: AsyncSession, ids: list[int], user_id: int) -> int:
        """批量删除历史记录（仅限本人），返回实际删除数量"""
        return await input_history_repository.batch_delete_by_user(db, user_id, ids)

    async def clear_history(self, db: AsyncSession, user_id: int) -> int:
        """清空用户所有历史记录"""
        return await input_history_repository.clear_by_user(db, user_id)

    async def _get_history_retention(self, db: AsyncSession, user_id: int) -> int:
        """按会员等级取历史保留条数（sys_member_benefit.history_retention）"""
        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            return DEFAULT_HISTORY_RETENTION
        benefit = await member_benefit_repository.get_by_level_code(db, member.level_code)
        if not benefit or not benefit.history_retention:
            return DEFAULT_HISTORY_RETENTION
        return benefit.history_retention

    def _to_vo(self, history: SysInputHistory) -> dict[str, Any]:
        """转换为 VO (对齐 Java InputHistoryVO 字段)"""
        return {
            "id": history.id,
            "userId": history.user_id,
            "originalImageUrl": history.original_image_url,
            "originalThumbnailUrl": history.original_thumbnail_url,
            "resultImageUrl": history.result_image_url,
            "resultThumbnailUrl": history.result_thumbnail_url,
            "algorithmId": history.algorithm_id,
            "algorithmName": history.algorithm_name,
            "algorithmParams": history.algorithm_params,
            "processingTime": history.processing_time,
            "status": history.status,
            "inputSource": history.input_source,
            "createTime": format_time(history.create_time),
            "updateTime": format_time(history.update_time),
        }


input_history_service = InputHistoryService()
