"""
图像输入历史记录服务
对齐 dehaze-java SysInputHistoryServiceImpl 逻辑
"""
import logging
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.models.entity.sys_input_history import SysInputHistory
from app.repository.input_history_repository import input_history_repository
from app.utils.datetime_utils import format_time

logger = logging.getLogger(__name__)


class InputHistoryService:
    """图像输入历史记录服务"""

    @staticmethod
    async def list_history(
        db: AsyncSession,
        user_id: int,
        status: Optional[int] = None,
        input_source: Optional[str] = None,
        favorite_only: bool = False,
        keywords: Optional[str] = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页查询历史记录"""
        histories, total = await input_history_repository.get_paginated(
            db=db,
            user_id=user_id,
            status=status,
            input_source=input_source,
            favorite_only=favorite_only,
            keywords=keywords,
            page=page,
            size=size,
        )
        list_vo = [InputHistoryService._to_vo(h) for h in histories]
        return list_vo, total

    @staticmethod
    async def get_history(db: AsyncSession, history_id: int) -> Optional[dict[str, Any]]:
        """查询历史记录详情"""
        history = await input_history_repository.get_by_id(db, history_id)
        if not history:
            return None
        return InputHistoryService._to_vo(history)

    @staticmethod
    async def create_history(db: AsyncSession, data: dict[str, Any], user_id: int) -> int:
        """创建历史记录 (对齐 Java SysInputHistoryServiceImpl.createHistory)"""
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
            is_favorite=False,
            sync_status=0,
        )
        await db.commit()
        return history.id

    @staticmethod
    async def update_history(
        db: AsyncSession,
        history_id: int,
        is_favorite: Optional[bool],
        user_id: int,
    ) -> None:
        """更新历史记录（仅支持收藏切换，对齐 Java updateHistory）"""
        history = await input_history_repository.get_by_id(db, history_id)
        if not history:
            raise BusinessException("历史记录不存在")
        if history.user_id != user_id:
            raise BusinessException("无权操作该历史记录")
        if is_favorite is not None:
            history.is_favorite = is_favorite
            await db.flush()
        await db.commit()

    @staticmethod
    async def delete_history(db: AsyncSession, history_id: int, user_id: int) -> None:
        """删除单条历史记录（幂等，对齐 Java deleteHistory）"""
        await input_history_repository.delete_by_user(db, user_id, history_id)
        await db.commit()

    @staticmethod
    async def batch_delete(db: AsyncSession, ids: list[int], user_id: int) -> int:
        """批量删除历史记录（仅限本人）"""
        result = await input_history_repository.batch_delete_by_user(db, user_id, ids)
        await db.commit()
        return result

    @staticmethod
    async def clear_history(db: AsyncSession, user_id: int) -> int:
        """清空用户所有历史记录"""
        result = await input_history_repository.clear_by_user(db, user_id)
        await db.commit()
        return result

    @staticmethod
    async def sync_history(db: AsyncSession, user_id: int) -> int:
        """同步历史记录（对齐 Java syncHistory：标记所有未同步记录为已同步）"""
        count = await input_history_repository.mark_all_synced(db, user_id)
        await db.commit()
        return 1 if count > 0 else 0

    @staticmethod
    def _to_vo(history: SysInputHistory) -> dict[str, Any]:
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
            "isFavorite": history.is_favorite,
            "syncStatus": history.sync_status,
            "createTime": format_time(history.create_time),
            "updateTime": format_time(history.update_time),
        }
