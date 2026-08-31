"""
收藏管理 Service

核心业务逻辑：
- 添加收藏（容量校验 + 唯一约束防重 + 取消后重新收藏恢复 deleted=0）
- 批量取消收藏（逻辑删除）
- 收藏列表分页查询
- 收藏状态查询
- 收藏数量统计
- mark_invalid 预留回调
"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.favorite_repository import favorite_repository
from app.repository.member_repository import member_repository
from app.service.dict_service import get_dict_int

# 会员等级 → favorite_capacity 字典键映射（对齐收藏管理 §5/§11.1）
LEVEL_TO_CAPACITY_KEY = {
    "level_0": "default",
    "level_1": "vip1",
    "level_2": "vip2",
    "level_3": "svip",
}
# 设计默认容量（与 config/sql/data/sys_dict.sql 种子一致，缺键时回退）
CAPACITY_DEFAULTS = {
    "default": 200,
    "vip1": 500,
    "vip2": 1000,
    "svip": 3000,
}


def _format_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class FavoriteService:
    async def _get_capacity(self, db: AsyncSession, user_id: int) -> int:
        """根据用户会员等级返回收藏容量上限（sys_dict: favorite_capacity 实时读取）"""
        member = await member_repository.get_by_user_id(db, user_id)
        level_code = member.level_code if member else "level_0"
        key = LEVEL_TO_CAPACITY_KEY.get(level_code, "default")
        return await get_dict_int(db, "favorite_capacity", key, CAPACITY_DEFAULTS.get(key, 200))

    async def add(
        self,
        db: AsyncSession,
        user_id: int,
        target_type: str,
        target_id: int,
    ) -> int:
        """添加收藏

        1. 对象存在性校验（algorithm/dataset/result 必须存在，否则 A0401）
        2. 容量校验
        3. 单条 upsert（冲突时复活 deleted=0, is_invalid=0）
        """
        if not await favorite_repository.target_exists(db, target_type, target_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "收藏目标不存在")

        current_count = await favorite_repository.count_user_favorites(db, user_id)
        capacity = await self._get_capacity(db, user_id)
        if current_count >= capacity:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR,
                f"收藏已达上限（{capacity}条），请清理后重试",
            )

        # upsert：已收藏（deleted=0）幂等返回原 id；已取消（deleted=1）复活返回原 id
        return await favorite_repository.upsert_by_user_and_target(
            db, user_id, target_type, target_id
        )

    async def delete_by_ids(
        self,
        db: AsyncSession,
        user_id: int,
        ids: list[int],
    ) -> None:
        """批量取消收藏（逻辑删除）"""
        if not ids:
            return
        await favorite_repository.soft_delete_by_ids(db, ids, user_id)

    async def get_page(
        self,
        db: AsyncSession,
        user_id: int,
        query: dict,
    ) -> dict:
        """收藏列表分页查询"""
        items, total = await favorite_repository.get_page(
            db,
            user_id,
            query["pageNum"],
            query["pageSize"],
            target_type=query.get("targetType"),
            keywords=query.get("keywords"),
            sort_by=query.get("sortBy"),
            sort_order=query.get("sortOrder"),
        )

        list_data = [
            {
                "id": item["favorite"].id,
                "userId": item["favorite"].user_id,
                "targetType": item["favorite"].target_type,
                "targetId": item["favorite"].target_id,
                "targetName": item.get("target_name"),
                "targetSummary": None,
                "targetThumbnail": item.get("target_thumbnail"),
                "isInvalid": bool(item["favorite"].is_invalid),
                "createTime": _format_dt(item["favorite"].create_time),
            }
            for item in items
        ]
        return {"list": list_data, "total": total}

    async def get_status(
        self,
        db: AsyncSession,
        user_id: int,
        target_type: str,
        target_id: int,
    ) -> dict:
        """检查是否已收藏"""
        existing = await favorite_repository.get_by_user_and_target(
            db, user_id, target_type, target_id
        )
        return {
            "targetType": target_type,
            "targetId": target_id,
            "favorited": existing is not None,
        }

    async def get_count(
        self,
        db: AsyncSession,
        user_id: int,
        target_type: str | None = None,
    ) -> list[dict]:
        """收藏数量统计（按类型分组）"""
        counts = await favorite_repository.get_count_by_type(db, user_id, target_type)
        return [{"targetType": c["target_type"], "count": c["count"]} for c in counts]


favorite_service = FavoriteService()

