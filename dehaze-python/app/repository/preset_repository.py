"""参数预设 Repository"""
from typing import Optional

from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_preset import SysPreset
from app.repository.base import BaseRepository


class PresetRepository(BaseRepository[SysPreset]):
    model = SysPreset

    async def list_presets(
        self,
        db: AsyncSession,
        user_id: int,
        algorithm_id: Optional[int] = None,
        is_system: Optional[bool] = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysPreset], int]:
        """获取预设列表：系统预设 + 用户自定义，支持分页和筛选"""
        conditions = []
        if is_system is True:
            conditions.append(SysPreset.type == "system")
        elif is_system is False:
            conditions.append((SysPreset.type == "custom") & (SysPreset.user_id == user_id))
        else:
            conditions.append(
                or_(
                    SysPreset.type == "system",
                    (SysPreset.type == "custom") & (SysPreset.user_id == user_id),
                )
            )

        if algorithm_id is not None:
            conditions.append(SysPreset.algorithm_id == algorithm_id)

        base = select(SysPreset).where(*conditions)

        # count
        count_stmt = select(func.count()).select_from(base.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0

        # page
        stmt = base.order_by(SysPreset.create_time.asc()).offset((page - 1) * size).limit(size)
        result = await db.execute(stmt)
        return list(result.scalars().all()), total

    async def get_by_user_and_name(self, db: AsyncSession, user_id: int, name: str) -> Optional[SysPreset]:
        """按用户和名称查找预设"""
        stmt = select(SysPreset).where(
            SysPreset.user_id == user_id,
            SysPreset.name == name,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_system_presets(self, db: AsyncSession) -> int:
        """统计系统预设数量"""
        stmt = select(func.count()).select_from(SysPreset).where(SysPreset.type == "system")
        result = await db.execute(stmt)
        return result.scalar() or 0


preset_repository = PresetRepository()
