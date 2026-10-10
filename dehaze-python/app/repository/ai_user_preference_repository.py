from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_user_preference import SysAiUserPreference
from app.repository.base import BaseRepository


class AiUserPreferenceRepository(BaseRepository[SysAiUserPreference]):
    model = SysAiUserPreference

    async def get_by_user_id(self, db: AsyncSession, user_id: int) -> SysAiUserPreference | None:
        stmt = select(SysAiUserPreference).where(SysAiUserPreference.user_id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_default_model(
        self, db: AsyncSession, user_id: int, default_model: str | None, actor_id: int
    ) -> SysAiUserPreference:
        """写入用户偏好默认模型（每用户一行，冲突时复活软删行并覆盖值）"""
        stmt = mysql_insert(SysAiUserPreference).values(
            user_id=user_id,
            default_model=default_model,
            create_by=actor_id,
            update_by=actor_id,
        )
        stmt = stmt.on_duplicate_key_update(
            default_model=default_model,
            deleted=0,
            update_by=actor_id,
            update_time=func.now(),
        )
        await db.execute(stmt)
        result = await db.execute(
            select(SysAiUserPreference).where(SysAiUserPreference.user_id == user_id)
        )
        return result.scalar_one()


ai_user_preference_repository = AiUserPreferenceRepository()
