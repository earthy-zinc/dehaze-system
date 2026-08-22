"""语音热词仓储层

提供 sys_voice_hotword 表的持久化访问。
"""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_voice_hotword import SysVoiceHotword
from app.repository.base import BaseRepository


class VoiceHotwordRepository(BaseRepository[SysVoiceHotword]):
    """语音热词仓储"""

    model = SysVoiceHotword

    async def get_by_id(self, db: AsyncSession, hotword_id: int) -> SysVoiceHotword | None:
        """按 ID 查询未删除的热词"""
        stmt = select(SysVoiceHotword).where(
            SysVoiceHotword.id == hotword_id, SysVoiceHotword.deleted == 0
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    async def list_by_scope(
        self,
        db: AsyncSession,
        scope: str,
        user_id: int | None,
    ) -> list[SysVoiceHotword]:
        """查询指定作用域下的热词（deleted=0）"""
        stmt = select(SysVoiceHotword).where(
            SysVoiceHotword.scope == scope, SysVoiceHotword.deleted == 0
        )
        if user_id is not None:
            stmt = stmt.where(SysVoiceHotword.user_id == user_id)
        else:
            stmt = stmt.where(SysVoiceHotword.user_id.is_(None))
        stmt = stmt.order_by(SysVoiceHotword.create_time.asc())
        return list((await db.execute(stmt)).scalars().all())

    async def count_user_hotwords(self, db: AsyncSession, user_id: int) -> int:
        """统计用户级热词数量（deleted=0）"""
        stmt = select(SysVoiceHotword).where(
            SysVoiceHotword.scope == "user",
            SysVoiceHotword.user_id == user_id,
            SysVoiceHotword.deleted == 0,
        )
        return await self.count(db, stmt)

    async def soft_delete(self, db: AsyncSession, hotword_id: int) -> bool:
        """软删除热词，返回是否命中"""
        stmt = (
            update(SysVoiceHotword)
            .where(SysVoiceHotword.id == hotword_id, SysVoiceHotword.deleted == 0)
            .values(deleted=1)
        )
        result = await db.execute(stmt)
        return result.rowcount > 0


# 模块级单例
voice_hotword_repository = VoiceHotwordRepository()
