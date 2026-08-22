from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_provider_key import SysAiProviderKey
from app.repository.base import BaseRepository


class AiProviderKeyRepository(BaseRepository[SysAiProviderKey]):
    model = SysAiProviderKey

    async def get_by_hash(self, db: AsyncSession, key_hash: str) -> SysAiProviderKey | None:
        """按哈希查重"""
        stmt = select(SysAiProviderKey).where(SysAiProviderKey.key_hash == key_hash)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_provider(self, db: AsyncSession, provider_id: int) -> list[SysAiProviderKey]:
        """查询某供应商下所有 Key（按 priority 升序）"""
        stmt = (
            select(SysAiProviderKey)
            .where(SysAiProviderKey.provider_id == provider_id)
            .order_by(SysAiProviderKey.priority, SysAiProviderKey.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_enabled_by_provider(
        self, db: AsyncSession, provider_id: int
    ) -> list[SysAiProviderKey]:
        """查询某供应商下启用且未过期的 Key（按 priority 升序）"""
        stmt = (
            select(SysAiProviderKey)
            .where(
                SysAiProviderKey.provider_id == provider_id,
                SysAiProviderKey.status == 1,
                (SysAiProviderKey.expires_at.is_(None))
                | (SysAiProviderKey.expires_at > datetime.now()),
            )
            .order_by(SysAiProviderKey.priority, SysAiProviderKey.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_enabled_by_provider(self, db: AsyncSession, provider_id: int) -> int:
        """统计某供应商下启用且未过期的 Key 数量"""
        stmt = (
            select(func.count())
            .select_from(SysAiProviderKey)
            .where(
                SysAiProviderKey.provider_id == provider_id,
                SysAiProviderKey.status == 1,
                (SysAiProviderKey.expires_at.is_(None))
                | (SysAiProviderKey.expires_at > datetime.now()),
            )
        )
        return (await db.execute(stmt)).scalar() or 0

    async def batch_update_last_used(
        self,
        db: AsyncSession,
        updates: list[tuple[int, datetime, int]],
    ) -> None:
        """批量更新 last_used_at + last_used_by（异步刷库用）"""
        if not updates:
            return
        for key_id, used_at, user_id in updates:
            stmt = (
                update(SysAiProviderKey)
                .where(SysAiProviderKey.id == key_id)
                .values(last_used_at=used_at, last_used_by=user_id)
            )
            await db.execute(stmt)


ai_provider_key_repository = AiProviderKeyRepository()
