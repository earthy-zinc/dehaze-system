from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_voice_provider_key import SysVoiceProviderKey
from app.repository.base import BaseRepository


class VoiceProviderKeyRepository(BaseRepository[SysVoiceProviderKey]):
    model = SysVoiceProviderKey

    async def get_by_hash(self, db: AsyncSession, key_hash: str) -> SysVoiceProviderKey | None:
        """按哈希查重"""
        stmt = select(SysVoiceProviderKey).where(SysVoiceProviderKey.key_hash == key_hash)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_enabled_by_provider(
        self, db: AsyncSession, provider_id: int
    ) -> list[SysVoiceProviderKey]:
        """查询某引擎下启用且未过期的 Key（按 priority 升序）"""
        stmt = (
            select(SysVoiceProviderKey)
            .where(
                SysVoiceProviderKey.provider_id == provider_id,
                SysVoiceProviderKey.status == 1,
                (SysVoiceProviderKey.expires_at.is_(None))
                | (SysVoiceProviderKey.expires_at > datetime.now()),
            )
            .order_by(SysVoiceProviderKey.priority, SysVoiceProviderKey.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


voice_provider_key_repository = VoiceProviderKeyRepository()
