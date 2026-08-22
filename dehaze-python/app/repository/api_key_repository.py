"""API 密钥数据访问层：查询/写入统一经此，service 不构建 SQL。
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.api_key import SysApiKey
from app.repository.base import BaseRepository


class ApiKeyRepository(BaseRepository[SysApiKey]):
    model = SysApiKey

    async def list_active_by_user(self, db: AsyncSession, user_id: int) -> list[SysApiKey]:
        """用户的未吊销密钥列表（revoked_at IS NULL），按 id 倒序"""
        stmt = (
            select(SysApiKey)
            .where(SysApiKey.user_id == user_id, SysApiKey.revoked_at.is_(None))
            .order_by(SysApiKey.id.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_active_by_id_and_user(
        self, db: AsyncSession, key_id: int, user_id: int
    ) -> SysApiKey | None:
        """按 id + 归属用户取未吊销密钥（吊销/不存在/越权均返回 None）"""
        stmt = select(SysApiKey).where(
            SysApiKey.id == key_id,
            SysApiKey.user_id == user_id,
            SysApiKey.revoked_at.is_(None),
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke(self, db: AsyncSession, entity: SysApiKey) -> None:
        """吊销密钥：设置 revoked_at，永久保留 hash 以拒绝已泄露的旧密钥"""
        entity.revoked_at = datetime.now()
        await db.flush()


api_key_repository = ApiKeyRepository()
