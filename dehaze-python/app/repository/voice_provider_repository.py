from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_voice_provider import SysVoiceProvider
from app.repository.base import BaseRepository, escape_like


class VoiceProviderRepository(BaseRepository[SysVoiceProvider]):
    model = SysVoiceProvider

    async def get_by_provider_code(
        self,
        db: AsyncSession,
        provider_code: str,
        include_deleted: bool = False,
    ) -> SysVoiceProvider | None:
        """按 provider_code 查询（联合唯一为 provider_code+engine_type，单列查询需结合 engine_type 使用）"""
        stmt = (
            select(SysVoiceProvider)
            .where(SysVoiceProvider.provider_code == provider_code)
            .order_by(SysVoiceProvider.engine_type)
        )
        if include_deleted:
            stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_by_provider_and_engine(
        self,
        db: AsyncSession,
        provider_code: str,
        engine_type: str,
        include_deleted: bool = False,
    ) -> SysVoiceProvider | None:
        """按业务唯一键 provider_code + engine_type 查询（同厂商按能力注册多条；查重须 include_deleted 绕过软删）"""
        stmt = select(SysVoiceProvider).where(
            SysVoiceProvider.provider_code == provider_code,
            SysVoiceProvider.engine_type == engine_type,
        )
        if include_deleted:
            stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalars().first()

    async def list_enabled(self, db: AsyncSession, engine_type: str) -> list[SysVoiceProvider]:
        """查询某能力维度下启用的引擎（status=1 且未删除），按 sort_order 排序"""
        stmt = (
            select(SysVoiceProvider)
            .where(SysVoiceProvider.engine_type == engine_type, SysVoiceProvider.status == 1)
            .order_by(SysVoiceProvider.sort_order, SysVoiceProvider.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_default(self, db: AsyncSession, engine_type: str) -> SysVoiceProvider | None:
        """查询某能力维度下的默认启用引擎（is_default=1 且 status=1）"""
        stmt = select(SysVoiceProvider).where(
            SysVoiceProvider.engine_type == engine_type,
            SysVoiceProvider.is_default == 1,
            SysVoiceProvider.status == 1,
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def paginate_providers(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
        engine_type: str | None = None,
    ) -> tuple[list[SysVoiceProvider], int]:
        """分页查询引擎（keyword 匹配 display_name/provider_code，可选按 engine_type 过滤）"""
        stmt = select(SysVoiceProvider)
        if keyword:
            escaped = escape_like(keyword)
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                (SysVoiceProvider.display_name.like(pattern, escape="\\"))
                | (SysVoiceProvider.provider_code.like(pattern, escape="\\"))
            )
        if engine_type:
            stmt = stmt.where(SysVoiceProvider.engine_type == engine_type)
        stmt = stmt.order_by(SysVoiceProvider.sort_order, SysVoiceProvider.id)
        return await self.paginate(db, stmt, page, size)


voice_provider_repository = VoiceProviderRepository()
