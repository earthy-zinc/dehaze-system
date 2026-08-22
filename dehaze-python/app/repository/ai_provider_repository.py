from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_model import SysAiModel
from app.models.entity.sys_ai_provider import SysAiProvider
from app.repository.base import BaseRepository, escape_like


class AiProviderRepository(BaseRepository[SysAiProvider]):
    model = SysAiProvider

    async def get_by_provider_code(
        self,
        db: AsyncSession,
        provider_code: str,
        include_deleted: bool = False,
    ) -> SysAiProvider | None:
        """按业务唯一键 provider_code 查询（查重时 include_deleted=True 绕过软删过滤）"""
        stmt = select(SysAiProvider).where(SysAiProvider.provider_code == provider_code)
        if include_deleted:
            stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_enabled(self, db: AsyncSession) -> list[SysAiProvider]:
        """查询启用的供应商（status=1 且未删除），按 sort_order 排序"""
        stmt = (
            select(SysAiProvider)
            .where(SysAiProvider.status == 1)
            .order_by(SysAiProvider.sort_order, SysAiProvider.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def paginate_providers(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str | None = None,
    ) -> tuple[list[SysAiProvider], int]:
        """分页查询供应商（keyword 匹配 display_name/provider_code）"""
        stmt = select(SysAiProvider)
        if keyword:
            escaped = escape_like(keyword)
            pattern = f"%{escaped}%"
            stmt = stmt.where(
                (SysAiProvider.display_name.like(pattern, escape="\\"))
                | (SysAiProvider.provider_code.like(pattern, escape="\\"))
            )
        stmt = stmt.order_by(SysAiProvider.sort_order, SysAiProvider.id)
        return await self.paginate(db, stmt, page, size)

    async def count_enabled_models(self, db: AsyncSession, provider_id: int) -> int:
        """统计该供应商下启用的模型数量（用于删除前校验）"""
        stmt = (
            select(func.count())
            .select_from(SysAiModel)
            .where(
                SysAiModel.provider_id == provider_id,
                SysAiModel.deleted == 0,
                SysAiModel.status == 1,
            )
        )
        return (await db.execute(stmt)).scalar() or 0


ai_provider_repository = AiProviderRepository()
