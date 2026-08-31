from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_voice_model import SysVoiceModel
from app.repository.base import BaseRepository


class VoiceModelRepository(BaseRepository[SysVoiceModel]):
    model = SysVoiceModel

    async def get_by_model_and_provider(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int,
    ) -> SysVoiceModel | None:
        """按 model_id + provider_id 联合查询（含已删除，供唯一性校验使用）"""
        stmt = select(SysVoiceModel).where(
            SysVoiceModel.model_id == model_id,
            SysVoiceModel.provider_id == provider_id,
        )
        # 绕过软删过滤查全表，避免软删后同组合查重漏检而依赖 DB 唯一索引报错
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_enabled(
        self, db: AsyncSession, engine_type: str, model_type: str | None = None
    ) -> list[SysVoiceModel]:
        """查询某能力维度下启用的模型/音色（status=1 且未删除，可选按 model_type 过滤）"""
        stmt = select(SysVoiceModel).where(
            SysVoiceModel.engine_type == engine_type, SysVoiceModel.status == 1
        )
        if model_type:
            stmt = stmt.where(SysVoiceModel.model_type == model_type)
        stmt = stmt.order_by(SysVoiceModel.provider_id, SysVoiceModel.model_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_engine_type(
        self, db: AsyncSession, engine_type: str | None = None
    ) -> list[SysVoiceModel]:
        """查询模型/音色列表（管理端展示，含全部状态；可选按 engine_type 过滤）"""
        stmt = select(SysVoiceModel)
        if engine_type:
            stmt = stmt.where(SysVoiceModel.engine_type == engine_type)
        stmt = stmt.order_by(SysVoiceModel.provider_id, SysVoiceModel.model_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())


voice_model_repository = VoiceModelRepository()
