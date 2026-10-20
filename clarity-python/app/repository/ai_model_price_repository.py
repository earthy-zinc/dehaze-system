from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_audit_update_values
from app.models.entity.sys_ai_model_price import SysAiModelPrice, SysAiModelPriceDetail
from app.repository.base import BaseRepository


class AiModelPriceRepository(BaseRepository[SysAiModelPrice]):
    model = SysAiModelPrice

    async def next_price_version(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int,
    ) -> int:
        """同模型同供应商的价格版本号递增。

        版本号按全部历史（含软删）递增：联合唯一键 (model_id, provider_id, price_version)
        使软删版本号不可复用（类别②），删除后不得回退版本号。
        """
        stmt = (
            select(func.max(SysAiModelPrice.price_version))
            .where(
                SysAiModelPrice.model_id == model_id,
                SysAiModelPrice.provider_id == provider_id,
            )
            .execution_options(include_deleted=True)
        )
        current = (await db.execute(stmt)).scalar()
        return (current or 0) + 1

    async def get_with_details(
        self, db: AsyncSession, price_id: int
    ) -> tuple[SysAiModelPrice, list[SysAiModelPriceDetail]] | None:
        """查询用户售价价格版本（含档位明细），返回 (版本, 档位明细列表)"""
        price = await self.get_by_id(db, price_id)
        if price is None:
            return None
        return price, await self.list_details(db, price.id)

    async def list_prices(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        *,
        model_id: str | None = None,
        provider_id: int | None = None,
    ) -> tuple[list[SysAiModelPrice], int]:
        stmt = select(SysAiModelPrice).order_by(
            SysAiModelPrice.create_time.desc(), SysAiModelPrice.id.desc()
        )
        if model_id:
            stmt = stmt.where(SysAiModelPrice.model_id == model_id)
        if provider_id:
            stmt = stmt.where(SysAiModelPrice.provider_id == provider_id)
        prices, total = await self.paginate(db, stmt, page, size)
        return prices, total

    async def list_details(
        self,
        db: AsyncSession,
        price_id: int,
    ) -> list[SysAiModelPriceDetail]:
        stmt = (
            select(SysAiModelPriceDetail)
            .where(SysAiModelPriceDetail.price_id == price_id)
            .order_by(SysAiModelPriceDetail.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_details(
        self,
        db: AsyncSession,
        price_id: int,
        details: list[dict],
    ) -> list[SysAiModelPriceDetail]:
        entities = [SysAiModelPriceDetail(price_id=price_id, **d) for d in details]
        if entities:
            # 明细实体非本仓储主模型 T，直接 add_all 批量插入，不走 create_all（其入参限主模型）
            db.add_all(entities)
            await db.flush()
        return entities

    async def get_effective_version(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int | None,
        at_time: datetime,
    ) -> SysAiModelPrice | None:
        """按调用时刻选取生效的用户售价版本（status=生效 且 effective_from <= t < effective_to）"""
        stmt = select(SysAiModelPrice).where(
            SysAiModelPrice.model_id == model_id,
            SysAiModelPrice.status == 1,
            SysAiModelPrice.effective_from <= at_time,
        )
        if provider_id is not None:
            stmt = stmt.where(SysAiModelPrice.provider_id == provider_id)
        stmt = stmt.where(
            (SysAiModelPrice.effective_to.is_(None)) | (SysAiModelPrice.effective_to > at_time)
        ).order_by(SysAiModelPrice.effective_from.desc(), SysAiModelPrice.price_version.desc())
        result = await db.execute(stmt)
        return result.scalars().first()

    async def soft_delete_details_by_price_id(self, db: AsyncSession, price_id: int) -> int:
        """逻辑删除某价格版本的全部档位明细"""
        stmt = (
            update(SysAiModelPriceDetail)
            .where(SysAiModelPriceDetail.price_id == price_id)
            .values(deleted=SysAiModelPriceDetail.id, **get_audit_update_values())
        )
        result = await db.execute(stmt)
        return result.rowcount


ai_model_price_repository = AiModelPriceRepository()
