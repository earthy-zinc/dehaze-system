from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_audit_update_values
from app.models.entity.sys_ai_model_cost import SysAiModelCost, SysAiModelCostDetail
from app.repository.base import BaseRepository


class AiModelCostRepository(BaseRepository[SysAiModelCost]):
    model = SysAiModelCost

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
            select(func.max(SysAiModelCost.price_version))
            .where(
                SysAiModelCost.model_id == model_id,
                SysAiModelCost.provider_id == provider_id,
            )
            .execution_options(include_deleted=True)
        )
        current = (await db.execute(stmt)).scalar()
        return (current or 0) + 1

    async def get_with_details(self, db: AsyncSession, cost_id: int) -> SysAiModelCost | None:
        """查询成本价格版本（含档位明细）"""
        cost = await self.get_by_id(db, cost_id)
        if cost is None:
            return None
        cost.details = await self.list_details(db, cost.id)
        return cost

    async def list_costs(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        *,
        keyword: str | None = None,
        model_id: str | None = None,
        provider_id: int | None = None,
    ) -> tuple[list[SysAiModelCost], int]:
        stmt = select(SysAiModelCost).order_by(
            SysAiModelCost.create_time.desc(), SysAiModelCost.id.desc()
        )
        if model_id:
            stmt = stmt.where(SysAiModelCost.model_id == model_id)
        if provider_id:
            stmt = stmt.where(SysAiModelCost.provider_id == provider_id)
        if keyword:
            stmt = stmt.where(
                SysAiModelCost.model_id.like(f"%{keyword}%")
            )
        costs, total = await self.paginate(db, stmt, page, size)
        return costs, total

    async def list_details(
        self,
        db: AsyncSession,
        price_id: int,
    ) -> list[SysAiModelCostDetail]:
        stmt = (
            select(SysAiModelCostDetail)
            .where(SysAiModelCostDetail.price_id == price_id)
            .order_by(SysAiModelCostDetail.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_details(
        self,
        db: AsyncSession,
        price_id: int,
        details: list[dict],
    ) -> list[SysAiModelCostDetail]:
        entities = [
            SysAiModelCostDetail(price_id=price_id, **d)
            for d in details
        ]
        if entities:
            await self.create_all(db, entities)
        return entities

    async def get_effective_version(
        self,
        db: AsyncSession,
        model_id: str,
        provider_id: int | None,
        at_time: datetime,
    ) -> SysAiModelCost | None:
        """按调用时刻选取生效的成本价格版本（status=生效 且 effective_from <= t < effective_to）"""
        stmt = select(SysAiModelCost).where(
            SysAiModelCost.model_id == model_id,
            SysAiModelCost.status == 1,
            SysAiModelCost.effective_from <= at_time,
        )
        if provider_id is not None:
            stmt = stmt.where(SysAiModelCost.provider_id == provider_id)
        stmt = stmt.where(
            (SysAiModelCost.effective_to.is_(None)) | (SysAiModelCost.effective_to > at_time)
        ).order_by(SysAiModelCost.effective_from.desc(), SysAiModelCost.price_version.desc())
        result = await db.execute(stmt)
        return result.scalars().first()


    async def soft_delete_details_by_price_id(self, db: AsyncSession, price_id: int) -> int:
        """逻辑删除某价格版本的全部档位明细"""
        stmt = (
            update(SysAiModelCostDetail)
            .where(SysAiModelCostDetail.price_id == price_id)
            .values(deleted=1, **get_audit_update_values())
        )
        result = await db.execute(stmt)
        return result.rowcount


ai_model_cost_repository = AiModelCostRepository()
