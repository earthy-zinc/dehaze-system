from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_billing_anomaly import SysAiBillingAnomaly
from app.repository.base import BaseRepository


class AiBillingAnomalyRepository(BaseRepository[SysAiBillingAnomaly]):
    model = SysAiBillingAnomaly

    async def create_anomaly(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        billing_id: int | None,
        anomaly_type: str,
        detail: str,
        trigger_at: datetime,
    ) -> SysAiBillingAnomaly:
        anomaly = SysAiBillingAnomaly(
            user_id=user_id,
            billing_id=billing_id,
            anomaly_type=anomaly_type,
            detail=detail,
            trigger_at=trigger_at,
        )
        return await self.create(db, anomaly)

    async def list_page(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        *,
        user_id: int | None = None,
        anomaly_type: str | None = None,
        status: int | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> tuple[list[SysAiBillingAnomaly], int]:
        stmt = select(SysAiBillingAnomaly)
        if user_id is not None:
            stmt = stmt.where(SysAiBillingAnomaly.user_id == user_id)
        if anomaly_type:
            stmt = stmt.where(SysAiBillingAnomaly.anomaly_type == anomaly_type)
        if status is not None:
            stmt = stmt.where(SysAiBillingAnomaly.status == status)
        if date_start is not None:
            stmt = stmt.where(SysAiBillingAnomaly.trigger_at >= date_start)
        if date_end is not None:
            stmt = stmt.where(SysAiBillingAnomaly.trigger_at < date_end)
        stmt = stmt.order_by(SysAiBillingAnomaly.trigger_at.desc(), SysAiBillingAnomaly.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def count_group_by_type(
        self,
        db: AsyncSession,
        *,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> list[tuple[str, int]]:
        """按异常类型聚合计数（趋势面板数据源）"""
        stmt = select(SysAiBillingAnomaly.anomaly_type, func.count().label("cnt"))
        if date_start is not None:
            stmt = stmt.where(SysAiBillingAnomaly.trigger_at >= date_start)
        if date_end is not None:
            stmt = stmt.where(SysAiBillingAnomaly.trigger_at < date_end)
        stmt = stmt.group_by(SysAiBillingAnomaly.anomaly_type)
        result = await db.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]


ai_billing_anomaly_repository = AiBillingAnomalyRepository()
