from datetime import datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_credit_log import SysAiCreditLog
from app.repository.base import BaseRepository


class AiCreditLogRepository(BaseRepository[SysAiCreditLog]):
    model = SysAiCreditLog

    async def create_log(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        source: str,
        amount: Decimal,
        balance_after: Decimal,
        related_id: int | None = None,
        reason: str | None = None,
        operator_id: int | None = None,
    ) -> SysAiCreditLog:
        log = SysAiCreditLog(
            user_id=user_id,
            source=source,
            amount=amount,
            balance_after=balance_after,
            related_id=related_id,
            reason=reason,
            operator_id=operator_id,
        )
        return await self.create(db, log)

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        *,
        source: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[list[SysAiCreditLog], int]:
        stmt = select(SysAiCreditLog).where(SysAiCreditLog.user_id == user_id)
        if source:
            stmt = stmt.where(SysAiCreditLog.source == source)
        if start:
            stmt = stmt.where(SysAiCreditLog.create_time >= start)
        if end:
            stmt = stmt.where(SysAiCreditLog.create_time <= end)

        stmt = stmt.order_by(SysAiCreditLog.create_time.desc(), SysAiCreditLog.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def sum_amount_by_user_and_source(
        self,
        db: AsyncSession,
        user_id: int,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Decimal]:
        """按 source 维度汇总某时间段内的金额变动"""
        stmt = select(
            SysAiCreditLog.source,
            func.coalesce(func.sum(SysAiCreditLog.amount), 0),
        ).where(SysAiCreditLog.user_id == user_id)
        if start:
            stmt = stmt.where(SysAiCreditLog.create_time >= start)
        if end:
            stmt = stmt.where(SysAiCreditLog.create_time <= end)
        stmt = stmt.group_by(SysAiCreditLog.source)

        rows = (await db.execute(stmt)).all()
        return {r[0]: Decimal(str(r[1])) for r in rows}

    async def distinct_user_ids_by_source(
        self,
        db: AsyncSession,
        source: str,
        start: datetime,
        end: datetime,
    ) -> list[int]:
        """查询某时间段内指定来源有流水的去重用户 ID 列表"""
        stmt = (
            select(SysAiCreditLog.user_id)
            .where(
                SysAiCreditLog.source == source,
                SysAiCreditLog.create_time >= start,
                SysAiCreditLog.create_time <= end,
            )
            .distinct()
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]

    async def get_balance_at_or_before(
        self,
        db: AsyncSession,
        user_id: int,
        end: datetime,
    ) -> Decimal:
        """查询指定时间点前最近一笔流水的变动后余额（用于账期期初余额）"""
        stmt = (
            select(SysAiCreditLog.balance_after)
            .where(
                SysAiCreditLog.user_id == user_id,
                SysAiCreditLog.create_time <= end,
            )
            .order_by(SysAiCreditLog.create_time.desc(), SysAiCreditLog.id.desc())
            .limit(1)
        )
        row = (await db.execute(stmt)).first()
        return Decimal(str(row[0])) if row else Decimal(0)


ai_credit_log_repository = AiCreditLogRepository()
