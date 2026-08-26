from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_balance_log import SysBalanceLog
from app.repository.base import BaseRepository


class BalanceLogRepository(BaseRepository[SysBalanceLog]):
    model = SysBalanceLog

    async def list_by_user(
        self, db: AsyncSession, user_id: int
    ) -> list[SysBalanceLog]:
        stmt = (
            select(SysBalanceLog)
            .where(SysBalanceLog.user_id == user_id)
            .order_by(SysBalanceLog.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_log(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        change_type: str,
        amount: int,
        balance_after: int,
        related_id: int | None = None,
    ) -> None:
        await self.create(
            db,
            SysBalanceLog(
                user_id=user_id,
                change_type=change_type,
                amount=amount,
                balance_after=balance_after,
                related_id=related_id,
            ),
        )


balance_log_repository = BalanceLogRepository()
