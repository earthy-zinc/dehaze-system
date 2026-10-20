from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_balance import SysBalance
from app.repository.base import BaseRepository


class BalanceAccountRepository(BaseRepository[SysBalance]):
    model = SysBalance

    async def get_by_user_id(self, db: AsyncSession, user_id: int) -> SysBalance | None:
        stmt = select(SysBalance).where(SysBalance.user_id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_or_create(
        self, db: AsyncSession, user_id: int, *, initial_balance: int = 0
    ) -> SysBalance:
        """按 user_id 获取余额账户，不存在则创建（初始可用余额）。"""
        account = await self.get_by_user_id(db, user_id)
        if account:
            return account
        account = SysBalance(user_id=user_id, balance=initial_balance, frozen_balance=0)
        await self.create(db, account)
        return account

    async def freeze(self, db: AsyncSession, user_id: int, amount: int) -> bool:
        """冻结：可用余额充足时 frozen_balance += amount。"""
        stmt = (
            update(SysBalance)
            .where(
                SysBalance.user_id == user_id,
                SysBalance.balance >= amount,
            )
            .values(frozen_balance=SysBalance.frozen_balance + amount)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def unfreeze(self, db: AsyncSession, user_id: int, amount: int) -> bool:
        """解冻：冻结余额充足时 frozen_balance -= amount。"""
        stmt = (
            update(SysBalance)
            .where(
                SysBalance.user_id == user_id,
                SysBalance.frozen_balance >= amount,
            )
            .values(frozen_balance=SysBalance.frozen_balance - amount)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def deduct(self, db: AsyncSession, user_id: int, amount: int) -> bool:
        """出账：balance 与 frozen_balance 同时扣减 amount。"""
        stmt = (
            update(SysBalance)
            .where(
                SysBalance.user_id == user_id,
                SysBalance.frozen_balance >= amount,
            )
            .values(
                balance=SysBalance.balance - amount,
                frozen_balance=SysBalance.frozen_balance - amount,
            )
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def refund(self, db: AsyncSession, user_id: int, amount: int) -> bool:
        """回充可用余额：balance += amount（amount 为负即扣减）。"""
        stmt = (
            update(SysBalance)
            .where(SysBalance.user_id == user_id)
            .values(balance=SysBalance.balance + amount)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0


balance_account_repository = BalanceAccountRepository()
