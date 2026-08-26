"""平台人民币余额账户服务：冻结/解冻/扣减/退款回充。

本账户为人民币交易媒介（充值/支付/退款退回），与 AI 计费模块的积分账户职责分离。
所有变动使用乐观锁（version 字段 CAS）单行 UPDATE，失败重试 ≤3 次，避免并发超卖。
每笔变动写入余额流水（sys_balance_log），供资金审计追溯。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.balance_account_repository import balance_account_repository
from app.repository.balance_log_repository import balance_log_repository

logger = logging.getLogger(__name__)

_CAS_RETRY = 3


class BalanceAccountService:
    def __init__(
        self,
        balance_account_repository=balance_account_repository,
        balance_log_repository=balance_log_repository,
    ):
        self.balance_account_repository = balance_account_repository
        self.balance_log_repository = balance_log_repository

    async def get_account(self, db: AsyncSession, user_id: int):
        return await self.balance_account_repository.get_by_user_id(db, user_id)

    async def get_balance(self, db: AsyncSession, user_id: int) -> dict:
        account = await self.balance_account_repository.get_or_create(db, user_id)
        return {
            "balance": account.balance,
            "frozenBalance": account.frozen_balance,
        }

    def _validate_amount(self, amount: int) -> None:
        if amount <= 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "金额必须为正数")

    async def freeze(self, db: AsyncSession, user_id: int, amount: int) -> None:
        """冻结可用余额：余额不足抛 BALANCE_INSUFFICIENT；乐观锁 CAS 失败重试。"""
        self._validate_amount(amount)
        account = await self.balance_account_repository.get_or_create(db, user_id)
        if account.balance < amount:
            raise BusinessException(ResultCode.BALANCE_INSUFFICIENT)
        for _ in range(_CAS_RETRY):
            account = await self.balance_account_repository.get_or_create(db, user_id)
            if account.balance < amount:
                raise BusinessException(ResultCode.BALANCE_INSUFFICIENT)
            if await self.balance_account_repository.freeze(
                db, user_id, amount, account.version
            ):
                await self.balance_log_repository.create_log(
                    db,
                    user_id=user_id,
                    change_type="freeze",
                    amount=-amount,
                    balance_after=account.balance,
                )
                return
        raise BusinessException(ResultCode.BALANCE_INSUFFICIENT, "余额冻结失败，请重试")

    async def unfreeze(self, db: AsyncSession, user_id: int, amount: int) -> None:
        """解冻余额（超时取消/支付失败场景）。"""
        self._validate_amount(amount)
        for _ in range(_CAS_RETRY):
            account = await self.balance_account_repository.get_or_create(db, user_id)
            if await self.balance_account_repository.unfreeze(
                db, user_id, amount, account.version
            ):
                await self.balance_log_repository.create_log(
                    db,
                    user_id=user_id,
                    change_type="unfreeze",
                    amount=amount,
                    balance_after=account.balance,
                )
                return
        raise BusinessException(ResultCode.BUSINESS_ERROR, "余额解冻失败，请重试")

    async def deduct(self, db: AsyncSession, user_id: int, amount: int) -> None:
        """扣减冻结余额出账（balance 与 frozen_balance 同时扣减）。"""
        self._validate_amount(amount)
        for _ in range(_CAS_RETRY):
            account = await self.balance_account_repository.get_or_create(db, user_id)
            if await self.balance_account_repository.deduct(
                db, user_id, amount, account.version
            ):
                await self.balance_log_repository.create_log(
                    db,
                    user_id=user_id,
                    change_type="consume",
                    amount=-amount,
                    balance_after=account.balance - amount,
                )
                return
        raise BusinessException(ResultCode.BUSINESS_ERROR, "余额扣减失败，请重试")

    async def refund(self, db: AsyncSession, user_id: int, amount: int) -> None:
        """退款回充可用余额。"""
        self._validate_amount(amount)
        for _ in range(_CAS_RETRY):
            account = await self.balance_account_repository.get_or_create(db, user_id)
            if await self.balance_account_repository.refund(
                db, user_id, amount, account.version
            ):
                await self.balance_log_repository.create_log(
                    db,
                    user_id=user_id,
                    change_type="refund",
                    amount=amount,
                    balance_after=account.balance + amount,
                )
                return
        raise BusinessException(ResultCode.BUSINESS_ERROR, "余额回充失败，请重试")

    async def withdraw(self, db: AsyncSession, user_id: int, amount: int) -> None:
        """余额退款扣减可用余额（原路退回后 balance 减少）。"""
        self._validate_amount(amount)
        for _ in range(_CAS_RETRY):
            account = await self.balance_account_repository.get_or_create(db, user_id)
            if account.balance < amount:
                raise BusinessException(ResultCode.BALANCE_INSUFFICIENT)
            if await self.balance_account_repository.refund(
                db, user_id, -amount, account.version
            ):
                await self.balance_log_repository.create_log(
                    db,
                    user_id=user_id,
                    change_type="refund",
                    amount=-amount,
                    balance_after=account.balance - amount,
                )
                return
        raise BusinessException(ResultCode.BUSINESS_ERROR, "余额退款扣减失败，请重试")


balance_account_service = BalanceAccountService()
