"""余额账户管理服务：Redis 原子 + MySQL 乐观锁 CAS

配额与余额双控中的"余额"一侧：可用积分池（财务资产），预扣/实扣/回补。
Redis 为准实时权威，MySQL（sys_user.credits_balance + credits_version）为持久化权威，
二者最终一致；MySQL 落地采用 version 字段乐观锁 CAS，失败重试 3 次。

Redis key（见后端实现 §7 缓存策略）：
- ai:balance:{user_id}    （积分余额，TTL 1 小时）
- ai:arrears:{user_id}    （欠费标记，无 TTL 持久）

db 为事务资源经参数显式传递；redis 经 get_redis_client() 自取。
"""

import logging
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.redis import get_redis_client
from app.repository.user_repository import user_repository

logger = logging.getLogger(__name__)

_CAS_RETRY = 3
_BALANCE_TTL = 3600  # ai:balance 缓存 1 小时
_BALANCE_KEY = "ai:balance:{user_id}"
_ARREARS_KEY = "ai:arrears:{user_id}"


class BalanceService:
    """余额账户管理：Redis 原子 + MySQL 乐观锁 CAS"""

    @staticmethod
    async def _increase_cas(db: AsyncSession, user_id: int, amount: Decimal) -> None:
        """余额增加落库：credits_balance += amount，乐观锁 CAS 失败重试 3 次"""
        for _ in range(_CAS_RETRY):
            current = await user_repository.get_credits_balance_and_version(db, user_id)
            if current is None:
                return
            _, version = current
            if await user_repository.increase_balance_cas(db, user_id, amount, version):
                return
        raise RuntimeError(f"余额增加落库失败（CAS 重试耗尽）: user_id={user_id}")

    @staticmethod
    async def _deduct_cas(db: AsyncSession, user_id: int, amount: Decimal) -> None:
        """余额扣减落库：credits_balance -= amount，乐观锁 CAS 失败重试 3 次"""
        for _ in range(_CAS_RETRY):
            current = await user_repository.get_credits_balance_and_version(db, user_id)
            if current is None:
                return
            _, version = current
            if await user_repository.deduct_balance_cas(db, user_id, amount, version):
                return
        raise RuntimeError(f"余额扣减落库失败（CAS 重试耗尽）: user_id={user_id}")

    @staticmethod
    async def get_balance(db: AsyncSession, user_id: int) -> Decimal:
        """查询余额：Redis 优先，未命中查 MySQL 并回填"""
        redis = await get_redis_client()
        val = await redis.get(_BALANCE_KEY.format(user_id=user_id))
        if val is not None:
            return Decimal(val)
        current = await user_repository.get_credits_balance_and_version(db, user_id)
        balance = current[0] if current else Decimal(0)
        await redis.setex(_BALANCE_KEY.format(user_id=user_id), _BALANCE_TTL, str(balance))
        return balance

    @staticmethod
    async def check_balance(db: AsyncSession, user_id: int, estimated_credits: int) -> bool:
        """余额校验：余额 >= 预估积分"""
        balance = await BalanceService.get_balance(db, user_id)
        return balance >= Decimal(estimated_credits)

    @staticmethod
    async def pre_deduct(db: AsyncSession, user_id: int, credits: int) -> bool:
        """余额预扣：Redis DECRBY，返回负数则回滚 INCRBY

        Returns:
            True 表示预扣成功；False 表示余额不足（已回滚，无副作用）
        """
        redis = await get_redis_client()
        # 确保 Redis 已缓存余额（TTL 过期后首次预扣需要从 MySQL 回填）
        await BalanceService.get_balance(db, user_id)
        balance = await redis.decrby(_BALANCE_KEY.format(user_id=user_id), credits)
        if balance < 0:
            await redis.incrby(_BALANCE_KEY.format(user_id=user_id), credits)
            return False
        return True

    @staticmethod
    async def refund(db: AsyncSession, user_id: int, credits: int) -> None:
        """退还余额（预扣与实际差额，多扣场景）"""
        redis = await get_redis_client()
        await redis.incrby(_BALANCE_KEY.format(user_id=user_id), credits)
        await BalanceService._increase_cas(db, user_id, Decimal(credits))

    @staticmethod
    async def deduct(db: AsyncSession, user_id: int, credits: int) -> None:
        """实扣减（少扣场景额外扣减），余额不足扣至 0 并标记欠费"""
        redis = await get_redis_client()
        balance = await redis.decrby(_BALANCE_KEY.format(user_id=user_id), credits)
        if balance < 0:
            # 扣至 0，差额记为欠费
            overdrawn = -balance
            await redis.set(_BALANCE_KEY.format(user_id=user_id), 0)
            await redis.set(_ARREARS_KEY.format(user_id=user_id), 1)
            await BalanceService._deduct_cas(db, user_id, Decimal(credits - overdrawn))
        else:
            await BalanceService._deduct_cas(db, user_id, Decimal(credits))

    @staticmethod
    async def increase(
        db: AsyncSession,
        user_id: int,
        amount: int,
        source: str,
        related_id: int | None = None,
        reason: str | None = None,
        operator_id: int | None = None,
    ) -> Decimal:
        """增加余额（充值/赠送/退款）

        流程：Redis INCRBY -> MySQL CAS 落库 -> 清除欠费标记 -> 写入流水

        Returns:
            变动后余额（balance_after）
        """
        # 统一为 Decimal 运算：Redis 用整型、落库/流水用 Decimal（避免 float/Decimal 混用）
        amount_dec = Decimal(amount)
        amount_int = int(amount_dec)

        redis = await get_redis_client()
        await redis.incrby(_BALANCE_KEY.format(user_id=user_id), amount_int)
        await BalanceService._increase_cas(db, user_id, amount_dec)
        await redis.delete(_ARREARS_KEY.format(user_id=user_id))
        from app.repository.ai_credit_log_repository import ai_credit_log_repository

        balance = await BalanceService.get_balance(db, user_id)
        await ai_credit_log_repository.create_log(
            db,
            user_id=user_id,
            source=source,
            amount=amount_dec,
            balance_after=balance,
            related_id=related_id,
            reason=reason,
            operator_id=operator_id,
        )
        return balance

    @staticmethod
    async def is_arrears(user_id: int) -> bool:
        """检查欠费状态：Redis GET ai:arrears:{user_id}"""
        redis = await get_redis_client()
        return await redis.get(_ARREARS_KEY.format(user_id=user_id)) is not None


balance_service = BalanceService()
