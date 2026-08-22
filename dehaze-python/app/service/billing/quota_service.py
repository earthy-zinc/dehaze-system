"""配额管理服务：日/月限额，Redis 原子操作预扣/实扣/退还

配额与余额双控中的"限额"一侧：仅计数已用量并做阈值判断，不扣余额。
日限额每日 0 点（Asia/Shanghai）重置，月限额每月 1 日重置。

Redis key（见后端实现 §7 缓存策略）：
- ai:quota:daily:{user_id}:{date}   （date: YYYY-MM-DD，TTL 至当日 0 点）
- ai:quota:monthly:{user_id}:{month}（month: YYYY-MM，TTL 至当月 1 日）

db 为事务资源经参数显式传递；redis 经 get_redis_client() 自取。
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.redis import get_redis_client
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository

_TZ = ZoneInfo("Asia/Shanghai")

# 预扣：日/月"已用"各 INCRBY ARGV[3]，任一超出限额则整体回滚返回 0（不足）。
# 限额经 ARGV 传入（0 表示不限额）；key 语义 = 已用值（与 get_used/check_quota 一致）。
_PRE_DEDUCT_LUA = """
local limit_daily = tonumber(ARGV[4])
local limit_monthly = tonumber(ARGV[5])
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], 0, 'EX', ARGV[1])
end
local daily = redis.call('INCRBY', KEYS[1], ARGV[3])
if limit_daily > 0 and daily > limit_daily then
    redis.call('DECRBY', KEYS[1], ARGV[3])
    return 0
end
if redis.call('EXISTS', KEYS[2]) == 0 then
    redis.call('SET', KEYS[2], 0, 'EX', ARGV[2])
end
local monthly = redis.call('INCRBY', KEYS[2], ARGV[3])
if limit_monthly > 0 and monthly > limit_monthly then
    redis.call('DECRBY', KEYS[1], ARGV[3])
    redis.call('DECRBY', KEYS[2], ARGV[3])
    return 0
end
return 1
"""

# 退还（多扣差额）：日/月"已用"各 DECRBY ARGV[3]
_REFUND_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], 0, 'EX', ARGV[1])
end
redis.call('DECRBY', KEYS[1], ARGV[3])
if redis.call('EXISTS', KEYS[2]) == 0 then
    redis.call('SET', KEYS[2], 0, 'EX', ARGV[2])
end
redis.call('DECRBY', KEYS[2], ARGV[3])
return 1
"""

# 实扣（少扣差额，不回滚）：日/月"已用"各 INCRBY ARGV[3]
_DEDUCT_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], 0, 'EX', ARGV[1])
end
redis.call('INCRBY', KEYS[1], ARGV[3])
if redis.call('EXISTS', KEYS[2]) == 0 then
    redis.call('SET', KEYS[2], 0, 'EX', ARGV[2])
end
redis.call('INCRBY', KEYS[2], ARGV[3])
return 1
"""


def _quota_keys_and_ttl(user_id: int) -> tuple[str, str, int, int]:
    """计算日/月配额 key 及到期 TTL（Asia/Shanghai）"""
    now = datetime.now(_TZ)
    daily_key = f"ai:quota:daily:{user_id}:{now:%Y-%m-%d}"
    monthly_key = f"ai:quota:monthly:{user_id}:{now:%Y-%m}"
    next_day = datetime(now.year, now.month, now.day, tzinfo=_TZ) + timedelta(days=1)
    daily_ttl = int((next_day - now).total_seconds())
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1, tzinfo=_TZ)
    else:
        next_month = datetime(now.year, now.month + 1, 1, tzinfo=_TZ)
    monthly_ttl = int((next_month - now).total_seconds())
    return daily_key, monthly_key, daily_ttl, monthly_ttl


async def _apply_quota(user_id: int, credits: int, script: str) -> None:
    """以 Lua 脚本原子操作日/月配额（INCRBY 或 DECRBY credits）"""
    redis = await get_redis_client()
    daily_key, monthly_key, daily_ttl, monthly_ttl = _quota_keys_and_ttl(user_id)
    await redis.eval(script, 2, daily_key, monthly_key, daily_ttl, monthly_ttl, credits)


class QuotaService:
    """配额管理：日/月限额，Redis 原子操作预扣/实扣/退还"""

    @staticmethod
    async def get_limits(db: AsyncSession, user_id: int) -> tuple[int, int]:
        """查询用户日/月限额（从 sys_member_benefit 按 VIP 等级查询）

        仅读取启用（status=1）的权益配置，停用/未配置等级按无限额处理。

        Returns:
            (日限额, 月限额)，无会员、权益未配置或已停用时返回 (0, 0)
        """
        member = await member_repository.get_by_user_id(db, user_id)
        if member is None:
            return 0, 0
        benefit = await member_benefit_repository.get_by_level_code(db, member.level_code)
        if benefit is None or benefit.status != 1:
            return 0, 0
        return benefit.ai_credits_daily or 0, benefit.ai_credits_monthly or 0

    @staticmethod
    async def get_used(user_id: int) -> tuple[int, int]:
        """查询日/月已用配额，Redis 不存在时返回 0"""
        redis = await get_redis_client()
        daily_key, monthly_key, _, _ = _quota_keys_and_ttl(user_id)
        daily_val = await redis.get(daily_key)
        monthly_val = await redis.get(monthly_key)
        return int(daily_val or 0), int(monthly_val or 0)

    @staticmethod
    async def check_quota(
        db: AsyncSession,
        user_id: int,
        estimated_credits: int,
    ) -> bool:
        """预校验：日已用 + 预估 <= 日限额 且 月已用 + 预估 <= 月限额

        限额为 0 视为无限额（与 get_limits 语义、pre_deduct Lua 对齐）。

        Returns:
            True 表示配额充足；False 表示超限
        """
        daily_used, monthly_used = await QuotaService.get_used(user_id)
        daily_limit, monthly_limit = await QuotaService.get_limits(db, user_id)
        daily_ok = daily_limit == 0 or daily_used + estimated_credits <= daily_limit
        monthly_ok = monthly_limit == 0 or monthly_used + estimated_credits <= monthly_limit
        return daily_ok and monthly_ok

    @staticmethod
    async def pre_deduct(
        db: AsyncSession,
        user_id: int,
        credits: int,
    ) -> bool:
        """预扣减：Redis 原子 INCRBY 日/月"已用"，任一超限整体回滚

        Returns:
            True 表示预扣成功；False 表示配额不足（已整体回滚，无副作用）
        """
        redis = await get_redis_client()
        daily_limit, monthly_limit = await QuotaService.get_limits(db, user_id)
        daily_key, monthly_key, daily_ttl, monthly_ttl = _quota_keys_and_ttl(user_id)
        ok = await redis.eval(
            _PRE_DEDUCT_LUA,
            2,
            daily_key,
            monthly_key,
            daily_ttl,
            monthly_ttl,
            credits,
            daily_limit,
            monthly_limit,
        )
        return bool(ok)

    @staticmethod
    async def refund(user_id: int, credits: int) -> None:
        """退还配额（预扣与实际差额，多扣场景）"""
        await _apply_quota(user_id, credits, _REFUND_LUA)

    @staticmethod
    async def deduct(user_id: int, credits: int) -> None:
        """实扣减（少扣场景额外扣减，不回滚，超限扣至负数）"""
        await _apply_quota(user_id, credits, _DEDUCT_LUA)


quota_service = QuotaService()
