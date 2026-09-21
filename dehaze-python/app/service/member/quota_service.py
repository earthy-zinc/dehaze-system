"""配额域：8 类任务权益校验 + Redis 原子扣减/归还 + 月度重置（归档历史 + 冻结顺延）。"""

import logging
from collections.abc import Awaitable
from datetime import datetime
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.entity.sys_member import QUOTA_TASK_TYPES, SysMember
from app.models.entity.sys_member_quota import SysMemberQuota
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.repository.package_repository import package_repository

logger = logging.getLogger(__name__)

_QUOTA_DEDUCT_LUA = """
local key = KEYS[1]
local expected = tonumber(ARGV[1])
local remaining = redis.call('get', key)
if not remaining then
    return nil
end
local val = tonumber(remaining)
if val > expected then
    -- 缓存剩余与生效配额漂移（如等级配额上调）时校正，避免长期误拦
    redis.call('set', key, expected)
    val = expected
end
if val <= 0 then
    return -1
end
return redis.call('decr', key)
"""

_QUOTA_RESTORE_LUA = """
if redis.call('exists', KEYS[1]) == 1 then
    return redis.call('incr', KEYS[1])
end
return 0
"""


def _quota_key(user_id: int, quota_type: str) -> str:
    return f"member:quota:{user_id}:{quota_type}"


def _quota_ttl_seconds() -> int:
    now = datetime.now()
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime(now.year, now.month + 1, 1)
    return max(1, int((next_month - now).total_seconds()))


def _effective_task_quota(benefit, overrides: dict | None) -> dict[str, int]:
    """合并等级权益与会员卡覆盖项，得到各任务的生效配额（覆盖与权益取较高值）。"""
    result = {}
    for task_type in QUOTA_TASK_TYPES:
        base = getattr(benefit, f"monthly_{task_type}_quota", 0) or 0
        if overrides:
            # overrides 的 key 为 camelCase（套餐侧写入/DB/Java 序列化统一口径，
            # 如 monthlySuperResolutionQuota）
            camel = "".join(part.capitalize() for part in task_type.split("_"))
            base = max(base, int(overrides.get(f"monthly{camel}Quota", 0) or 0))
        result[task_type] = base
    return result


def resolve_card_overrides(member: SysMember, package) -> dict | None:
    """已购会员卡（level_source=purchase 且未到期）生效的覆盖项，其余回退等级权益。"""
    if member.level_source != "purchase":
        return None
    if member.expire_time is not None and member.expire_time < datetime.now():
        return None
    if package is None or not package.benefit_overrides:
        return None
    return package.benefit_overrides


class MemberQuotaService:
    def __init__(
        self,
        member_repository=member_repository,
        member_benefit_repository=member_benefit_repository,
        package_repository=package_repository,
    ):
        self.member_repository = member_repository
        self.member_benefit_repository = member_benefit_repository
        self.package_repository = package_repository

    async def check_and_deduct_quota(self, db: AsyncSession, user_id: int, quota_type: str) -> None:
        """权益校验 + Redis 原子扣减 + 落库

        Args:
            db: 数据库会话
            user_id: 用户ID
            quota_type: 8 类任务之一
            （dehaze/derain/desnow/lowlight/super_resolution/denoise/inpaint/evaluate）

        Raises:
            BusinessException: 会员不存在/已冻结/次数用完
        """
        if quota_type not in QUOTA_TASK_TYPES:
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的配额类型: {quota_type}")

        member = await self.member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)
        if member.status != 1:
            raise BusinessException(ResultCode.MEMBER_FROZEN)

        used_field = f"monthly_{quota_type}_used"
        used = getattr(member, used_field, 0) or 0

        # 生效配额：已购会员卡取覆盖值与等级权益较高值，无覆盖则用等级权益
        benefit = await self.member_benefit_repository.get_by_level_code(db, member.level_code)
        if benefit:
            package = await self.package_repository.get_by_level_code(db, member.level_code)
            quota = _effective_task_quota(benefit, resolve_card_overrides(member, package)).get(
                quota_type, 0
            )
        else:
            quota = 0

        if used >= quota:
            raise BusinessException(ResultCode.QUOTA_EXCEEDED)

        cache_key = _quota_key(user_id, quota_type)
        ttl = _quota_ttl_seconds()

        async def _deduct_via_redis():
            redis = await get_redis_client()
            # redis-py 同步/异步客户端共用 eval 签名（Union[Awaitable[str], str]）；
            # 此处为异步客户端，运行时恒返回协程，await 正确，用 cast 收窄类型；
            # Lua 内 tonumber 解析数值，数值实参按字符串传入
            return await cast(
                Awaitable[str], redis.eval(_QUOTA_DEDUCT_LUA, 1, cache_key, str(quota - used))
            )

        result = await redis_operation_with_fallback(
            _deduct_via_redis, default=None, operation_name=f"quota_deduct:{quota_type}"
        )

        if result == -1:
            raise BusinessException(ResultCode.QUOTA_EXCEEDED)

        if result is None:
            # Redis 未命中，直接落库；条件更新防超扣（已用 < 生效配额）
            updated = await member_repository.increase_used_conditional(
                db, user_id, quota_type, quota
            )
            if not updated:
                raise BusinessException(ResultCode.QUOTA_EXCEEDED)

            async def _init_cache():
                redis = await get_redis_client()
                await redis.setex(cache_key, ttl, max(0, quota - used - 1))

            await redis_operation_with_fallback(
                _init_cache, default=None, operation_name=f"quota_cache_init:{quota_type}"
            )
            return

        # Redis 命中且扣减成功，落库条件更新防超扣；落库失败回补 Redis 计数保持两侧一致
        updated = await member_repository.increase_used_conditional(db, user_id, quota_type, quota)
        if not updated:

            async def _compensate():
                redis = await get_redis_client()
                await redis.incr(cache_key)

            await redis_operation_with_fallback(
                _compensate, default=None, operation_name=f"quota_deduct_compensate:{quota_type}"
            )
            raise BusinessException(ResultCode.QUOTA_EXCEEDED)

    async def restore_quota(self, db: AsyncSession, user_id: int, quota_type: str) -> None:
        """归还配额（任务失败时调用），8 类任务通用"""
        if quota_type not in QUOTA_TASK_TYPES:
            return

        member = await self.member_repository.get_by_user_id(db, user_id)
        if not member:
            return

        used_field = f"monthly_{quota_type}_used"
        used = getattr(member, used_field, 0) or 0
        if used > 0:
            setattr(member, used_field, used - 1)
            await db.flush()

        cache_key = _quota_key(user_id, quota_type)

        async def _incr():
            redis = await get_redis_client()
            # 键不存在时不创建（incr 会生成无 TTL 的持久键，污染后续扣减）
            # redis-py 同步/异步共用 eval 签名 → 异步客户端恒返回协程
            await cast(Awaitable[str], redis.eval(_QUOTA_RESTORE_LUA, 1, cache_key))

        await redis_operation_with_fallback(
            _incr, default=None, operation_name=f"quota_restore:{quota_type}"
        )

    async def refresh_member_quota(self, db: AsyncSession, member: SysMember, benefit) -> None:
        """会员卡履约/等级联动时刷新会员 8 类任务配额（不含已用量）"""
        for task_type in QUOTA_TASK_TYPES:
            setattr(
                member, f"monthly_{task_type}_quota", getattr(benefit, f"monthly_{task_type}_quota")
            )
        await db.flush()

    async def reset_monthly_quota(self, db: AsyncSession) -> int:
        """月度配额重置：归档上月使用情况 -> 按当前等级（含会员卡覆盖）刷新配额 -> 清零已用

        幂等由 quota_reset_month 条件保证；冻结中的会员跳过（解冻时顺延重置时点）；
        分批处理（每批 500），单条失败不影响其他。

        Returns:
            已重置的会员数量
        """
        now = datetime.now()
        current_month = int(now.strftime("%Y%m"))

        benefits = await self.member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}

        batch_size = 500
        total_count = 0

        while True:
            stmt = (
                select(SysMember)
                .where(
                    SysMember.deleted == 0,
                    SysMember.status == 1,
                    (SysMember.quota_reset_month.is_(None))
                    | (SysMember.quota_reset_month != current_month),
                )
                .limit(batch_size)
            )
            result = await db.execute(stmt)
            members = result.scalars().all()

            if not members:
                break

            for member in members:
                try:
                    # 归档上月使用情况（历史表，quota_month 为上月 yyyyMM）
                    if member.quota_reset_month is not None:
                        db.add(
                            SysMemberQuota(
                                user_id=member.user_id,
                                quota_month=member.quota_reset_month,
                                level_code=member.level_code,
                                **{
                                    f"{t}_quota": getattr(member, f"monthly_{t}_quota")
                                    for t in QUOTA_TASK_TYPES
                                },
                                **{
                                    f"{t}_used": getattr(member, f"monthly_{t}_used")
                                    for t in QUOTA_TASK_TYPES
                                },
                                reset_time=now,
                            )
                        )

                    benefit = benefit_map.get(member.level_code)
                    if benefit:
                        package = await self.package_repository.get_by_level_code(
                            db, member.level_code
                        )
                        effective = _effective_task_quota(
                            benefit, resolve_card_overrides(member, package)
                        )
                        for task_type in QUOTA_TASK_TYPES:
                            setattr(member, f"monthly_{task_type}_quota", effective[task_type])
                    for task_type in QUOTA_TASK_TYPES:
                        setattr(member, f"monthly_{task_type}_used", 0)
                    member.quota_reset_month = current_month
                    total_count += 1
                except Exception:
                    logger.warning(
                        "月度配额重置失败，跳过: user_id=%s", member.user_id, exc_info=True
                    )

            await db.flush()

            async def _invalidate_quota_cache(batch_members=members):
                redis = await get_redis_client()
                keys = [
                    f"member:quota:{m.user_id}:{t}" for m in batch_members for t in QUOTA_TASK_TYPES
                ]
                if keys:
                    await redis.delete(*keys)

            await redis_operation_with_fallback(
                _invalidate_quota_cache, default=None, operation_name="quota_reset_cache_invalidate"
            )

        return total_count


member_quota_service = MemberQuotaService()
