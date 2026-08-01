from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.base import get_current_user_id
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_quota import SysMemberQuota
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.repository.member_sign_in_repository import member_sign_in_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
import json
import logging

logger = logging.getLogger(__name__)

MEMBER_LEVEL_CACHE_TTL = 1800
MEMBER_BENEFIT_CACHE_TTL = 3600

SIGN_IN_BASE_GROWTH = 3
SIGN_IN_BONUS_GROWTH = 20
SIGN_IN_BONUS_INTERVAL = 7

_QUOTA_DEDUCT_LUA = """
local key = KEYS[1]
local remaining = redis.call('get', key)
if remaining then
    local val = tonumber(remaining)
    if val <= 0 then
        return -1
    end
    return redis.call('decr', key)
else
    return nil
end
"""

BENEFIT_FIELD_MAP = {
    "levelName": "level_name",
    "growthMin": "growth_min",
    "growthMax": "growth_max",
    "monthlyDehazeQuota": "monthly_dehaze_quota",
    "monthlyEvaluateQuota": "monthly_evaluate_quota",
    "historyRetention": "history_retention",
    "batchLimit": "batch_limit",
    "priority": "priority",
    "advancedParams": "advanced_params",
    "hdExport": "hd_export",
    "reportExport": "report_export",
    "batchDownload": "batch_download",
    "sort": "sort",
    "status": "status",
}


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_date(d: Optional[date]) -> Optional[str]:
    if d is None:
        return None
    return d.strftime("%Y-%m-%d")


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _benefit_to_vo(b) -> dict:
    return {
        "levelCode": b.level_code,
        "levelName": b.level_name,
        "growthMin": b.growth_min,
        "growthMax": b.growth_max,
        "monthlyDehazeQuota": b.monthly_dehaze_quota,
        "monthlyEvaluateQuota": b.monthly_evaluate_quota,
        "historyRetention": b.history_retention,
        "batchLimit": b.batch_limit,
        "priority": b.priority,
        "advancedParams": b.advanced_params,
        "hdExport": b.hd_export,
        "reportExport": b.report_export,
        "batchDownload": b.batch_download,
        "sort": b.sort,
        "status": b.status,
    }


def _calc_progress(benefits: list, level_code: str, growth_value: int) -> tuple[int, Optional[int]]:
    current = next((b for b in benefits if b.level_code == level_code), None)
    if not current:
        return 0, None

    next_benefit = next((b for b in benefits if b.growth_min > current.growth_min), None)

    if current.growth_max == 0:
        return 100, None

    if current.growth_max > current.growth_min:
        progress = int((growth_value - current.growth_min) / (current.growth_max - current.growth_min) * 100)
        progress = max(0, min(100, progress))
    else:
        progress = 100

    if next_benefit:
        next_level_growth = next_benefit.growth_min - growth_value
        if next_level_growth < 0:
            next_level_growth = 0
    else:
        next_level_growth = None

    return progress, next_level_growth


def _calculate_level(benefits: list, growth_value: int) -> str:
    for b in benefits:
        if b.growth_max == 0:
            if growth_value >= b.growth_min:
                return b.level_code
        elif b.growth_min <= growth_value <= b.growth_max:
            return b.level_code
    return benefits[0].level_code if benefits else "level_0"


async def _check_and_adjust_level(db: AsyncSession, member: SysMember) -> None:
    if member.level_source != "growth":
        return
    benefits = await member_benefit_repository.list_ordered_by_growth_min(db)
    target_level = _calculate_level(benefits, member.growth_value)
    if target_level != member.level_code:
        member.level_code = target_level
        benefit = next((b for b in benefits if b.level_code == target_level), None)
        if benefit:
            member.monthly_dehaze_quota = benefit.monthly_dehaze_quota
            member.monthly_evaluate_quota = benefit.monthly_evaluate_quota
        await db.flush()


async def _invalidate_member_cache(user_id: Optional[int] = None, level_code: Optional[str] = None) -> None:
    keys = []
    if user_id is not None:
        keys.append(f"member:level:{user_id}")
        keys.append(f"member:quota:{user_id}:dehaze")
        keys.append(f"member:quota:{user_id}:evaluate")
    if level_code is not None:
        keys.append(f"member:benefit:{level_code}")
    keys.append("member:benefit:all")
    if not keys:
        return

    async def _del():
        redis = await get_redis_client()
        await redis.delete(*keys)

    await redis_operation_with_fallback(_del, default=None, operation_name="member_cache_invalidate")


def _quota_key(user_id: int, quota_type: str) -> str:
    return f"member:quota:{user_id}:{quota_type}"


def _quota_ttl_seconds() -> int:
    now = datetime.now()
    if now.month == 12:
        next_month = datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime(now.year, now.month + 1, 1)
    return max(1, int((next_month - now).total_seconds()))


class MemberService:

    @staticmethod
    async def get_profile(db: AsyncSession, user_id: int) -> dict:
        data = await member_repository.get_with_user(db, user_id)
        if not data:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        member = data["member"]
        benefit = await member_benefit_repository.get_by_level_code(db, member.level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        benefits = await member_benefit_repository.list_all(db)
        progress_percent, next_level_growth = _calc_progress(benefits, member.level_code, member.growth_value)

        return {
            "userId": member.user_id,
            "username": data.get("username") or "",
            "nickname": data.get("nickname"),
            "avatar": data.get("avatar"),
            "levelCode": member.level_code,
            "levelName": benefit.level_name,
            "growthValue": member.growth_value,
            "nextLevelGrowth": next_level_growth,
            "progressPercent": progress_percent,
            "expireTime": _format_dt(member.expire_time),
            "monthlyDehazeQuota": member.monthly_dehaze_quota,
            "monthlyDehazeUsed": member.monthly_dehaze_used,
            "monthlyEvaluateQuota": member.monthly_evaluate_quota,
            "monthlyEvaluateUsed": member.monthly_evaluate_used,
            "benefits": _benefit_to_vo(benefit),
            "status": member.status,
        }

    @staticmethod
    async def list_growth_logs(db: AsyncSession, user_id: int, query: dict) -> dict:
        items, total = await member_growth_log_repository.get_page(
            db,
            user_id,
            query["pageNum"],
            query["pageSize"],
            change_type=query.get("changeType"),
            start_time=query.get("startTime"),
            end_time=query.get("endTime"),
        )

        list_data = [
            {
                "id": log.id,
                "changeType": log.change_type,
                "changeValue": log.change_value,
                "balance": log.balance,
                "relatedId": log.related_id,
                "reason": log.reason,
                "operatorId": log.operator_id,
                "createTime": _format_dt(log.create_time),
            }
            for log in items
        ]

        return {"list": list_data, "total": total}

    @staticmethod
    async def sign_in(db: AsyncSession, user_id: int) -> dict:
        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        today = date.today()
        existing = await member_sign_in_repository.get_by_user_and_date(db, user_id, today)
        if existing:
            raise BusinessException(ResultCode.SIGN_IN_ALREADY)

        yesterday = today - timedelta(days=1)
        yesterday_continuous = await member_sign_in_repository.get_latest_continuous_days(db, user_id, yesterday)
        continuous_days = yesterday_continuous + 1 if yesterday_continuous else 1

        base_growth = SIGN_IN_BASE_GROWTH
        bonus_growth = SIGN_IN_BONUS_GROWTH if continuous_days % SIGN_IN_BONUS_INTERVAL == 0 else 0
        total_growth = base_growth + bonus_growth

        sign_in_record = SysMemberSignIn(
            user_id=user_id,
            sign_date=today,
            continuous_days=continuous_days,
            growth_value=total_growth,
        )
        db.add(sign_in_record)
        await db.flush()

        old_growth = member.growth_value
        new_growth = old_growth + total_growth
        member.growth_value = new_growth
        await db.flush()

        await member_growth_log_repository.create_log(
            db,
            user_id=user_id,
            change_type="sign_in",
            change_value=base_growth,
            balance=old_growth + base_growth,
            related_id=str(sign_in_record.id),
        )

        if bonus_growth > 0:
            await member_growth_log_repository.create_log(
                db,
                user_id=user_id,
                change_type="sign_in_bonus",
                change_value=bonus_growth,
                balance=new_growth,
                related_id=str(sign_in_record.id),
            )

        old_level = member.level_code
        await _check_and_adjust_level(db, member)
        await _invalidate_member_cache(user_id=user_id)
        if member.level_code != old_level:
            await _invalidate_member_cache(level_code=old_level)
            await _invalidate_member_cache(level_code=member.level_code)

        return {
            "signDate": _format_date(today),
            "continuousDays": continuous_days,
            "growthValue": base_growth,
            "bonusGrowth": bonus_growth,
        }

    @staticmethod
    async def get_sign_in_calendar(db: AsyncSession, user_id: int, year: int, month: int) -> dict:
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        records = await member_sign_in_repository.get_by_user_and_date_range(
            db, user_id, start_date, end_date
        )

        sign_dates = [_format_date(r.sign_date) for r in records]
        total_days = len(records)
        continuous_days = records[-1].continuous_days if records else 0

        return {
            "signDates": sign_dates,
            "continuousDays": continuous_days,
            "totalDays": total_days,
        }

    @staticmethod
    async def list_paged_members(db: AsyncSession, query: dict) -> dict:
        items, total = await member_repository.get_page(
            db,
            query["pageNum"],
            query["pageSize"],
            keywords=query.get("keywords"),
            level_code=query.get("levelCode"),
            status=query.get("status"),
            expire_time_start=query.get("expireTimeStart"),
            expire_time_end=query.get("expireTimeEnd"),
            growth_min=query.get("growthMin"),
            growth_max=query.get("growthMax"),
        )

        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}

        list_data = []
        for item in items:
            member = item["member"]
            benefit = benefit_map.get(member.level_code)
            level_name = benefit.level_name if benefit else ""
            monthly_used = member.monthly_dehaze_used + member.monthly_evaluate_used
            list_data.append({
                "userId": member.user_id,
                "username": item.get("username") or "",
                "nickname": item.get("nickname"),
                "levelCode": member.level_code,
                "levelName": level_name,
                "growthValue": member.growth_value,
                "monthlyUsed": monthly_used,
                "expireTime": _format_dt(member.expire_time),
                "status": member.status,
                "becomeMemberTime": _format_dt(member.become_member_time),
            })

        return {"list": list_data, "total": total}

    @staticmethod
    async def get_member_detail(db: AsyncSession, user_id: int) -> dict:
        data = await member_repository.get_with_user(db, user_id)
        if not data:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        member = data["member"]
        benefit = await member_benefit_repository.get_by_level_code(db, member.level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        benefits = await member_benefit_repository.list_all(db)
        progress_percent, next_level_growth = _calc_progress(benefits, member.level_code, member.growth_value)

        profile = {
            "userId": member.user_id,
            "username": data.get("username") or "",
            "nickname": data.get("nickname"),
            "avatar": data.get("avatar"),
            "levelCode": member.level_code,
            "levelName": benefit.level_name,
            "growthValue": member.growth_value,
            "nextLevelGrowth": next_level_growth,
            "progressPercent": progress_percent,
            "expireTime": _format_dt(member.expire_time),
            "monthlyDehazeQuota": member.monthly_dehaze_quota,
            "monthlyDehazeUsed": member.monthly_dehaze_used,
            "monthlyEvaluateQuota": member.monthly_evaluate_quota,
            "monthlyEvaluateUsed": member.monthly_evaluate_used,
            "benefits": _benefit_to_vo(benefit),
            "status": member.status,
        }

        profile["levelSource"] = member.level_source
        profile["totalConsumption"] = member.total_consumption
        profile["becomeMemberTime"] = _format_dt(member.become_member_time)
        profile["frozenReason"] = member.frozen_reason
        profile["frozenTime"] = _format_dt(member.frozen_time)
        profile["quotaResetMonth"] = member.quota_reset_month

        return profile

    @staticmethod
    async def adjust_level(db: AsyncSession, user_id: int, form: dict, operator_id: int) -> None:
        if not form.get("reason"):
            raise BusinessException(ResultCode.PARAM_ERROR, "调整原因必填")

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        old_level = member.level_code
        member.level_code = form["levelCode"]
        member.level_source = "admin"

        expire_time = form.get("expireTime")
        if expire_time:
            member.expire_time = _parse_dt(expire_time)
        else:
            member.expire_time = None

        if member.become_member_time is None:
            member.become_member_time = datetime.now()

        benefit = await member_benefit_repository.get_by_level_code(db, form["levelCode"])
        if benefit:
            member.monthly_dehaze_quota = benefit.monthly_dehaze_quota
            member.monthly_evaluate_quota = benefit.monthly_evaluate_quota

        await db.flush()
        await _invalidate_member_cache(user_id=user_id, level_code=old_level)
        await _invalidate_member_cache(level_code=form["levelCode"])

        mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="member",
            target_id=user_id,
            action="level_change",
            module="member",
            before_value={"levelCode": old_level},
            after_value=form.dict() if hasattr(form, "dict") else form,
        )

    @staticmethod
    async def adjust_growth(db: AsyncSession, user_id: int, form: dict, operator_id: int) -> None:
        if not form.get("reason"):
            raise BusinessException(ResultCode.PARAM_ERROR, "调整原因必填")

        change_value = form["changeValue"]
        if change_value == 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "变动值不能为0")

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        new_growth = member.growth_value + change_value
        if new_growth < 0:
            new_growth = 0

        member.growth_value = new_growth
        await db.flush()

        await member_growth_log_repository.create_log(
            db,
            user_id=user_id,
            change_type="admin_adjust",
            change_value=change_value,
            balance=new_growth,
            reason=form["reason"],
            operator_id=operator_id,
        )

        old_level = member.level_code
        await _check_and_adjust_level(db, member)
        if member.level_code != old_level:
            await _invalidate_member_cache(user_id=user_id, level_code=old_level)
            await _invalidate_member_cache(level_code=member.level_code)

        mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="member",
            target_id=user_id,
            action="growth_change",
            module="member",
            after_value=form.dict() if hasattr(form, "dict") else form,
        )

    @staticmethod
    async def update_status(db: AsyncSession, user_id: int, form: dict) -> None:
        status = form["status"]
        reason = form.get("reason")

        if status == 0 and not reason:
            raise BusinessException(ResultCode.PARAM_ERROR, "冻结原因必填")

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        old_status = member.status
        if status == 0:
            member.status = 0
            member.frozen_reason = reason
            member.frozen_time = datetime.now()
        else:
            member.status = 1

        await db.flush()
        await _invalidate_member_cache(user_id=user_id)

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="member",
            target_id=user_id,
            action="status_change",
            module="member",
            before_value={"status": old_status},
            after_value=form.dict() if hasattr(form, "dict") else form,
        )

    @staticmethod
    async def list_benefits(db: AsyncSession) -> list[dict]:
        cache_key = "member:benefit:all"

        async def _get_cache():
            redis = await get_redis_client()
            return await redis.get(cache_key)

        cached_raw = await redis_operation_with_fallback(_get_cache, default=None, operation_name="member_benefit_list_cache_get")
        if cached_raw:
            try:
                return json.loads(cached_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        benefits = await member_benefit_repository.list_all(db)
        result = [_benefit_to_vo(b) for b in benefits]

        async def _set_cache():
            redis = await get_redis_client()
            await redis.setex(cache_key, MEMBER_BENEFIT_CACHE_TTL, json.dumps(result, ensure_ascii=False, default=str))
        await redis_operation_with_fallback(_set_cache, default=None, operation_name="member_benefit_list_cache_set")

        return result

    @staticmethod
    async def update_benefit(db: AsyncSession, level_code: str, form: dict) -> None:
        benefit = await member_benefit_repository.get_by_level_code(db, level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        for camel_key, snake_key in BENEFIT_FIELD_MAP.items():
            if camel_key in form and form[camel_key] is not None:
                setattr(benefit, snake_key, form[camel_key])

        if benefit.growth_max and benefit.growth_max > 0 and benefit.growth_min > benefit.growth_max:
            raise BusinessException(ResultCode.BENEFIT_CONFIG_INVALID, "成长值下限不能大于上限")

        await db.flush()
        await _invalidate_member_cache(level_code=level_code)

    @staticmethod
    async def check_and_deduct_quota(db: AsyncSession, user_id: int, quota_type: str) -> None:
        """权益校验 + Redis 原子扣减 + 异步落库

        Args:
            db: 数据库会话
            user_id: 用户ID
            quota_type: "dehaze" 或 "evaluate"

        Raises:
            BusinessException: 会员不存在/已冻结/次数用完
        """
        if quota_type not in ("dehaze", "evaluate"):
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的配额类型: {quota_type}")

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)
        if member.status != 1:
            raise BusinessException(ResultCode.MEMBER_FROZEN)

        quota_field = f"monthly_{quota_type}_quota"
        used_field = f"monthly_{quota_type}_used"
        quota = getattr(member, quota_field, 0) or 0
        used = getattr(member, used_field, 0) or 0
        remaining = quota - used

        if remaining <= 0:
            raise BusinessException(ResultCode.QUOTA_EXCEEDED)

        cache_key = _quota_key(user_id, quota_type)
        ttl = _quota_ttl_seconds()

        async def _deduct_via_redis():
            redis = await get_redis_client()
            result = await redis.eval(_QUOTA_DEDUCT_LUA, 1, cache_key)
            return result

        result = await redis_operation_with_fallback(_deduct_via_redis, default=None, operation_name=f"quota_deduct:{quota_type}")

        if result is None:
            setattr(member, used_field, used + 1)
            await db.flush()

            async def _init_cache():
                redis = await get_redis_client()
                await redis.setex(cache_key, ttl, max(0, remaining - 1))
            await redis_operation_with_fallback(_init_cache, default=None, operation_name=f"quota_cache_init:{quota_type}")

            mongo_audit_log_repository.create_audit_async(
                operator_id=get_current_user_id(),
                target_type="member",
                target_id=user_id,
                action="quota_deduct",
                module="member",
                after_value={"quota_type": quota_type, "amount": 1},
            )
            return

        if result == -1:
            raise BusinessException(ResultCode.QUOTA_EXCEEDED)

        setattr(member, used_field, used + 1)
        await db.flush()

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="member",
            target_id=user_id,
            action="quota_deduct",
            module="member",
            after_value={"quota_type": quota_type, "amount": 1},
        )

    @staticmethod
    async def restore_quota(db: AsyncSession, user_id: int, quota_type: str) -> None:
        """归还配额（任务失败时调用）"""
        if quota_type not in ("dehaze", "evaluate"):
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的配额类型: {quota_type}")

        member = await member_repository.get_by_user_id(db, user_id)
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
            await redis.incr(cache_key)
        await redis_operation_with_fallback(_incr, default=None, operation_name=f"quota_restore:{quota_type}")

    @staticmethod
    async def reset_monthly_quota(db: AsyncSession) -> int:
        """月度配额重置：按当前等级权益重置所有会员的当月配额

        Returns:
            已重置的会员数量
        """
        now = datetime.now()
        current_month = int(now.strftime("%Y%m"))

        benefits = await member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}

        batch_size = 500
        total_count = 0

        while True:
            stmt = (
                select(SysMember)
                .where(
                    SysMember.deleted == 0,
                    or_(
                        SysMember.quota_reset_month.is_(None),
                        SysMember.quota_reset_month != current_month,
                    ),
                )
                .limit(batch_size)
            )
            result = await db.execute(stmt)
            members = result.scalars().all()

            if not members:
                break

            for member in members:
                if member.quota_reset_month is not None:
                    archive = SysMemberQuota(
                        user_id=member.user_id,
                        quota_month=member.quota_reset_month,
                        level_code=member.level_code,
                        dehaze_quota=member.monthly_dehaze_quota,
                        dehaze_used=member.monthly_dehaze_used,
                        evaluate_quota=member.monthly_evaluate_quota,
                        evaluate_used=member.monthly_evaluate_used,
                        reset_time=now,
                    )
                    db.add(archive)

                benefit = benefit_map.get(member.level_code)
                if benefit:
                    member.monthly_dehaze_quota = benefit.monthly_dehaze_quota
                    member.monthly_evaluate_quota = benefit.monthly_evaluate_quota
                member.monthly_dehaze_used = 0
                member.monthly_evaluate_used = 0
                member.quota_reset_month = current_month
                total_count += 1

            await db.flush()

            async def _invalidate_quota_cache(batch_members=members):
                redis = await get_redis_client()
                keys = [f"member:quota:{m.user_id}:dehaze" for m in batch_members] + \
                       [f"member:quota:{m.user_id}:evaluate" for m in batch_members]
                if keys:
                    await redis.delete(*keys)
            await redis_operation_with_fallback(_invalidate_quota_cache, default=None, operation_name="quota_reset_cache_invalidate")

        return total_count

    @staticmethod
    async def process_expired_members(db: AsyncSession) -> int:
        """会员过期降级处理

        扫描 expire_time < NOW() AND level_source != 'growth' 的会员，
        按成长值重算等级、置 level_source='growth'、清空 expire_time、刷新权益。

        Returns:
            已处理的会员数量
        """
        now = datetime.now()
        stmt = select(SysMember).where(
            SysMember.deleted == 0,
            SysMember.expire_time.isnot(None),
            SysMember.expire_time < now,
            SysMember.level_source != "growth",
        )
        result = await db.execute(stmt)
        members = result.scalars().all()

        if not members:
            return 0

        benefits = await member_benefit_repository.list_ordered_by_growth_min(db)
        benefit_map = {b.level_code: b for b in benefits}

        count = 0
        for member in members:
            old_level = member.level_code
            target_level = _calculate_level(benefits, member.growth_value)
            member.level_code = target_level
            member.level_source = "growth"
            member.expire_time = None
            benefit = benefit_map.get(target_level)
            if benefit:
                member.monthly_dehaze_quota = benefit.monthly_dehaze_quota
                member.monthly_evaluate_quota = benefit.monthly_evaluate_quota
            count += 1
            await _invalidate_member_cache(user_id=member.user_id, level_code=old_level)
            await _invalidate_member_cache(level_code=target_level)

            if target_level != old_level:
                try:
                    from app.service.message_service import MessageService
                    old_benefit = benefit_map.get(old_level)
                    new_benefit = benefit_map.get(target_level)
                    await MessageService.send(db, {
                        "type": "member",
                        "recipientIds": [member.user_id],
                        "bizModule": "member",
                        "bizId": f"level_change:{member.user_id}:{int(now.timestamp())}",
                        "templateCode": "member_downgrade_warning",
                        "variables": {
                            "currentLevel": old_benefit.level_name if old_benefit else old_level,
                            "days": "0",
                            "downgradeLevel": new_benefit.level_name if new_benefit else target_level,
                        },
                    })
                except Exception as e:
                    logger.warning(f"等级变更通知发送失败: userId={member.user_id}, old={old_level}, new={target_level}", exc_info=e)

        await db.flush()
        logger.debug(f"会员过期降级处理完成: 共处理 {count} 条记录")
        return count

    @staticmethod
    async def send_expire_reminders(db: AsyncSession) -> int:
        from app.service.message_service import MessageService

        now = datetime.now()
        benefits = await member_benefit_repository.list_ordered_by_growth_min(db)
        benefit_map = {b.level_code: b for b in benefits}

        day_template_map = {
            7: ("expire_reminder_7d", "member_expire_reminder_7"),
            3: ("expire_reminder_3d", "member_expire_reminder_3"),
            1: ("expire_reminder_1d", "member_expire_reminder_1"),
        }

        sent_count = 0
        for days, (biz_prefix, template_code) in day_template_map.items():
            window_start = (now + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            window_end = window_start + timedelta(days=1)

            stmt = select(SysMember).where(
                SysMember.deleted == 0,
                SysMember.expire_time.isnot(None),
                SysMember.expire_time >= window_start,
                SysMember.expire_time < window_end,
                SysMember.level_source != "growth",
            )
            result = await db.execute(stmt)
            members = result.scalars().all()
            if not members:
                continue

            for member in members:
                try:
                    current_benefit = benefit_map.get(member.level_code)
                    variables = {
                        "currentLevel": current_benefit.level_name if current_benefit else member.level_code,
                        "days": str(days),
                        "expireDate": member.expire_time.strftime("%Y-%m-%d") if member.expire_time else "",
                    }
                    if days == 3:
                        target_level = _calculate_level(benefits, member.growth_value)
                        downgrade_benefit = benefit_map.get(target_level)
                        variables["downgradeLevel"] = downgrade_benefit.level_name if downgrade_benefit else target_level
                        if current_benefit and downgrade_benefit:
                            variables["benefitCompare"] = (
                                f"去雾:{current_benefit.monthly_dehaze_quota}→{downgrade_benefit.monthly_dehaze_quota}次/月，"
                                f"评估:{current_benefit.monthly_evaluate_quota}→{downgrade_benefit.monthly_evaluate_quota}次/月"
                            )
                        else:
                            variables["benefitCompare"] = ""

                    await MessageService.send(db, {
                        "type": "member",
                        "recipientIds": [member.user_id],
                        "bizModule": "member",
                        "bizId": f"{biz_prefix}:{member.user_id}:{now.strftime('%Y-%m-%d')}",
                        "templateCode": template_code,
                        "variables": variables,
                    })
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"到期提醒发送失败: userId={member.user_id}, days={days}", exc_info=e)

        logger.debug(f"会员到期预警完成: 共发送 {sent_count} 条提醒")
        return sent_count
