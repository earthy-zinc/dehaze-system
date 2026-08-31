"""成长值/签到域：每日签到、签到日历、成长值变动明细、使用行为激励。"""

import logging
from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.repository.member_sign_in_repository import member_sign_in_repository
from app.service.dict_service import get_dict_int
from app.service.member.member_service import (
    _check_and_adjust_level,
    _format_date,
    _format_dt,
    _invalidate_member_cache,
)

logger = logging.getLogger(__name__)

# 连续签到额外奖励的触发间隔（固定每 7 天，非运营可调参数）
SIGN_IN_BONUS_INTERVAL = 7

# 会员成长值规则默认值（与 config/sql/data/sys_dict.sql 的 member_growth_rules 种子一致，缺键回退）
SIGN_IN_BASE_GROWTH_DEFAULT = 3
SIGN_IN_BONUS_GROWTH_DEFAULT = 20

# 使用行为激励：change_type -> (单次成长值, 每日上限, 流水原因)
# 口径见需求规格 §2.3.1：图像处理/AI 对话每日各 10 次，效果评估每日 5 次
BEHAVIOR_GROWTH_RULES = {
    "process": (1, 10, "图像处理激励"),
    "evaluate": (1, 5, "效果评估激励"),
    "ai_consume": (1, 10, "AI 对话激励"),
}


class MemberGrowthService:
    def __init__(
        self,
        member_repository=member_repository,
        member_sign_in_repository=member_sign_in_repository,
        member_growth_log_repository=member_growth_log_repository,
        member_benefit_repository=member_benefit_repository,
    ):
        self.member_repository = member_repository
        self.member_sign_in_repository = member_sign_in_repository
        self.member_growth_log_repository = member_growth_log_repository
        self.member_benefit_repository = member_benefit_repository

    async def sign_in(self, db: AsyncSession, user_id: int) -> dict:
        member = await self.member_repository.get_or_init_member(db, user_id)

        today = date.today()
        existing = await self.member_sign_in_repository.get_by_user_and_date(db, user_id, today)
        if existing:
            raise BusinessException(ResultCode.SIGN_IN_ALREADY)

        yesterday = today - timedelta(days=1)
        yesterday_continuous = await self.member_sign_in_repository.get_latest_continuous_days(
            db, user_id, yesterday
        )
        continuous_days = yesterday_continuous + 1 if yesterday_continuous else 1

        base_growth = await get_dict_int(
            db, "member_growth_rules", "sign_in_value", SIGN_IN_BASE_GROWTH_DEFAULT
        )
        bonus_value = await get_dict_int(
            db, "member_growth_rules", "sign_in_streak_bonus", SIGN_IN_BONUS_GROWTH_DEFAULT
        )
        bonus_growth = bonus_value if continuous_days % SIGN_IN_BONUS_INTERVAL == 0 else 0
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

        await self.member_growth_log_repository.create_log(
            db,
            user_id=user_id,
            change_type="sign_in",
            change_value=base_growth,
            balance=old_growth + base_growth,
            related_id=str(sign_in_record.id),
        )

        if bonus_growth > 0:
            await self.member_growth_log_repository.create_log(
                db,
                user_id=user_id,
                change_type="sign_in_bonus",
                change_value=bonus_growth,
                balance=new_growth,
                related_id=str(sign_in_record.id),
            )

        old_level = member.level_code
        await _check_and_adjust_level(db, member, self.member_benefit_repository)
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

    async def get_sign_in_calendar(self, db: AsyncSession, user_id: int, year: int, month: int) -> dict:
        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        records = await self.member_sign_in_repository.get_by_user_and_date_range(
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

    async def list_growth_logs(self, db: AsyncSession, user_id: int, query: dict) -> dict:
        items, total = await self.member_growth_log_repository.get_page(
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

    async def add_behavior_growth(
        self, db: AsyncSession, user_id: int, change_type: str, related_id: str | None = None
    ) -> bool:
        """使用行为激励（process / evaluate / ai_consume）：按行为类型累计成长值，每日上限由 Redis 计数控制。

        Returns:
            True 表示已累计成长值；False 表示当日已达该行为激励上限
        """
        growth, daily_limit, reason = BEHAVIOR_GROWTH_RULES[change_type]
        count_key = f"member:growth:{change_type}:{user_id}:{date.today():%Y-%m-%d}"

        async def _incr():
            redis = await get_redis_client()
            count = await redis.incr(count_key)
            if count == 1:
                # 当日首次，设置 TTL 至次日 0 点后
                await redis.expire(count_key, 86400)
            return count

        count = await redis_operation_with_fallback(
            _incr, default=daily_limit, operation_name=f"{change_type}_growth_counter"
        )
        if count > daily_limit:
            return False

        member = await self.member_repository.get_or_init_member(db, user_id)
        new_growth = member.growth_value + growth
        member.growth_value = new_growth
        await db.flush()

        await self.member_growth_log_repository.create_log(
            db,
            user_id=user_id,
            change_type=change_type,
            change_value=growth,
            balance=new_growth,
            related_id=related_id,
            reason=reason,
        )

        old_level = member.level_code
        await _check_and_adjust_level(db, member, self.member_benefit_repository)
        await _invalidate_member_cache(user_id=user_id)
        if member.level_code != old_level:
            await _invalidate_member_cache(level_code=old_level)
            await _invalidate_member_cache(level_code=member.level_code)

        return True


member_growth_service = MemberGrowthService()
