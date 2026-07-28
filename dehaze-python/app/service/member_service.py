from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.repository.member_sign_in_repository import member_sign_in_repository

SIGN_IN_BASE_GROWTH = 3
SIGN_IN_BONUS_GROWTH = 20
SIGN_IN_BONUS_INTERVAL = 7

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
    benefits = await member_benefit_repository.list_ordered_by_growth_min(db)
    target_level = _calculate_level(benefits, member.growth_value)
    if target_level != member.level_code:
        member.level_code = target_level
        member.level_source = "growth"
        await db.flush()


class MemberService:

    @staticmethod
    async def get_profile(db: AsyncSession, user_id: int) -> dict:
        data = await member_repository.get_with_user(db, user_id)
        if not data:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会员不存在")

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
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会员不存在")

        today = date.today()
        existing = await member_sign_in_repository.get_by_user_and_date(db, user_id, today)
        if existing:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "SIGN_IN_ALREADY")

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

        await _check_and_adjust_level(db, member)

        return {
            "signDate": _format_date(today),
            "continuousDays": continuous_days,
            "growthValue": total_growth,
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
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会员不存在")

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
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会员不存在")

        member.level_code = form["levelCode"]
        member.level_source = "admin"

        expire_time = form.get("expireTime")
        if expire_time:
            member.expire_time = _parse_dt(expire_time)

        if member.become_member_time is None:
            member.become_member_time = datetime.now()

        await db.flush()

    @staticmethod
    async def adjust_growth(db: AsyncSession, user_id: int, form: dict, operator_id: int) -> None:
        if not form.get("reason"):
            raise BusinessException(ResultCode.PARAM_ERROR, "调整原因必填")

        change_value = form["changeValue"]
        if change_value == 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "变动值不能为0")

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会员不存在")

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

        await _check_and_adjust_level(db, member)

    @staticmethod
    async def update_status(db: AsyncSession, user_id: int, form: dict) -> None:
        status = form["status"]
        reason = form.get("reason")

        if status == 0 and not reason:
            raise BusinessException(ResultCode.PARAM_ERROR, "冻结原因必填")

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会员不存在")

        if status == 0:
            member.status = 0
            member.frozen_reason = reason
            member.frozen_time = datetime.now()
        else:
            member.status = 1

        await db.flush()

    @staticmethod
    async def list_benefits(db: AsyncSession) -> list[dict]:
        benefits = await member_benefit_repository.list_all(db)
        return [_benefit_to_vo(b) for b in benefits]

    @staticmethod
    async def update_benefit(db: AsyncSession, level_code: str, form: dict) -> None:
        benefit = await member_benefit_repository.get_by_level_code(db, level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        for camel_key, snake_key in BENEFIT_FIELD_MAP.items():
            if camel_key in form and form[camel_key] is not None:
                setattr(benefit, snake_key, form[camel_key])

        await db.flush()
