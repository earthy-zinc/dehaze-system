"""会员核心域：会员档案、管理员等级/成长值/状态调整、履约回调、权益概览与试用引导。

本模块同时承载会员域共享支撑（格式化、等级计算、等级联动调整、8 类任务配额刷新、缓存失效），
growth/benefit/quota/expiry 子域从此处引用。
"""

import logging
from datetime import date, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.models.base import get_current_user_id
from app.models.entity.sys_member import IMAGE_TASK_TYPES, QUOTA_TASK_TYPES, SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.repository.ai_credit_log_repository import ai_credit_log_repository
from app.repository.coupon_repository import user_coupon_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.order_repository import order_repository
from app.repository.package_repository import package_repository
from app.service.billing.balance_service import balance_service
from app.service.billing.quota_service import quota_service as billing_quota_service
from app.service.member.quota_service import member_quota_service

logger = logging.getLogger(__name__)

# 试用引导常量（体验券默认 3 天 / 100 AI 试用积分）
TRIAL_DEFAULT_DAYS = 3
TRIAL_DEFAULT_CREDITS = 100

# 会员卡续费叠加上限（3 年）
MEMBER_CARD_MAX_DAYS = 365 * 3


def _format_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_date(d: date | None) -> str | None:
    if d is None:
        return None
    return d.strftime("%Y-%m-%d")


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _benefit_to_vo(b) -> dict:
    vo = {
        "levelCode": b.level_code,
        "levelName": b.level_name,
        "growthMin": b.growth_min,
        "growthMax": b.growth_max,
        "aiCreditsDaily": b.ai_credits_daily,
        "aiCreditsMonthly": b.ai_credits_monthly,
        "multimodalLimit": b.multimodal_limit,
        "vipGiftCredits": b.vip_gift_credits,
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
    vo.update({f"monthly{_camel(t)}Quota": getattr(b, f"monthly_{t}_quota") for t in QUOTA_TASK_TYPES})
    return vo


def _camel(s: str) -> str:
    parts = s.split("_")
    return parts[0].title() + "".join(p.title() for p in parts[1:])


def _apply_benefit_quotas(member: SysMember, benefit: SysMemberBenefit) -> None:
    """按等级权益刷新会员 8 类任务配额（不含已用量，AI 限额由权益配置读取）。"""
    for task_type in QUOTA_TASK_TYPES:
        setattr(member, f"monthly_{task_type}_quota", getattr(benefit, f"monthly_{task_type}_quota"))


def _calc_progress(benefits: list, level_code: str, growth_value: int) -> tuple[int, int | None]:
    current = next((b for b in benefits if b.level_code == level_code), None)
    if not current:
        return 0, None

    next_benefit = next((b for b in benefits if b.growth_min > current.growth_min), None)

    if current.growth_max == 0:
        return 100, None

    if current.growth_max > current.growth_min:
        progress = int(
            (growth_value - current.growth_min) / (current.growth_max - current.growth_min) * 100
        )
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


def _can_downgrade(member: SysMember) -> bool:
    """自动降级约束：仅成长值来源或会员卡已到期的会员可降级。"""
    if member.level_source == "growth":
        return True
    return member.expire_time is not None and member.expire_time < datetime.now()


async def _check_and_adjust_level(
    db: AsyncSession, member: SysMember, member_benefit_repository
) -> None:
    """成长值变动后触发等级检查：升级不限来源；降级受来源与有效期约束。

    升级后 level_source 保持不变（保留会员卡/管理员来源与 expire_time），
    会员卡期间享有 max(会员卡等级, 成长值等级)。
    """
    benefits = await member_benefit_repository.list_ordered_by_growth_min(db)
    if not benefits:
        return
    target_level = _calculate_level(benefits, member.growth_value)
    if target_level == member.level_code:
        return

    current = next((b for b in benefits if b.level_code == member.level_code), None)
    target = next((b for b in benefits if b.level_code == target_level), None)

    if current is not None and target is not None and target.growth_min > current.growth_min:
        # 自动升级不限来源
        member.level_code = target_level
        _apply_benefit_quotas(member, target)
        await db.flush()
    elif _can_downgrade(member):
        # 降级：来源切 growth、清空到期时间、刷新配额（保级时等级不变不在此处理）
        member.level_code = target_level
        member.level_source = "growth"
        member.expire_time = None
        if target is not None:
            _apply_benefit_quotas(member, target)
        await db.flush()


async def _invalidate_member_cache(
    user_id: int | None = None, level_code: str | None = None
) -> None:
    keys = []
    if user_id is not None:
        keys.append(f"member:profile:{user_id}")
        keys.append(f"member:level:{user_id}")
        keys.append(f"member:benefit-summary:{user_id}")
        keys.extend(f"member:quota:{user_id}:{t}" for t in QUOTA_TASK_TYPES)
    if level_code is not None:
        keys.append(f"member:benefit:{level_code}")
    keys.append("member:benefit:all")
    if not keys:
        return

    async def _del():
        redis = await get_redis_client()
        await redis.delete(*keys)

    await redis_operation_with_fallback(
        _del, default=None, operation_name="member_cache_invalidate"
    )


class MemberService:
    def __init__(
        self,
        member_repository=member_repository,
        member_benefit_repository=member_benefit_repository,
        member_growth_log_repository=member_growth_log_repository,
        mongo_audit_log_repository=mongo_audit_log_repository,
        order_repository=order_repository,
        package_repository=package_repository,
        ai_credit_log_repository=ai_credit_log_repository,
        user_coupon_repository=user_coupon_repository,
        balance_service=balance_service,
        billing_quota_service=billing_quota_service,
        member_quota_service=member_quota_service,
    ):
        self.member_repository = member_repository
        self.member_benefit_repository = member_benefit_repository
        self.member_growth_log_repository = member_growth_log_repository
        self.mongo_audit_log_repository = mongo_audit_log_repository
        self.order_repository = order_repository
        self.package_repository = package_repository
        self.ai_credit_log_repository = ai_credit_log_repository
        self.user_coupon_repository = user_coupon_repository
        self.balance_service = balance_service
        self.billing_quota_service = billing_quota_service
        self.member_quota_service = member_quota_service

    async def get_profile(self, db: AsyncSession, user_id: int) -> dict:
        await self.member_repository.get_or_init_member(db, user_id)
        data = await self.member_repository.get_with_user(db, user_id)
        if not data:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        member = data["member"]
        benefit = await self.member_benefit_repository.get_by_level_code(db, member.level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        benefits = await self.member_benefit_repository.list_all(db)
        progress_percent, next_level_growth = _calc_progress(
            benefits, member.level_code, member.growth_value
        )

        return {
            "userId": member.user_id,
            "username": data.get("username") or "",
            "nickname": data.get("nickname"),
            "avatar": data.get("avatar"),
            "levelCode": member.level_code,
            "levelSource": member.level_source,
            "levelName": benefit.level_name,
            "growthValue": member.growth_value,
            "nextLevelGrowth": next_level_growth,
            "progressPercent": progress_percent,
            "expireTime": _format_dt(member.expire_time),
            "monthlyUsed": sum(getattr(member, f"monthly_{t}_used", 0) or 0 for t in QUOTA_TASK_TYPES),
            "benefits": _benefit_to_vo(benefit),
            "status": member.status,
        }

    async def list_paged_members(self, db: AsyncSession, query: dict) -> dict:
        items, total = await self.member_repository.get_page(
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

        benefits = await self.member_benefit_repository.list_all(db)
        benefit_map = {b.level_code: b for b in benefits}

        list_data = []
        for item in items:
            member = item["member"]
            benefit = benefit_map.get(member.level_code)
            level_name = benefit.level_name if benefit else ""
            monthly_used = sum(
                getattr(member, f"monthly_{t}_used", 0) or 0 for t in QUOTA_TASK_TYPES
            )
            list_data.append(
                {
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
                }
            )

        return {"list": list_data, "total": total}

    async def get_member_detail(self, db: AsyncSession, user_id: int) -> dict:
        data = await self.member_repository.get_with_user(db, user_id)
        if not data:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        member = data["member"]
        benefit = await self.member_benefit_repository.get_by_level_code(db, member.level_code)
        if not benefit:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "权益配置不存在")

        benefits = await self.member_benefit_repository.list_all(db)
        progress_percent, next_level_growth = _calc_progress(
            benefits, member.level_code, member.growth_value
        )

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
            "monthlyUsed": sum(
                getattr(member, f"monthly_{t}_used", 0) or 0 for t in QUOTA_TASK_TYPES
            ),
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

    async def adjust_level(self, db: AsyncSession, user_id: int, form: dict, operator_id: int) -> None:
        if not form.get("reason"):
            raise BusinessException(ResultCode.PARAM_ERROR, "调整原因必填")

        member = await self.member_repository.get_by_user_id(db, user_id)
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

        benefit = await self.member_benefit_repository.get_by_level_code(db, form["levelCode"])
        if benefit:
            _apply_benefit_quotas(member, benefit)

        await db.flush()
        await _invalidate_member_cache(user_id=user_id, level_code=old_level)
        await _invalidate_member_cache(level_code=form["levelCode"])

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="member",
            target_id=user_id,
            action="level_change",
            module="member",
            before_value={"levelCode": old_level},
            after_value=form.dict() if hasattr(form, "dict") else form,
        )

    async def adjust_growth(self, db: AsyncSession, user_id: int, form: dict, operator_id: int) -> None:
        if not form.get("reason"):
            raise BusinessException(ResultCode.PARAM_ERROR, "调整原因必填")

        change_value = form["changeValue"]
        if change_value == 0:
            raise BusinessException(ResultCode.PARAM_ERROR, "变动值不能为0")

        member = await self.member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        new_growth = member.growth_value + change_value
        if new_growth < 0:
            new_growth = 0

        member.growth_value = new_growth
        await db.flush()

        await self.member_growth_log_repository.create_log(
            db,
            user_id=user_id,
            change_type="admin_adjust",
            change_value=change_value,
            balance=new_growth,
            reason=form["reason"],
            operator_id=operator_id,
        )

        old_level = member.level_code
        await _check_and_adjust_level(db, member, self.member_benefit_repository)
        if member.level_code != old_level:
            await _invalidate_member_cache(user_id=user_id, level_code=old_level)
            await _invalidate_member_cache(level_code=member.level_code)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=operator_id,
            target_type="member",
            target_id=user_id,
            action="growth_change",
            module="member",
            after_value=form.dict() if hasattr(form, "dict") else form,
        )

    async def update_status(self, db: AsyncSession, user_id: int, form: dict) -> None:
        status = form["status"]
        reason = form.get("reason")

        if status == 0 and not reason:
            raise BusinessException(ResultCode.PARAM_ERROR, "冻结原因必填")

        member = await self.member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        old_status = member.status
        if status == 0:
            member.status = 0
            member.frozen_reason = reason
            member.frozen_time = datetime.now()
        else:
            # 解冻补回：会员卡到期时间顺延冻结天数，配额重置时点顺延；冻结原因/时间保留便于追溯
            frozen_days = 0
            if member.frozen_time is not None:
                frozen_days = (datetime.now() - member.frozen_time).days
            member.status = 1
            if frozen_days > 0:
                await self.member_repository.extend_expire_days(db, user_id, frozen_days)
                # 重置时点顺延后本周期不再被重置，当前配额需按权益刷新
                benefit = await self.member_benefit_repository.get_by_level_code(
                    db, member.level_code
                )
                if benefit:
                    await self.member_quota_service.refresh_member_quota(db, member, benefit)

        await db.flush()
        await _invalidate_member_cache(user_id=user_id)

        self.mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="member",
            target_id=user_id,
            action="status_change",
            module="member",
            before_value={"status": old_status},
            after_value=form.dict() if hasattr(form, "dict") else form,
        )

    # ───────────────────── 履约回调（订单模块同事务调用） ─────────────────────

    async def on_order_paid(self, db: AsyncSession, order) -> None:
        """订单支付成功回调：会员卡升级并刷新权益；所有商品按实付累积成长值（consume）。"""
        member = await self.member_repository.get_or_init_member(db, order.user_id)

        # 成长值累积：实付金额 1:1（金额单位分，1 元 = 100 分 = 100 成长值）
        consume_growth = int(order.paid_amount or 0)
        if consume_growth > 0:
            new_growth = member.growth_value + consume_growth
            member.growth_value = new_growth
            member.total_consumption = (member.total_consumption or 0) + consume_growth
            await self.member_growth_log_repository.create_log(
                db,
                user_id=order.user_id,
                change_type="consume",
                change_value=consume_growth,
                balance=new_growth,
                related_id=str(order.id),
                reason=f"购买{order.package_name}",
            )

        if order.package_type != "vip" or not order.package_level:
            # 积分卡不改变会员等级与权益，仅累积成长值
            if consume_growth > 0:
                await _check_and_adjust_level(db, member, self.member_benefit_repository)
            await _invalidate_member_cache(user_id=order.user_id)
            return

        # 会员卡：按商品等级升级，到期时间叠加（上限 3 年），刷新 8 类任务配额
        old_level = member.level_code
        benefit = await self.member_benefit_repository.get_by_level_code(db, order.package_level)
        if benefit:
            _apply_benefit_quotas(member, benefit)

        member.level_code = order.package_level
        member.level_source = "purchase"

        now = datetime.now()
        base_time = member.expire_time if member.expire_time and member.expire_time > now else now
        period_days = int(getattr(order, "period_days", None) or 0)
        if period_days > 0:
            new_expire = base_time + timedelta(days=period_days)
            cap = now + timedelta(days=MEMBER_CARD_MAX_DAYS)
            member.expire_time = new_expire if new_expire <= cap else cap
        else:
            member.expire_time = base_time

        if member.become_member_time is None:
            member.become_member_time = now

        await db.flush()
        await _invalidate_member_cache(user_id=order.user_id, level_code=old_level)
        await _invalidate_member_cache(level_code=order.package_level)

    async def on_order_refunded(self, db: AsyncSession, order, refund_record) -> None:
        """订单退款成功回调：扣减成长值（refund_deduct）、调整到期、重算等级（含降级）。

        仅会员卡涉及：积分卡积分回退由 AI 计费处理，本方法不处理积分卡。
        """
        if order.package_type != "vip":
            return

        member = await self.member_repository.get_by_user_id(db, order.user_id)
        if not member:
            return

        # 扣减成长值 = 该单消费累积成长值 × 未使用比例（与退款折算比例一致，最低 0）
        consumed_growth = int(order.paid_amount or 0)
        period_days = int(getattr(order, "period_days", None) or 0)
        used_days = int(getattr(refund_record, "used_days", 0) or 0)
        unused_ratio = max(0, period_days - used_days) / period_days if period_days > 0 else 0
        deduct_growth = int(consumed_growth * unused_ratio)

        new_growth = member.growth_value - deduct_growth
        if new_growth < 0:
            new_growth = 0
        actual_deduct = member.growth_value - new_growth

        old_level = member.level_code
        member.growth_value = new_growth

        if actual_deduct > 0:
            await self.member_growth_log_repository.create_log(
                db,
                user_id=order.user_id,
                change_type="refund_deduct",
                change_value=-actual_deduct,
                balance=new_growth,
                related_id=str(order.id),
                reason=f"退款{order.package_name}",
            )

        # 调整到期时间 = 支付时间 + 已用天数
        paid_time = getattr(order, "paid_time", None) or datetime.now()
        member.expire_time = paid_time + timedelta(days=used_days)

        # 基于扣减后成长值重算等级（触发降级）
        await _check_and_adjust_level(db, member, self.member_benefit_repository)
        await _invalidate_member_cache(user_id=order.user_id, level_code=old_level)

    # ───────────────────── 权益概览 / 试用引导 ─────────────────────

    async def get_benefit_summary(self, db: AsyncSession, user_id: int) -> dict:
        cache_key = f"member:benefit-summary:{user_id}"
        result = await self._read_summary_cache(cache_key)
        if result is not None:
            return result

        member = await self.member_repository.get_or_init_member(db, user_id)
        benefit = await self.member_benefit_repository.get_by_level_code(db, member.level_code)

        # 图像处理 7 类：各自剩余取最低值，details 返回各任务明细
        image_details = []
        image_remaining = None
        for task_type in IMAGE_TASK_TYPES:
            quota = getattr(member, f"monthly_{task_type}_quota", 0) or 0
            used = getattr(member, f"monthly_{task_type}_used", 0) or 0
            remaining = quota - used
            if image_remaining is None or remaining < image_remaining:
                image_remaining = remaining
            image_details.append({"taskType": task_type, "quota": quota, "used": used, "remaining": remaining})

        # 评估类目：剩余 = quota - used
        evaluate_quota = member.monthly_evaluate_quota or 0
        evaluate_used = member.monthly_evaluate_used or 0

        # AI 类目：余额/今日已用/限额
        credits_balance = int(await self.balance_service.get_balance(db, user_id))
        today_used, _ = await self.billing_quota_service.get_used(user_id)

        daily_limit = benefit.ai_credits_daily if benefit else 0
        monthly_limit = benefit.ai_credits_monthly if benefit else 0
        # 已购会员卡取覆盖值与等级权益较高值
        overrides = await self._active_card_overrides(db, member)
        if overrides:
            daily_limit = max(daily_limit, int(overrides.get("ai_credits_daily") or 0))
            monthly_limit = max(monthly_limit, int(overrides.get("ai_credits_monthly") or 0))

        result = {
            "imageCategory": {
                "remaining": image_remaining if image_remaining is not None else 0,
                "details": image_details,
            },
            "evaluateCategory": {
                "remaining": evaluate_quota - evaluate_used,
            },
            "aiCategory": {
                "creditsBalance": int(credits_balance),
                "todayUsed": int(today_used),
                "dailyLimit": int(daily_limit),
                "monthlyLimit": int(monthly_limit),
            },
        }

        await self._set_summary_cache(cache_key, result)
        return result

    async def _active_card_overrides(self, db: AsyncSession, member: SysMember) -> dict | None:
        """已购会员卡（level_source=purchase 且未到期）的 benefit_overrides，无则返回 None。"""
        if member.level_source != "purchase" or (
            member.expire_time is not None and member.expire_time < datetime.now()
        ):
            return None
        package = await self.package_repository.get_by_level_code(db, member.level_code)
        if package is None or not package.benefit_overrides:
            return None
        return package.benefit_overrides

    async def get_trial_status(self, db: AsyncSession, user_id: int) -> dict:
        member = await self.member_repository.get_or_init_member(db, user_id)

        # 体验券激活状态：持有未使用且未过期的 trial 券即视为已激活
        trial_coupon = await self.user_coupon_repository.get_active_trial_coupon(db, user_id)
        voucher_activated = trial_coupon is not None
        voucher_expire_time = _format_dt(trial_coupon.expire_time) if trial_coupon else None

        # AI 试用积分余额：trial 来源累计（无记录返回 0）
        ai_trial_balance = 0
        try:
            by_source = await self.ai_credit_log_repository.sum_amount_by_user_and_source(
                db, user_id
            )
            ai_trial_balance = int(by_source.get("trial", 0))
        except Exception:
            logger.debug("AI 试用积分余额查询失败: user_id=%s", user_id)

        # 新用户专享可用：无历史付费订单
        new_user_exclusive_available = not await self.order_repository.has_paid_order(db, user_id)

        paid_membership = member.level_source == "purchase" or member.expire_time is not None

        show_trial_entry = (
            (not voucher_activated)
            or ai_trial_balance > 0
            or new_user_exclusive_available
        )

        return {
            "showTrialEntry": bool(show_trial_entry),
            "trialDays": TRIAL_DEFAULT_DAYS,
            "trialCredits": TRIAL_DEFAULT_CREDITS,
            "voucherActivated": voucher_activated,
            "voucherExpireTime": voucher_expire_time,
            "aiTrialCreditsBalance": ai_trial_balance,
            "newUserExclusiveAvailable": bool(new_user_exclusive_available),
            "paidMembership": paid_membership,
        }

    async def _read_summary_cache(self, key: str):
        async def _get():
            redis = await get_redis_client()
            return await redis.get(key)

        raw = await redis_operation_with_fallback(
            _get, default=None, operation_name="member_benefit_summary_cache_get"
        )
        if not raw:
            return None
        import json
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def _set_summary_cache(self, key: str, value: dict) -> None:
        import json

        async def _set():
            redis = await get_redis_client()
            await redis.setex(key, 300, json.dumps(value, ensure_ascii=False, default=str))

        await redis_operation_with_fallback(
            _set, default=None, operation_name="member_benefit_summary_cache_set"
        )


member_service = MemberService()
