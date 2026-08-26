"""
会员核心 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：get_profile 等级来源、履约回调 on_order_paid/on_order_refunded、
成长值调整降级规则、权益概览 get_benefit_summary、试用引导 get_trial_status、
8 类任务月度已用统计。

遵循 dehaze 测试规范：
- 仅依赖 db fixture 与 mock_redis（autouse）；只断言业务结果，不 mock 调用序列
- 命名 test_功能_场景
"""

from datetime import datetime, timedelta

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_package import SysPackage
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.repository.package_repository import package_repository
from app.service.member.member_service import member_service

pytestmark = pytest.mark.requires_db

USER_ID = 1002001


async def _setup_benefit(db, level_code: str, *, quota: int = 50, ai_daily: int = 100,
                         ai_monthly: int = 1000, growth_min: int = 1000,
                         growth_max: int = 4999):
    """更新指定等级权益配置：8 类任务配额统一为 quota，并设定 AI 限额与成长值区间"""
    benefit = await member_benefit_repository.get_by_level_code(db, level_code)
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", quota)
    benefit.ai_credits_daily = ai_daily
    benefit.ai_credits_monthly = ai_monthly
    benefit.growth_min = growth_min
    benefit.growth_max = growth_max
    await db.flush()
    return benefit


async def _setup_member(db, user_id: int = USER_ID, *, level_code: str = "level_0",
                        growth_value: int = 0, level_source: str = "growth",
                        expire_time: datetime | None = None):
    member = await member_repository.get_or_init_member(db, user_id)
    member.level_code = level_code
    member.level_source = level_source
    member.growth_value = growth_value
    member.expire_time = expire_time
    await db.flush()
    return member


def _make_order(user_id: int, *, package_type: str, package_level: str | None,
                paid_amount: int, period_days: int = 30, package_name: str = "测试套餐",
                paid_time: datetime | None = None) -> SysOrder:
    return SysOrder(
        order_no=f"test_mem_{user_id}_{package_type}",
        user_id=user_id,
        package_id=1,
        package_level=package_level or "",
        package_name=package_name,
        package_type=package_type,
        original_price=paid_amount,
        payable_amount=paid_amount,
        paid_amount=paid_amount,
        period_days=period_days,
        pay_method="test",
        status=2,
        paid_time=paid_time or datetime.now(),
        expire_time=(paid_time or datetime.now()) + timedelta(days=period_days),
    )


# ===================== 修复验证：_check_and_adjust_level 不再 NameError =====================

async def test_adjust_growth_triggers_level_check_no_name_error(db):
    """成长值调整触发等级检查，模块级函数不再因引用注入仓储而 NameError"""
    member = await _setup_member(db, level_code="level_1", growth_value=1500)
    await member_service.adjust_growth(
        db, USER_ID, {"changeValue": -100, "reason": "测试降级"}, operator_id=2
    )
    await db.flush()
    updated = await member_repository.get_by_user_id(db, USER_ID)
    assert updated.growth_value == 1400
    # level_1 下限 1000，1400 仍达标，等级不变且不报错
    assert updated.level_code == "level_1"


# ===================== get_profile 含 levelSource =====================

async def test_get_profile_contains_level_source(db):
    await _setup_member(db, level_code="level_2", growth_value=8000)
    profile = await member_service.get_profile(db, USER_ID)
    assert profile["levelCode"] == "level_2"
    assert profile["levelSource"] == "growth"
    assert profile["levelName"] == "VIP2"
    assert "growthValue" in profile
    assert profile["monthlyUsed"] == 0


async def test_get_profile_monthly_used_sums_8_types(db):
    member = await _setup_member(db, level_code="level_1", growth_value=1500)
    member.monthly_dehaze_used = 3
    member.monthly_derain_used = 2
    member.monthly_evaluate_used = 1
    await db.flush()
    profile = await member_service.get_profile(db, USER_ID)
    assert profile["monthlyUsed"] == 6


# ===================== on_order_paid：vip 升级 =====================

async def test_on_order_paid_vip_upgrade_and_quota_refresh(db):
    await _setup_benefit(db, "level_1", quota=60, ai_daily=200, ai_monthly=2000)
    order = _make_order(USER_ID, package_type="vip", package_level="level_1",
                        paid_amount=8000, period_days=30)
    order.id = 1001
    await member_service.on_order_paid(db, order)
    await db.flush()

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.level_code == "level_1"
    assert member.level_source == "purchase"
    assert member.expire_time is not None
    assert (member.expire_time - datetime.now()) > timedelta(days=25)
    # 8 类任务配额已按等级权益刷新
    for task_type in QUOTA_TASK_TYPES:
        assert getattr(member, f"monthly_{task_type}_quota") == 60
    # 成长值按实付累积（分单位 1:1）
    assert member.growth_value == 8000
    assert member.total_consumption == 8000

    # consume 成长值流水已写
    logs, _ = await member_growth_log_repository.get_page(db, USER_ID, 1, 10)
    assert any(l.change_type == "consume" and l.change_value == 8000 for l in logs)


async def test_on_order_paid_vip_renewal_stack_expire(db):
    """续费在原到期时间上叠加（上限 3 年）"""
    await _setup_benefit(db, "level_1", quota=60)
    base_expire = datetime.now() + timedelta(days=10)
    await _setup_member(db, level_code="level_1", level_source="purchase",
                        expire_time=base_expire)
    order = _make_order(USER_ID, package_type="vip", package_level="level_1",
                        paid_amount=8000, period_days=30)
    order.id = 1002
    await member_service.on_order_paid(db, order)
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    # 原到期 + 30 天
    assert (member.expire_time - base_expire) >= timedelta(days=29)


async def test_on_order_paid_credit_only_growth_no_level_change(db):
    """积分卡仅累积成长值：不设置 purchase 来源、不刷新权益配额、不改到期时间"""
    await _setup_member(db, level_code="level_0", growth_value=0)
    order = _make_order(USER_ID, package_type="credit", package_level=None,
                        paid_amount=100, period_days=0)
    order.id = 1003
    await member_service.on_order_paid(db, order)
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    # 小额积分卡未触发成长值升级
    assert member.level_code == "level_0"
    assert member.level_source == "growth"
    assert member.growth_value == 100
    assert member.expire_time is None


# ===================== on_order_refunded：vip 扣成长值 =====================

async def test_on_order_refunded_vip_deduct_and_downgrade(db):
    await _setup_benefit(db, "level_1", quota=60, growth_min=1000, growth_max=4999)
    # 已购会员卡，成长值 3000，全部未使用退款 → 扣回全部 3000
    await _setup_member(db, level_code="level_1", level_source="purchase",
                        growth_value=3000, expire_time=datetime.now() + timedelta(days=30))
    order = _make_order(USER_ID, package_type="vip", package_level="level_1",
                        paid_amount=3000, period_days=30)
    order.id = 1004

    class RefundRecord:
        used_days = 0

    await member_service.on_order_refunded(db, order, RefundRecord())
    await db.flush()

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 0
    # 成长值跌至 0 → 降级 level_0，来源切 growth、到期清空
    assert member.level_code == "level_0"
    assert member.level_source == "growth"
    assert member.expire_time is None

    logs, _ = await member_growth_log_repository.get_page(db, USER_ID, 1, 10)
    assert any(l.change_type == "refund_deduct" and l.change_value == -3000 for l in logs)


async def test_on_order_refunded_vip_used_days_no_deduct(db):
    """已用天数等于周期天数 → 未使用比例 0，不扣成长值"""
    await _setup_benefit(db, "level_1", quota=60, growth_min=1000, growth_max=4999)
    await _setup_member(db, level_code="level_1", level_source="purchase",
                        growth_value=3000, expire_time=datetime.now() + timedelta(days=5))
    order = _make_order(USER_ID, package_type="vip", package_level="level_1",
                        paid_amount=3000, period_days=30)
    order.id = 1005

    class RefundRecord:
        used_days = 30

    await member_service.on_order_refunded(db, order, RefundRecord())
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 3000


async def test_on_order_refunded_credit_noop(db):
    """积分卡退款不改变会员成长值与等级"""
    await _setup_member(db, level_code="level_0", growth_value=5000)
    order = _make_order(USER_ID, package_type="credit", package_level=None,
                        paid_amount=5000, period_days=0)
    order.id = 1006

    class RefundRecord:
        used_days = 0

    await member_service.on_order_refunded(db, order, RefundRecord())
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 5000
    assert member.level_code == "level_0"


# ===================== adjust_growth 降级规则 =====================

async def test_adjust_growth_growth_source_downgrade(db):
    await _setup_benefit(db, "level_1", quota=60, growth_min=1000, growth_max=4999)
    await _setup_member(db, level_code="level_1", level_source="growth", growth_value=1500)
    # 扣减 1000 → 500，低于 level_1 下限 → 降级 level_0
    await member_service.adjust_growth(
        db, USER_ID, {"changeValue": -1000, "reason": "测试降级"}, operator_id=2
    )
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.level_code == "level_0"
    assert member.level_source == "growth"


async def test_adjust_growth_admin_source_no_downgrade(db):
    await _setup_benefit(db, "level_1", quota=60, growth_min=1000, growth_max=4999)
    await _setup_member(db, level_code="level_1", level_source="admin", growth_value=1500)
    await member_service.adjust_growth(
        db, USER_ID, {"changeValue": -1000, "reason": "测试"}, operator_id=2
    )
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    # admin 来源不自动降级
    assert member.level_code == "level_1"


async def test_adjust_growth_deduct_not_negative(db):
    await _setup_member(db, level_code="level_0", growth_value=100)
    await member_service.adjust_growth(
        db, USER_ID, {"changeValue": -500, "reason": "扣减超过现有"}, operator_id=2
    )
    await db.flush()
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 0


# ===================== 权益概览 =====================

async def test_benefit_summary_image_category_min_remaining(db):
    """图像处理类目 remaining 取 7 类任务最低剩余；各项数字非空"""
    await _setup_benefit(db, "level_1", quota=50)
    member = await _setup_member(db, level_code="level_1", growth_value=1500)
    # 权益概览按会员当前 8 类配额/已用读取
    for task_type in ["dehaze", "derain", "desnow", "lowlight",
                      "super_resolution", "denoise", "inpaint"]:
        setattr(member, f"monthly_{task_type}_quota", 50)
    member.monthly_dehaze_used = 10   # remaining 40
    member.monthly_derain_used = 40   # remaining 10（最低）
    await db.flush()

    summary = await member_service.get_benefit_summary(db, USER_ID)
    image = summary["imageCategory"]
    # 各任务 remaining 均为 int
    assert all(isinstance(d["remaining"], int) for d in image["details"])
    assert image["remaining"] == 10
    dehaze_detail = next(d for d in image["details"] if d["taskType"] == "dehaze")
    assert dehaze_detail["quota"] == 50
    assert dehaze_detail["used"] == 10
    assert dehaze_detail["remaining"] == 40
    assert len(image["details"]) == 7


async def test_benefit_summary_evaluate_and_ai_category(db):
    await _setup_benefit(db, "level_1", quota=50, ai_daily=100, ai_monthly=1000)
    member = await _setup_member(db, level_code="level_1", growth_value=1500)
    member.monthly_evaluate_quota = 50
    member.monthly_evaluate_used = 20
    await db.flush()

    summary = await member_service.get_benefit_summary(db, USER_ID)
    evaluate = summary["evaluateCategory"]
    assert evaluate["remaining"] == 30
    ai = summary["aiCategory"]
    assert isinstance(ai["creditsBalance"], int)
    assert isinstance(ai["todayUsed"], int)
    assert ai["dailyLimit"] == 100
    assert ai["monthlyLimit"] == 1000


async def test_benefit_summary_cached(db, mock_redis):
    """权益概览结果缓存于 member:benefit-summary:{userId}，二次读取命中"""
    await _setup_member(db, level_code="level_0", growth_value=0)
    await member_service.get_benefit_summary(db, USER_ID)
    key = f"member:benefit-summary:{USER_ID}"
    cached = await mock_redis.get(key)
    assert cached is not None


# ===================== 试用引导 =====================

async def test_trial_status_structure_complete(db):
    """试用引导状态结构字段完整且类型正确"""
    await _setup_member(db, level_code="level_0", growth_value=0)
    status = await member_service.get_trial_status(db, USER_ID)
    assert isinstance(status["showTrialEntry"], bool)
    assert isinstance(status["newUserExclusiveAvailable"], bool)
    assert status["trialDays"] == 3
    assert status["trialCredits"] == 100
    assert status["voucherActivated"] is False
    assert status["voucherExpireTime"] is None
    assert isinstance(status["aiTrialCreditsBalance"], int)
    assert status["paidMembership"] is False
    # 无历史付费订单 → 新用户专享可用
    assert status["newUserExclusiveAvailable"] is True
    assert status["showTrialEntry"] is True


async def test_trial_status_paid_membership_false_new_user(db):
    await _setup_member(db, level_code="level_0", growth_value=0)
    status = await member_service.get_trial_status(db, USER_ID)
    assert status["paidMembership"] is False


async def test_trial_status_purchase_member(db):
    await _setup_member(db, level_code="level_1", level_source="purchase",
                        expire_time=datetime.now() + timedelta(days=30))
    status = await member_service.get_trial_status(db, USER_ID)
    assert status["paidMembership"] is True


# ===================== list_paged_members 8 类月度已用 =====================

async def test_list_paged_members_monthly_used_8_types(db):
    await _setup_member(db, level_code="level_1", growth_value=1500)
    member = await member_repository.get_by_user_id(db, USER_ID)
    member.monthly_dehaze_used = 2
    member.monthly_inpaint_used = 3
    member.monthly_denoise_used = 1
    await db.flush()

    page = await member_service.list_paged_members(
        db, {"pageNum": 1, "pageSize": 10, "keywords": None, "levelCode": None,
             "status": None, "expireTimeStart": None, "expireTimeEnd": None,
             "growthMin": None, "growthMax": None}
    )
    row = next(x for x in page["list"] if x["userId"] == USER_ID)
    assert row["monthlyUsed"] == 6
