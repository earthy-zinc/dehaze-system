"""
成长值/签到域 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：签到（修复验证不再 NameError）、AI 使用激励 add_ai_consume_growth（每日上限）、
成长值流水与等级联动、签到日历与流水列表。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from datetime import date

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.service.member.growth_service import (
    AI_CONSUME_DAILY_LIMIT,
    member_growth_service,
)

pytestmark = pytest.mark.requires_db

USER_ID = 1004001


async def _setup_benefit(db, level_code: str = "level_1"):
    benefit = await member_benefit_repository.get_by_level_code(db, level_code)
    benefit.growth_min = 1000
    benefit.growth_max = 4999
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", 50)
    await db.flush()
    return benefit


async def _setup_member(db, *, level_code: str = "level_0", growth_value: int = 0,
                        level_source: str = "growth"):
    member = await member_repository.get_or_init_member(db, USER_ID)
    member.level_code = level_code
    member.level_source = level_source
    member.growth_value = growth_value
    await db.flush()
    return member


# ===================== 签到（修复验证） =====================

async def test_sign_in_success_no_name_error(db):
    """签到不再因仓储注入不一致而 NameError，且正确累计成长值"""
    await _setup_member(db)
    result = await member_growth_service.sign_in(db, USER_ID)
    assert result["growthValue"] == 3
    assert result["bonusGrowth"] == 0
    assert result["continuousDays"] == 1
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 3


async def test_sign_in_already_today(db):
    await _setup_member(db)
    await member_growth_service.sign_in(db, USER_ID)
    with pytest.raises(BusinessException) as exc:
        await member_growth_service.sign_in(db, USER_ID)
    assert exc.value.code == ResultCode.SIGN_IN_ALREADY


async def test_sign_in_weekly_bonus(db):
    """第 7 天连续签到触发额外成长值奖励"""
    from datetime import timedelta
    from app.models.entity.sys_member_sign_in import SysMemberSignIn
    await _setup_member(db)
    today = date.today()
    # 构造昨天为止连续 6 天的签到记录，使今天签到为第 7 天
    for i in range(6):
        db.add(SysMemberSignIn(
            user_id=USER_ID,
            sign_date=today - timedelta(days=6 - i),
            continuous_days=i + 1,
            growth_value=3,
        ))
    await db.flush()
    result = await member_growth_service.sign_in(db, USER_ID)
    assert result["continuousDays"] == 7
    assert result["bonusGrowth"] == 20


async def test_sign_in_calendar(db, mock_redis):
    await _setup_member(db)
    await member_growth_service.sign_in(db, USER_ID)
    today = date.today()
    calendar = await member_growth_service.get_sign_in_calendar(
        db, USER_ID, today.year, today.month
    )
    assert calendar["totalDays"] == 1
    assert calendar["signDates"] == [today.strftime("%Y-%m-%d")]


# ===================== 成长值流水 =====================

async def test_list_growth_logs(db):
    await _setup_member(db)
    member = await member_repository.get_by_user_id(db, USER_ID)
    member.growth_value = 100
    await db.flush()
    await member_growth_log_repository.create_log(
        db, user_id=USER_ID, change_type="admin_adjust",
        change_value=100, balance=100, reason="测试",
    )
    result = await member_growth_service.list_growth_logs(
        db, USER_ID, {"pageNum": 1, "pageSize": 10, "changeType": None,
                      "startTime": None, "endTime": None}
    )
    assert result["total"] >= 1
    assert result["list"][0]["changeType"] == "admin_adjust"


# ===================== AI 使用激励 =====================

async def test_add_ai_consume_growth_once(db, mock_redis):
    await _setup_member(db)
    ok = await member_growth_service.add_ai_consume_growth(db, USER_ID)
    assert ok is True
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 1
    logs, _ = await member_growth_log_repository.get_page(db, USER_ID, 1, 10)
    assert any(l.change_type == "ai_consume" and l.change_value == 1 for l in logs)


async def test_add_ai_consume_growth_daily_limit(db, mock_redis):
    """每日上限 10 次，第 11 次不再累计"""
    await _setup_member(db)
    for _ in range(AI_CONSUME_DAILY_LIMIT):
        ok = await member_growth_service.add_ai_consume_growth(db, USER_ID)
        assert ok is True
    blocked = await member_growth_service.add_ai_consume_growth(db, USER_ID)
    assert blocked is False
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == AI_CONSUME_DAILY_LIMIT


async def test_add_ai_consume_growth_triggers_level_upgrade(db, mock_redis):
    """AI 激励成长值累计触发等级联动升级"""
    await _setup_benefit(db, "level_1")
    await _setup_member(db, level_code="level_0", growth_value=999)
    ok = await member_growth_service.add_ai_consume_growth(db, USER_ID)
    assert ok is True
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.growth_value == 1000
    assert member.level_code == "level_1"
