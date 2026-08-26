"""
会员到期处理域 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：到期降级 process_expired_members（8 类任务配额刷新、来源切换、到期清空）、
到期提醒 send_expire_reminders（修复验证不再 NameError）。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from datetime import datetime, timedelta

import pytest

from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.service.member.expiry_service import member_expiry_service

pytestmark = pytest.mark.requires_db

USER_ID = 1005001


async def _setup_member(db, *, level_code: str, level_source: str, growth_value: int,
                        expire_time: datetime | None):
    member = await member_repository.get_or_init_member(db, USER_ID)
    member.level_code = level_code
    member.level_source = level_source
    member.growth_value = growth_value
    member.expire_time = expire_time
    await db.flush()
    return member


# ===================== 到期降级 =====================

async def test_process_expired_members_downgrade_and_refresh_8_types(db):
    """到期会员按成长值重算等级、来源切 growth、到期清空、8 类配额刷新"""
    # 目标降级等级为 level_0（成长值 100），其权益配置 8 类配额设为 99
    benefit = await member_benefit_repository.get_by_level_code(db, "level_0")
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", 99)
    await db.flush()

    member = await _setup_member(
        db, level_code="level_2", level_source="purchase", growth_value=100,
        expire_time=datetime.now() - timedelta(days=1),
    )
    member.monthly_dehaze_quota = 500
    member.monthly_derain_quota = 500
    await db.flush()

    count = await member_expiry_service.process_expired_members(db)
    assert count >= 1

    updated = await member_repository.get_by_user_id(db, USER_ID)
    # 成长值 100 → level_0
    assert updated.level_code == "level_0"
    assert updated.level_source == "growth"
    assert updated.expire_time is None
    # 8 类配额按目标等级权益刷新
    for task_type in QUOTA_TASK_TYPES:
        assert getattr(updated, f"monthly_{task_type}_quota") == 99


async def test_process_expired_members_keep_level_no_quota_gap(db):
    """成长值仍达原等级时保持等级（保级），到期期间不出现权益空窗"""
    benefit = await member_benefit_repository.get_by_level_code(db, "level_2")
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", 88)
    await db.flush()

    member = await _setup_member(
        db, level_code="level_2", level_source="purchase", growth_value=8000,
        expire_time=datetime.now() - timedelta(days=1),
    )
    member.monthly_dehaze_quota = 0
    await db.flush()

    count = await member_expiry_service.process_expired_members(db)
    assert count >= 1

    updated = await member_repository.get_by_user_id(db, USER_ID)
    # 成长值 8000 仍达 level_2 → 保级，来源切 growth、到期清空、配额不空窗
    assert updated.level_code == "level_2"
    assert updated.level_source == "growth"
    assert updated.expire_time is None
    assert updated.monthly_dehaze_quota == 88


async def test_process_expired_members_skips_non_expired(db):
    """未到期会员不被处理"""
    await _setup_member(
        db, level_code="level_1", level_source="purchase", growth_value=1500,
        expire_time=datetime.now() + timedelta(days=30),
    )
    count = await member_expiry_service.process_expired_members(db)
    updated = await member_repository.get_by_user_id(db, USER_ID)
    assert updated.level_source == "purchase"
    assert updated.expire_time is not None


# ===================== 到期提醒（修复验证） =====================

async def test_send_expire_reminders_no_name_error(db):
    """send_expire_reminders 不再因 staticmethod 引用 self 而 NameError"""
    await _setup_member(
        db, level_code="level_1", level_source="purchase", growth_value=1500,
        expire_time=datetime.now() + timedelta(days=3),
    )
    # 不抛异常即验证通过；消息发送失败被内部捕获不影响主流程
    count = await member_expiry_service.send_expire_reminders(db)
    assert count >= 0
