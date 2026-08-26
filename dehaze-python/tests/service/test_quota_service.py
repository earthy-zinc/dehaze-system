"""
会员配额域 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：8 类任务配额校验扣减（含 derain/desnow 等非 dehaze 任务）、冻结会员 A0511、
配额不足 A0515、并发防超扣、月度配额重置归档与 8 类刷新、幂等。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from datetime import datetime

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import QUOTA_TASK_TYPES, SysMember
from app.models.entity.sys_member_quota import SysMemberQuota
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_repository import member_repository
from app.service.member.quota_service import member_quota_service
from sqlalchemy import select

pytestmark = pytest.mark.requires_db

USER_ID = 1003001


async def _setup_benefit(db, level_code: str = "level_1", quota: int = 50):
    benefit = await member_benefit_repository.get_by_level_code(db, level_code)
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", quota)
    await db.flush()
    return benefit


async def _setup_member(db, user_id: int = USER_ID, *, level_code: str = "level_1",
                        status: int = 1):
    member = await member_repository.get_or_init_member(db, user_id)
    member.level_code = level_code
    member.status = status
    await db.flush()
    return member


# ===================== 8 类任务扣减 =====================

async def test_check_and_deduct_derain_task(db):
    """derain 等非 dehaze 任务正常校验扣减"""
    await _setup_benefit(db, "level_1", quota=50)
    await _setup_member(db, level_code="level_1")
    await member_quota_service.check_and_deduct_quota(db, USER_ID, "derain")
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.monthly_derain_used == 1


async def test_check_and_deduct_desnow_and_inpaint(db):
    await _setup_benefit(db, "level_1", quota=50)
    await _setup_member(db, level_code="level_1")
    await member_quota_service.check_and_deduct_quota(db, USER_ID, "desnow")
    await member_quota_service.check_and_deduct_quota(db, USER_ID, "inpaint")
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member.monthly_desnow_used == 1
    assert member.monthly_inpaint_used == 1


async def test_check_and_deduct_unsupported_type(db):
    await _setup_member(db, level_code="level_1")
    with pytest.raises(BusinessException) as exc:
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "unknown")
    assert exc.value.code == ResultCode.PARAM_ERROR


# ===================== 冻结会员 =====================

async def test_check_and_deduct_frozen_raises(db):
    await _setup_benefit(db, "level_1", quota=50)
    await _setup_member(db, level_code="level_1", status=0)
    with pytest.raises(BusinessException) as exc:
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "dehaze")
    assert exc.value.code == ResultCode.MEMBER_FROZEN


# ===================== 配额不足 =====================

async def test_check_and_deduct_quota_exceeded(db):
    await _setup_benefit(db, "level_1", quota=2)
    member = await _setup_member(db, level_code="level_1")
    member.monthly_derain_used = 2
    await db.flush()
    with pytest.raises(BusinessException) as exc:
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "derain")
    assert exc.value.code == ResultCode.QUOTA_EXCEEDED


async def test_check_and_deduct_redis_remaining_zero(db, mock_redis):
    """Redis 剩余为 0 时直接抛配额不足"""
    await _setup_benefit(db, "level_1", quota=50)
    await _setup_member(db, level_code="level_1")
    await mock_redis.set(f"member:quota:{USER_ID}:dehaze", 0)
    with pytest.raises(BusinessException) as exc:
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "dehaze")
    assert exc.value.code == ResultCode.QUOTA_EXCEEDED


# ===================== 并发防超扣 =====================

async def test_concurrent_deduct_not_over(db):
    """配额仅 1 时，模拟两次扣减（Redis 未命中路径）第二次应失败不超扣"""
    await _setup_benefit(db, "level_1", quota=1)
    member = await _setup_member(db, level_code="level_1")
    # 预置 Redis 无缓存，走落库条件更新路径；第一次成功
    await member_quota_service.check_and_deduct_quota(db, USER_ID, "dehaze")
    updated = await member_repository.get_by_user_id(db, USER_ID)
    assert updated.monthly_dehaze_used == 1
    # 第二次：条件更新（quota=1 > used=1 为假）失败 → 配额不足
    with pytest.raises(BusinessException) as exc:
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "dehaze")
    assert exc.value.code == ResultCode.QUOTA_EXCEEDED
    assert (await member_repository.get_by_user_id(db, USER_ID)).monthly_dehaze_used == 1


# ===================== 归还配额 =====================

async def test_restore_quota_derain(db):
    await _setup_benefit(db, "level_1", quota=50)
    member = await _setup_member(db, level_code="level_1")
    member.monthly_derain_used = 3
    await db.flush()
    await member_quota_service.restore_quota(db, USER_ID, "derain")
    assert (await member_repository.get_by_user_id(db, USER_ID)).monthly_derain_used == 2


# ===================== 月度配额重置 =====================

async def test_reset_monthly_quota_archives_and_resets_8_types(db):
    await _setup_benefit(db, "level_1", quota=80)
    member = await _setup_member(db, level_code="level_1")
    member.quota_reset_month = 202607  # 上月
    for task_type in QUOTA_TASK_TYPES:
        setattr(member, f"monthly_{task_type}_quota", 10)
        setattr(member, f"monthly_{task_type}_used", 3)
    await db.flush()

    count = await member_quota_service.reset_monthly_quota(db)

    assert count >= 1
    # 历史表已归档上月使用情况（8 类）
    archived = (await db.execute(
        select(SysMemberQuota).where(
            SysMemberQuota.user_id == USER_ID, SysMemberQuota.quota_month == 202607
        )
    )).scalars().all()
    assert len(archived) == 1
    assert archived[0].dehaze_used == 3
    assert archived[0].derain_used == 3
    assert archived[0].inpaint_used == 3
    assert archived[0].derain_quota == 10

    # 本月配额按等级权益刷新、已用清零、quota_reset_month 更新
    updated = await member_repository.get_by_user_id(db, USER_ID)
    assert updated.quota_reset_month == int(datetime.now().strftime("%Y%m"))
    for task_type in QUOTA_TASK_TYPES:
        assert getattr(updated, f"monthly_{task_type}_quota") == 80
        assert getattr(updated, f"monthly_{task_type}_used") == 0


async def test_reset_monthly_quota_idempotent(db):
    """quota_reset_month 已为本月 → 幂等，不再重复归档"""
    await _setup_benefit(db, "level_1", quota=80)
    current_month = int(datetime.now().strftime("%Y%m"))
    member = await _setup_member(db, level_code="level_1")
    member.quota_reset_month = current_month
    await db.flush()

    await member_quota_service.reset_monthly_quota(db)
    archived = (await db.execute(
        select(SysMemberQuota).where(SysMemberQuota.user_id == USER_ID)
    )).scalars().all()
    assert len(archived) == 0


async def test_reset_monthly_quota_skips_frozen(db):
    """冻结会员跳过重置（解冻时顺延重置时点）"""
    await _setup_benefit(db, "level_1", quota=80)
    member = await _setup_member(db, level_code="level_1", status=0)
    member.quota_reset_month = 202607
    member.monthly_dehaze_used = 5
    await db.flush()

    await member_quota_service.reset_monthly_quota(db)
    updated = await member_repository.get_by_user_id(db, USER_ID)
    assert updated.quota_reset_month == 202607
    assert updated.monthly_dehaze_used == 5
