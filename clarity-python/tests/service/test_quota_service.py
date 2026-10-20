"""
会员配额域 service 单元测试（真实 MySQL 测试库 clarity_test + SAVEPOINT 回滚）。

覆盖：生效配额解析（等级权益与会员卡覆盖取较高值）。
任务消费（校验/扣减/归还）与月度重置已随 D1 全积分制退役，对应用例随实现删除。
"""

import pytest

from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.repository.member_benefit_repository import member_benefit_repository
from app.service.member.quota_service import _effective_task_quota

pytestmark = pytest.mark.requires_db


async def _setup_benefit(db, level_code: str = "level_1", quota: int = 50):
    benefit = await member_benefit_repository.get_by_level_code(db, level_code)
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", quota)
    await db.flush()
    return benefit


async def test_effective_task_quota_overrides_camelcase(db):
    """会员卡覆盖项 key 为 camelCase（套餐侧写入/DB/Java 统一口径），合并取较高值生效"""

    benefit = await _setup_benefit(db, "level_1", quota=50)
    assert benefit is not None

    overrides = {
        "monthlyDehazeQuota": 100,
        "monthlySuperResolutionQuota": 200,
        "monthlyDenoiseQuota": 10,
    }
    effective = _effective_task_quota(benefit, overrides)
    assert effective["dehaze"] == 100
    assert effective["super_resolution"] == 200
    assert effective["denoise"] == max(benefit.monthly_denoise_quota, 10)
    for task_type in QUOTA_TASK_TYPES:
        assert effective[task_type] >= getattr(benefit, f"monthly_{task_type}_quota", 0) or 0
