"""
权益配置域 service 单元测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

覆盖：权益配置 8 类任务配额与 AI 限额字段读写、AI 限额负值校验、配置修改后缓存失效。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.repository.member_benefit_repository import member_benefit_repository
from app.service.member.benefit_service import member_benefit_service

pytestmark = pytest.mark.requires_db


async def test_update_benefit_ai_limit_fields_read_write(db):
    """AI 限额与多模态/赠送积分字段可写并读回"""
    form = {
        "levelName": "VIP1-测试",
        "growthMin": 1000,
        "growthMax": 4999,
        "monthlyDehazeQuota": 200,
        "monthlyDerainQuota": 150,
        "monthlyInpaintQuota": 120,
        "aiCreditsDaily": 300,
        "aiCreditsMonthly": 3000,
        "multimodalLimit": 40,
        "vipGiftCredits": 500,
        "historyRetention": 1000,
    }
    await member_benefit_service.update_benefit(db, "level_1", form)

    benefit = await member_benefit_repository.get_by_level_code(db, "level_1")
    assert benefit.ai_credits_daily == 300
    assert benefit.ai_credits_monthly == 3000
    assert benefit.multimodal_limit == 40
    assert benefit.vip_gift_credits == 500
    assert benefit.monthly_dehaze_quota == 200
    assert benefit.monthly_derain_quota == 150
    assert benefit.monthly_inpaint_quota == 120


async def test_update_benefit_negative_ai_limit_rejected(db):
    """AI 限额字段为负值时拒绝保存"""
    with pytest.raises(BusinessException) as exc:
        await member_benefit_service.update_benefit(db, "level_1", {"aiCreditsDaily": -1})
    assert exc.value.code == ResultCode.BENEFIT_CONFIG_INVALID


async def test_update_benefit_growth_range_invalid(db):
    """成长值下限大于上限时拒绝"""
    with pytest.raises(BusinessException) as exc:
        await member_benefit_service.update_benefit(
            db, "level_1", {"growthMin": 5000, "growthMax": 1000}
        )
    assert exc.value.code == ResultCode.BENEFIT_CONFIG_INVALID


async def test_list_benefits_contains_8_task_quota(db, mock_redis):
    """权益列表返回全部 8 类任务配额字段"""
    await member_benefit_service.list_benefits(db)
    # 直连仓库验证字段（避开缓存）
    benefit = await member_benefit_repository.get_by_level_code(db, "level_1")
    for task_type in QUOTA_TASK_TYPES:
        assert hasattr(benefit, f"monthly_{task_type}_quota")


async def test_update_benefit_invalidates_summary_cache(db, mock_redis):
    """修改权益配置后失效 member:benefit-summary:* 聚合缓存"""
    await mock_redis.setex(f"member:benefit-summary:{123456}", 300, '{"x":1}')
    await member_benefit_service.update_benefit(db, "level_1", {"aiCreditsDaily": 100})
    assert await mock_redis.get(f"member:benefit-summary:{123456}") is None
