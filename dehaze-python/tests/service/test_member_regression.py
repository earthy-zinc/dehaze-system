"""
会员管理模块回归与边界强化测试（真实 MySQL 测试库 dehaze_test + SAVEPOINT 回滚）。

聚焦历史 bug 回归与任务书指定场景：
- get_or_init_member 三段式回归（活跃会员数据不被清零、幂等、软删复活）
- 成长值→等级迁移边界（阈值恰好达成/差 1）
- 签到断签重置、同日唯一索引兜底、成长值规则字典缺键回退
- 管理员调整成长值审计（operator_id）与对抗性脏语料 reason
- 关键字模糊搜索 LIKE 通配符转义不变量
- 配额扣减不变量（used ≤ quota）

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

from datetime import date, timedelta

import pytest
from sqlalchemy import func, select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_member import QUOTA_TASK_TYPES
from app.models.entity.sys_member_sign_in import SysMemberSignIn
from app.repository.dict_repository import dict_repository
from app.repository.member_benefit_repository import member_benefit_repository
from app.repository.member_growth_log_repository import member_growth_log_repository
from app.repository.member_repository import member_repository
from app.service.member.growth_service import (
    SIGN_IN_BASE_GROWTH_DEFAULT,
    member_growth_service,
)
from app.service.member.member_service import member_service
from app.service.member.quota_service import member_quota_service

pytestmark = pytest.mark.requires_db

USER_ID = 1006001


async def _setup_benefit(db, level_code: str, *, growth_min: int, growth_max: int, quota: int = 50):
    benefit = await member_benefit_repository.get_by_level_code(db, level_code)
    assert benefit is not None
    benefit.growth_min = growth_min
    benefit.growth_max = growth_max
    for task_type in QUOTA_TASK_TYPES:
        setattr(benefit, f"monthly_{task_type}_quota", quota)
    await db.flush()
    return benefit


async def _setup_member(
    db,
    user_id: int = USER_ID,
    *,
    level_code: str = "level_0",
    growth_value: int = 0,
    level_source: str = "growth",
    expire_time=None,
):
    member = await member_repository.get_or_init_member(db, user_id)
    member.level_code = level_code
    member.level_source = level_source
    member.growth_value = growth_value
    member.expire_time = expire_time
    await db.flush()
    return member


# ===================== get_or_init_member 三段式回归 =====================


async def test_get_or_init_member_active_member_not_cleared(db):
    """历史 bug 回归：原 upsert 实现会把活跃会员等级/成长值/配额清零，三段式必须完整保留"""
    member = await _setup_member(
        db,
        level_code="level_2",
        growth_value=8000,
        level_source="purchase",
        expire_time=None,
    )
    member.monthly_dehaze_used = 7
    member.total_consumption = 12345
    await db.flush()

    again = await member_repository.get_or_init_member(db, USER_ID)

    assert again.level_code == "level_2"
    assert again.level_source == "purchase"
    assert again.growth_value == 8000
    assert again.monthly_dehaze_used == 7
    assert again.total_consumption == 12345


async def test_get_or_init_member_repeated_call_same_row(db):
    """重复初始化幂等：不产生第二行会员记录"""
    await _setup_member(db, growth_value=100)
    await member_repository.get_or_init_member(db, USER_ID)

    count = (
        await db.execute(
            select(func.count())
            .select_from(member_repository.model.__table__)
            .where(member_repository.model.user_id == USER_ID)
        )
    ).scalar()
    assert count == 1


async def test_get_or_init_member_revive_soft_deleted_keeps_total_consumption(db):
    """软删会员复活：重置等级/成长值/配额，但保留 total_consumption 便于追溯"""
    member = await _setup_member(
        db, level_code="level_1", growth_value=5000, level_source="purchase"
    )
    member.total_consumption = 9999
    member.deleted = 1
    await db.flush()

    revived = await member_repository.get_or_init_member(db, USER_ID)

    assert revived.deleted == 0
    assert revived.level_code == "level_0"
    assert revived.level_source == "growth"
    assert revived.growth_value == 0
    for task_type in QUOTA_TASK_TYPES:
        assert getattr(revived, f"monthly_{task_type}_quota") == 0
        assert getattr(revived, f"monthly_{task_type}_used") == 0
    assert revived.total_consumption == 9999


# ===================== 成长值→等级迁移边界 =====================


@pytest.mark.parametrize(
    ("growth_before", "delta", "expected_level"),
    [
        (4998, 1, "level_1"),  # 差 1 不升级（4999 < level_2 下限 5000）
        (4999, 1, "level_2"),  # 恰好达成下限 → 升级
        (19999, 1, "level_3"),  # level_2 上限 19999，+1 到 20000 恰达 SVIP 下限（§2.1 SVIP ≥20000）
        (20000, 1, "level_3"),  # 恰好达成 level_3 下限 → 跨级升级
    ],
    ids=["below_threshold", "exact_threshold", "level2_upper_boundary", "cross_level"],
)
async def test_growth_threshold_boundary_level_migration(db, growth_before, delta, expected_level):
    """成长值阈值边界：等级迁移严格按 [growth_min, growth_max] 闭区间判定"""
    await _setup_benefit(db, "level_1", growth_min=1000, growth_max=4999)
    await _setup_benefit(db, "level_2", growth_min=5000, growth_max=19999)
    await _setup_benefit(db, "level_3", growth_min=20000, growth_max=0)
    await _setup_member(db, level_code="level_1", growth_value=growth_before)

    await member_service.adjust_growth(
        db, USER_ID, {"changeValue": delta, "reason": "阈值边界"}, operator_id=2
    )
    await db.flush()

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member is not None
    assert member.growth_value == growth_before + delta
    assert member.level_code == expected_level


# ===================== 签到 =====================


async def test_sign_in_streak_resets_after_gap(db):
    """断签重置：昨日未签到，今日签到 continuous_days 重置为 1（T-MM-072）"""
    await _setup_member(db, growth_value=30)
    today = date.today()
    # 只签了 3 天前，昨天与今天之间的链路已断
    db.add(
        SysMemberSignIn(
            user_id=USER_ID,
            sign_date=today - timedelta(days=3),
            continuous_days=1,
            growth_value=3,
        )
    )
    await db.flush()

    result = await member_growth_service.sign_in(db, USER_ID)

    assert result["continuousDays"] == 1
    assert result["bonusGrowth"] == 0


async def test_sign_in_unique_index_blocks_same_day_duplicate(db):
    """同日双签由 uk_user_sign_date 唯一索引兜底（T-MM-074），预检被绕过时也不产生重复数据"""
    await _setup_member(db)
    db.add(
        SysMemberSignIn(
            user_id=USER_ID,
            sign_date=date.today(),
            continuous_days=1,
            growth_value=3,
        )
    )
    await db.flush()

    from sqlalchemy.exc import IntegrityError

    db.add(
        SysMemberSignIn(
            user_id=USER_ID,
            sign_date=date.today(),
            continuous_days=2,
            growth_value=3,
        )
    )
    with pytest.raises(IntegrityError):
        await db.flush()
    await db.rollback()  # 回滚到 SAVEPOINT，恢复 session 可用


async def test_sign_in_dict_missing_key_falls_back_to_default(db, mock_redis):
    """成长值规则字典缺键回退设计默认值，不阻断签到（T-MM-069 兜底分支）"""
    from app.service.dict_service import _invalidate_dict_value_cache

    await _setup_member(db)
    item = await dict_repository.get_by_type_code_and_name(
        db, "member_growth_rules", "sign_in_value"
    )
    await db.delete(item)
    await db.flush()
    await _invalidate_dict_value_cache(mock_redis, "member_growth_rules")

    result = await member_growth_service.sign_in(db, USER_ID)

    assert result["growthValue"] == SIGN_IN_BASE_GROWTH_DEFAULT
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member is not None
    assert member.growth_value == SIGN_IN_BASE_GROWTH_DEFAULT


# ===================== 管理员调整成长值：审计 + 脏语料 =====================


async def test_adjust_growth_audit_log_operator_and_dirty_reason(db):
    """管理员调整成长值：流水记录 operator_id（T-MM-024/T-MM-029）；
    对抗性脏语料 reason（emoji/零宽字符/CRLF/全半角混排）原样入流水不崩"""
    await _setup_member(db, level_code="level_1", growth_value=1500)
    dirty_reason = (
        "审计🚀测试😀\u200b\u200b全角ＡＢＣ１２３半角abc123\r\n换行Tab\t混排" + "长" * 150
    )
    await member_service.adjust_growth(
        db, USER_ID, {"changeValue": 50, "reason": dirty_reason}, operator_id=2
    )
    await db.flush()

    logs, _ = await member_growth_log_repository.get_page(db, USER_ID, 1, 10)
    target = next(log for log in logs if log.change_type == "admin_adjust")
    assert target.change_value == 50
    assert target.balance == 1550
    assert target.operator_id == 2
    assert target.reason == dirty_reason


async def test_adjust_growth_rejects_zero_change(db):
    """变动值为 0 直接拒绝（无意义的流水不应落库）"""
    await _setup_member(db, growth_value=100)
    with pytest.raises(BusinessException) as exc:
        await member_service.adjust_growth(
            db, USER_ID, {"changeValue": 0, "reason": "测试"}, operator_id=2
        )
    assert exc.value.code == ResultCode.PARAM_ERROR
    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member is not None
    assert member.growth_value == 100


# ===================== 关键字模糊搜索 LIKE 转义不变量 =====================


async def test_get_page_keywords_like_wildcards_escaped(db):
    """关键字含 LIKE 通配符（% _）与引号时按字面匹配，不膨胀结果集、不抛 500"""
    await _setup_member(db)

    query = {
        "pageNum": 1,
        "pageSize": 10,
        "keywords": None,
        "levelCode": None,
        "status": None,
        "expireTimeStart": None,
        "expireTimeEnd": None,
        "growthMin": None,
        "growthMax": None,
    }
    all_page = await member_service.list_paged_members(db, dict(query))
    total_all = all_page["total"]
    assert total_all >= 1

    for dirty in ["%", "_", "%_%", "'", '"', "\\", "％", "１００％"]:
        query["keywords"] = dirty
        page = await member_service.list_paged_members(db, dict(query))
        # 通配符未转义时 % 会匹配全部；转义正确则按字面匹配，结果必然少于全量
        assert page["total"] < total_all, f"keywords={dirty!r} 疑似未转义（命中全量）"
        assert isinstance(page["list"], list)


# ===================== 配额扣减不变量 =====================


async def test_quota_invariant_used_not_exceed_quota_after_cycle(db):
    """扣减-归还循环后 used ≤ quota 恒成立，权益概览 remaining 不为负"""
    await _setup_benefit(db, "level_1", growth_min=1000, growth_max=4999, quota=5)
    member = await _setup_member(db, level_code="level_1", growth_value=1500)
    # 配额快照仅随等级联动路径刷新（adjust_level/on_order_paid/月度重置等），
    # 直改 level_code 绕过联动时需同步快照，与真实履约后的会员状态对齐
    for task_type in QUOTA_TASK_TYPES:
        setattr(member, f"monthly_{task_type}_quota", 5)
    await db.flush()

    for _ in range(5):
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "dehaze")
    with pytest.raises(BusinessException) as exc:
        await member_quota_service.check_and_deduct_quota(db, USER_ID, "dehaze")
    assert exc.value.code == ResultCode.QUOTA_EXCEEDED
    await member_quota_service.restore_quota(db, USER_ID, "dehaze")
    await db.flush()

    member = await member_repository.get_by_user_id(db, USER_ID)
    assert member is not None
    assert member.monthly_dehaze_quota == 5
    assert member.monthly_dehaze_used == 4
    assert member.monthly_dehaze_used <= member.monthly_dehaze_quota

    summary = await member_service.get_benefit_summary(db, USER_ID)
    dehaze_detail = next(
        d for d in summary["imageCategory"]["details"] if d["taskType"] == "dehaze"
    )
    assert dehaze_detail["remaining"] == 1
