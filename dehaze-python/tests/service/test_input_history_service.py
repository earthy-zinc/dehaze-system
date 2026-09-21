"""图像输入历史记录服务层测试（配额自动清理 / 用户隔离 / LIKE 通配符转义）。

基于真实 MySQL 测试库（db fixture 外部事务回滚，种子零污染）。
配额口径：sys_member_benefit.history_retention 按会员等级，缺失时回退 100。
"""

import uuid

import pytest

from app.models.entity.sys_member import SysMember
from app.models.entity.sys_member_benefit import SysMemberBenefit
from app.service.input_history_service import (
    DEFAULT_HISTORY_RETENTION,
    input_history_service,
)

pytestmark = pytest.mark.api


def _form(**overrides) -> dict:
    data = {
        "originalImageUrl": f"/images/{uuid.uuid4().hex}.jpg",
        "algorithmId": 1,
        "algorithmName": "DCP",
        "processingTime": 1000,
        "status": 1,
        "inputSource": "upload",
    }
    data.update(overrides)
    return data


async def _make_member_with_retention(db, user_id: int, retention: int) -> str:
    """构造会员档案 + 独立等级权益（history_retention 按测试指定）"""
    level_code = f"lvl_{uuid.uuid4().hex[:8]}"
    db.add(
        SysMemberBenefit(
            level_code=level_code,
            level_name="测试等级",
            history_retention=retention,
        )
    )
    db.add(SysMember(user_id=user_id, level_code=level_code))
    await db.flush()
    return level_code


# ===== 配额自动清理 =====


async def test_quota_autocleanup_deletes_oldest_on_overflow(db):
    """配额已满时创建新记录：自动清理最旧一条，总数不超上限"""
    await _make_member_with_retention(db, user_id=9001, retention=2)
    first_id = await input_history_service.create_history(
        db, _form(algorithmName="oldest"), user_id=9001
    )
    await input_history_service.create_history(db, _form(algorithmName="second"), user_id=9001)

    third_id = await input_history_service.create_history(db, _form(), user_id=9001)

    list_vo, total = await input_history_service.list_history(db, user_id=9001, page=1, size=10)
    assert total == 2
    ids = {v["id"] for v in list_vo}
    assert first_id not in ids
    assert third_id in ids


async def test_retention_falls_back_when_no_member(db):
    """无会员档案 → 回退默认保留条数"""
    retention = await input_history_service._get_history_retention(db, user_id=987654)
    assert retention == DEFAULT_HISTORY_RETENTION


async def test_retention_falls_back_when_benefit_missing(db):
    """会员等级无对应权益配置 → 回退默认保留条数"""
    db.add(SysMember(user_id=9002, level_code=f"lvl_{uuid.uuid4().hex[:8]}"))
    await db.flush()
    retention = await input_history_service._get_history_retention(db, user_id=9002)
    assert retention == DEFAULT_HISTORY_RETENTION


async def test_retention_falls_back_when_benefit_retention_is_zero(db):
    """权益存在但 history_retention=0（未配置）→ 回退默认保留条数，而非 0 条上限"""
    level_code = f"lvl_{uuid.uuid4().hex[:8]}"
    db.add(SysMemberBenefit(level_code=level_code, level_name="零配额等级", history_retention=0))
    db.add(SysMember(user_id=9003, level_code=level_code))
    await db.flush()
    retention = await input_history_service._get_history_retention(db, user_id=9003)
    assert retention == DEFAULT_HISTORY_RETENTION


# ===== 用户隔离（越权路径）=====


async def test_get_history_denies_other_user(db):
    history_id = await input_history_service.create_history(db, _form(), user_id=100)

    assert await input_history_service.get_history(db, history_id, user_id=100) is not None
    # 他人查询返回 None（路由层转 A0401，不泄露存在性）
    assert await input_history_service.get_history(db, history_id, user_id=200) is None


async def test_delete_history_is_scoped_to_owner(db):
    history_id = await input_history_service.create_history(db, _form(), user_id=100)

    await input_history_service.delete_history(db, history_id, user_id=200)
    assert await input_history_service.get_history(db, history_id, user_id=100) is not None

    await input_history_service.delete_history(db, history_id, user_id=100)
    assert await input_history_service.get_history(db, history_id, user_id=100) is None


async def test_batch_delete_cross_user_returns_zero_and_keeps_rows(db):
    history_id = await input_history_service.create_history(db, _form(), user_id=100)

    deleted = await input_history_service.batch_delete(db, [history_id], user_id=200)
    assert deleted == 0
    assert await input_history_service.get_history(db, history_id, user_id=100) is not None


async def test_batch_delete_returns_actual_deleted_count(db):
    id1 = await input_history_service.create_history(db, _form(), user_id=300)
    id2 = await input_history_service.create_history(db, _form(), user_id=300)

    deleted = await input_history_service.batch_delete(db, [id1, id2, 999999], user_id=300)
    assert deleted == 2


async def test_clear_history_scoped_to_user(db):
    own_id = await input_history_service.create_history(db, _form(), user_id=100)
    other_id = await input_history_service.create_history(db, _form(), user_id=200)

    cleared = await input_history_service.clear_history(db, user_id=100)
    assert cleared >= 1
    assert await input_history_service.get_history(db, own_id, user_id=100) is None
    assert await input_history_service.get_history(db, other_id, user_id=200) is not None


# ===== 关键词检索 =====


async def test_keyword_like_wildcards_escaped(db):
    """关键词中的 %/_ 应按字面匹配而非通配符（对齐 repository escape_like）"""
    await input_history_service.create_history(db, _form(algorithmName="A%B"), user_id=400)
    await input_history_service.create_history(db, _form(algorithmName="AXB"), user_id=400)

    list_vo, _ = await input_history_service.list_history(
        db, user_id=400, keywords="A%B", page=1, size=10
    )
    names = {v["algorithmName"] for v in list_vo}
    assert names == {"A%B"}


async def test_list_filters_by_status_and_input_source(db):
    await input_history_service.create_history(
        db, _form(status=1, inputSource="upload"), user_id=500
    )
    await input_history_service.create_history(
        db, _form(status=2, inputSource="camera"), user_id=500
    )

    _, total_success = await input_history_service.list_history(
        db, user_id=500, status=1, page=1, size=10
    )
    _, total_failed = await input_history_service.list_history(
        db, user_id=500, status=2, page=1, size=10
    )
    _, total_camera = await input_history_service.list_history(
        db, user_id=500, input_source="camera", page=1, size=10
    )
    assert total_success == 1
    assert total_failed == 1
    assert total_camera == 1
