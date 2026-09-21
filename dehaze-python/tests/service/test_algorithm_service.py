"""算法管理服务测试（真实 MySQL 测试库 + SAVEPOINT 回滚）。

覆盖：三端统一状态机白名单流转、可删除状态校验与软删语义（deleted=行id）、
名称唯一性（软删行不占键位可重建）、版本 is_active 单活跃管理与防重复回滚、
下拉选项软删过滤（推荐轮登记待办核实）。

遵循 dehaze 测试规范：仅依赖 db fixture 与 mock_redis（autouse），
只断言业务结果，命名 test_功能_场景。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import AlgorithmStatus, algorithm_repository
from app.service.algorithm_service import algorithm_service

pytestmark = pytest.mark.requires_db


async def _create_algorithm(
    db, name: str, status: int = AlgorithmStatus.DRAFT, parent_id: int = 0
) -> SysAlgorithm:
    algo = SysAlgorithm(parent_id=parent_id, type="TEST", name=name, status=status)
    return await algorithm_repository.create(db, algo)


# ===== 状态机（三端统一白名单）=====


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (1, 2),  # 草稿→测试中
        (2, 3),  # 测试中→待审核
        (2, 1),  # 测试中→草稿
        (3, 4),  # 待审核→已发布
        (3, 2),  # 待审核→测试中（驳回流转）
        (4, 5),  # 已发布→已停用
        (4, 6),  # 已发布→已归档
        (5, 4),  # 已停用→已发布（重新上架）
        (5, 6),  # 已停用→已归档
    ],
)
async def test_update_status_allowed_transitions(db, current, target):
    """白名单内流转全部成功并落库"""
    algo = await _create_algorithm(db, f"状态机-合法-{current}-{target}", status=current)

    await algorithm_service.update_status(db, algo.id, target)

    refreshed = await algorithm_repository.get_by_id(db, algo.id)
    assert refreshed is not None
    assert refreshed.status == target


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (1, 4),  # 草稿→已发布（跳过审核）
        (1, 5),  # 草稿→已停用
        (1, 6),  # 草稿→已归档
        (2, 4),  # 测试中→已发布
        (2, 5),  # 测试中→已停用
        (3, 5),  # 待审核→已停用
        (3, 1),  # 待审核→草稿
        (4, 2),  # 已发布→测试中
        (5, 2),  # 已停用→测试中
        (6, 1),  # 已归档→草稿（终态）
        (6, 4),  # 已归档→已发布（终态）
    ],
)
async def test_update_status_rejects_illegal_transitions(db, current, target):
    """白名单外流转全部拒绝（A0502），状态不被篡改"""
    algo = await _create_algorithm(db, f"状态机-非法-{current}-{target}", status=current)

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_service.update_status(db, algo.id, target)

    assert exc_info.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    refreshed = await algorithm_repository.get_by_id(db, algo.id)
    assert refreshed is not None
    assert refreshed.status == current


@pytest.mark.parametrize("invalid_status", [0, 7, 99, -1])
async def test_update_status_rejects_invalid_value(db, invalid_status):
    """无效状态值不在任何白名单内，拒绝"""
    algo = await _create_algorithm(db, f"状态机-无效值-{invalid_status}", status=1)

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_service.update_status(db, algo.id, invalid_status)

    assert exc_info.value.code == ResultCode.DATA_STATE_NOT_ALLOW


async def test_update_status_missing_algorithm_rejected(db):
    with pytest.raises(BusinessException) as exc_info:
        await algorithm_service.update_status(db, 999999999, 2)

    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 删除（状态校验 + 软删语义）=====


@pytest.mark.parametrize("status", [2, 3, 4])
async def test_delete_algorithms_rejects_non_deletable_status(db, status):
    """测试中/待审核/已发布不允许删除（A0502），消息携带算法名"""
    algo = await _create_algorithm(db, f"删除-禁删-{status}", status=status)

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_service.delete_algorithms(db, [algo.id])

    assert exc_info.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    assert algo.name in exc_info.value.message
    # 数据未被删除
    assert await algorithm_repository.get_by_id(db, algo.id) is not None


async def test_delete_algorithms_soft_delete_marks_deleted_with_row_id(db):
    """草稿算法删除为软删：deleted=行 id（uk 治理新语义），活跃查询不可见"""
    algo = await _create_algorithm(db, "删除-软删验证", status=1)

    count = await algorithm_service.delete_algorithms(db, [algo.id])

    assert count >= 1
    assert await algorithm_repository.get_by_id(db, algo.id) is None
    # 软删行仍在库中，deleted=行 id
    stmt = await db.execute(
        __import__("sqlalchemy").text("SELECT deleted FROM sys_algorithm WHERE id = :id"),
        {"id": algo.id},
    )
    assert (stmt.scalar_one()) == algo.id


async def test_delete_algorithms_cascades_descendants(db):
    """删除父算法级联软删所有子孙"""
    parent = await _create_algorithm(db, "删除-父", status=1)
    child = await _create_algorithm(db, "删除-子", status=1, parent_id=parent.id)
    grandchild = await _create_algorithm(db, "删除-孙", status=1, parent_id=child.id)

    count = await algorithm_service.delete_algorithms(db, [parent.id])

    assert count == 3
    assert await algorithm_repository.get_by_id(db, parent.id) is None
    assert await algorithm_repository.get_by_id(db, child.id) is None
    assert await algorithm_repository.get_by_id(db, grandchild.id) is None


async def test_delete_algorithms_missing_id_rejected(db):
    with pytest.raises(BusinessException) as exc_info:
        await algorithm_service.delete_algorithms(db, [999999999])

    assert exc_info.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 名称唯一性（A0501 + 软删行不占键位）=====


async def test_create_algorithm_rejects_duplicate_name(db):
    """同名算法新建被拒绝（A0501）"""
    await _create_algorithm(db, "名称-重复验证")

    with pytest.raises(BusinessException) as exc_info:
        await algorithm_service.create_algorithm(db, {"name": "名称-重复验证", "type": "TEST"})

    assert exc_info.value.code == ResultCode.DATA_EXISTS


async def test_create_algorithm_allows_rebuild_after_soft_delete(db):
    """软删后同名可重建（软删行不占唯一键位）"""
    algo = await _create_algorithm(db, "名称-删后重建")
    await algorithm_service.delete_algorithms(db, [algo.id])

    rebuilt_id = await algorithm_service.create_algorithm(
        db, {"name": "名称-删后重建", "type": "TEST"}
    )

    assert rebuilt_id != algo.id


# ===== 版本管理（is_active 单活跃 + 防重复回滚）=====


async def test_create_version_marks_new_version_active_only(db):
    """新增版本后仅新版本 is_active=1，旧活跃版本被置非活跃，主表版本号更新"""
    algo = await _create_algorithm(db, "版本-单活跃")
    await algorithm_service.create_version(db, algo.id, "v1.0.0")
    await algorithm_service.create_version(db, algo.id, "v1.1.0")

    versions = await algorithm_repository.list_versions(db, algo.id)
    active = [v for v in versions if v.is_active == 1]
    assert len(active) == 1
    assert active[0].version == "v1.1.0"

    refreshed = await algorithm_repository.get_by_id(db, algo.id)
    assert refreshed is not None
    assert refreshed.version == "v1.1.0"


async def test_create_version_rejects_duplicate_version(db):
    algo = await _create_algorithm(db, "版本-重复号")
    await algorithm_service.create_version(db, algo.id, "v2.0.0")

    with pytest.raises(BusinessException, match=r"v2\.0\.0"):
        await algorithm_service.create_version(db, algo.id, "v2.0.0")


async def test_rollback_version_switches_active_and_main_table(db):
    """回滚切换 is_active 并回写主表版本号"""
    algo = await _create_algorithm(db, "版本-回滚")
    await algorithm_service.create_version(db, algo.id, "v1.0.0")
    await algorithm_service.create_version(db, algo.id, "v1.1.0")

    old_version = next(
        v for v in await algorithm_repository.list_versions(db, algo.id) if v.version == "v1.0.0"
    )
    await algorithm_service.rollback_version(db, algo.id, old_version.id)

    refreshed = await algorithm_repository.get_by_id(db, algo.id)
    assert refreshed is not None
    assert refreshed.version == "v1.0.0"
    active = [v for v in await algorithm_repository.list_versions(db, algo.id) if v.is_active == 1]
    assert len(active) == 1
    assert active[0].version == "v1.0.0"


async def test_rollback_version_rejects_already_active(db):
    """重复回滚到当前活跃版本被拒绝"""
    algo = await _create_algorithm(db, "版本-重复回滚")
    await algorithm_service.create_version(db, algo.id, "v1.0.0")

    active = next(
        v for v in await algorithm_repository.list_versions(db, algo.id) if v.is_active == 1
    )
    with pytest.raises(BusinessException, match="无需回滚"):
        await algorithm_service.rollback_version(db, algo.id, active.id)


async def test_rollback_version_rejects_foreign_version(db):
    """回滚不属于该算法的版本被拒绝"""
    algo = await _create_algorithm(db, "版本-外键校验")
    other = await _create_algorithm(db, "版本-外键校验-其他")
    await algorithm_service.create_version(db, other.id, "v9.9.9")
    foreign = next(
        v for v in await algorithm_repository.list_versions(db, other.id) if v.is_active == 1
    )

    with pytest.raises(BusinessException, match="不属于该算法"):
        await algorithm_service.rollback_version(db, algo.id, foreign.id)


# ===== 下拉选项软删过滤（推荐轮登记待办核实）=====


async def test_get_algorithm_options_excludes_soft_deleted_and_unpublished(db):
    """下拉选项仅含已发布活跃算法：软删行与未发布行均不出现"""
    published = await _create_algorithm(db, "选项-已发布", status=4)
    draft = await _create_algorithm(db, "选项-草稿", status=1)
    soft_deleted = await _create_algorithm(db, "选项-软删", status=4)
    # 已发布需先停用再删除（DELETABLE_STATUSES）
    await algorithm_service.update_status(db, soft_deleted.id, 5)
    await algorithm_service.delete_algorithms(db, [soft_deleted.id])

    options = await algorithm_service.get_algorithm_options(db)
    option_ids = {opt["value"] for opt in options}

    assert published.id in option_ids
    assert draft.id not in option_ids
    assert soft_deleted.id not in option_ids


async def test_algorithm_list_excludes_soft_deleted(db):
    """算法树/扁平列表不返回软删行（ORM 全局软删过滤）"""
    algo = await _create_algorithm(db, "列表-软删过滤")
    await algorithm_service.delete_algorithms(db, [algo.id])

    flat = await algorithm_service.list_all_algorithms(db)
    assert all(item["id"] != algo.id for item in flat)

    tree = await algorithm_service.get_algorithm_list(db)
    assert all(item["id"] != algo.id for item in tree)
