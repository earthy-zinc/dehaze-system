"""部门服务层测试（真实 MySQL 测试库 + fakeredis）。

对应部门管理测试用例.md：T-DPT-008~014（新增/层级）、T-DPT-018a~024（编辑/循环引用/
移动一致性）、T-DPT-028~035b（删除保护/软删）、T-DPT-042c/047（options 缓存与过滤）、
T-DPT-049（数据权限可见性）、T-DPT-045（排序）。

【错误码契约】按全局统一口径断言：A0401 资源不存在、A0501 数据已存在（同级+含软删）、
A0502 子部门/关联用户拦截、A0503 内置保护（根部门不可删除/修改上级）与循环引用、
A0504 层级超限。API接口.md §4 中 A0234 为历史口径，与三端实现（A0503）不一致，已报 main 决策。
"""

import uuid

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.dept_service import ROOT_DEPT_ID, dept_service
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.requires_db

# 种子部门（config/sql/data/sys_dept.sql）：1=重庆邮电大学(根) 2=软件工程学院 3=计算机学院
ROOT_ID = ROOT_DEPT_ID
CS_ID = 3


def _unique_name(prefix: str = "svc部门") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def _create(db, mock_redis, *, parent_id=ROOT_ID, name=None, **overrides) -> int:
    data = {"name": name or _unique_name(), "parentId": parent_id, "status": 1, "sort": 1}
    data.update(overrides)
    return await dept_service.create_dept(db, mock_redis, data)


async def _get_dept(db, dept_id):
    from app.repository.dept_repository import dept_repository

    return await dept_repository.get_by_id(db, dept_id)


# ===== 新增（T-DPT-008/009/013/014） =====


async def test_create_dept_persists_tree_path_chain(db, mock_redis):
    parent_id = await _create(db, mock_redis, parent_id=ROOT_ID)
    child_id = await _create(db, mock_redis, parent_id=parent_id)

    parent = await _get_dept(db, parent_id)
    child = await _get_dept(db, child_id)
    assert parent is not None
    assert child is not None
    assert parent.tree_path == "0,1"
    # 不变量：子.tree_path == 父.tree_path + "," + 父.id
    assert child.tree_path == f"{parent.tree_path},{parent_id}"


async def test_create_dept_name_duplicate_rejected_a0501(db, mock_redis):
    """暴露：service 抛裸 BusinessException → B0001，文档要求 A0501（数据已存在）"""
    name = _unique_name()
    await _create(db, mock_redis, name=name)
    with pytest.raises(BusinessException) as ei:
        await _create(db, mock_redis, name=name)
    assert ei.value.code == ResultCode.DATA_EXISTS, (
        f"名称重复应返回 A0501，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_create_dept_name_unique_within_same_parent_only(db, mock_redis):
    """同级唯一（对齐 Go 实现 + 文档 T-DPT-009"同级部门名称已存在"）：
    不同父部门下同名允许创建"""
    name = _unique_name()
    await _create(db, mock_redis, parent_id=ROOT_ID, name=name)
    sibling_id = await _create(db, mock_redis, parent_id=CS_ID, name=name)
    assert sibling_id > 0


async def test_create_dept_name_blocked_after_soft_delete_same_parent(db, mock_redis):
    """唯一性含软删行（对齐文档 T-DPT-035b"唯一性校验含已删除记录"与
    dept-backend 重构实现）：同级软删后同名称重建被拒（A0501）"""
    name = _unique_name()
    dept_id = await _create(db, mock_redis, parent_id=ROOT_ID, name=name)
    await dept_service.delete_depts(db, mock_redis, [dept_id])

    with pytest.raises(BusinessException) as ei:
        await _create(db, mock_redis, parent_id=ROOT_ID, name=name)
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_dept_parent_not_found_rejected_a0401(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await _create(db, mock_redis, parent_id=99999999)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND, (
        f"父部门不存在应返回 A0401，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_create_dept_depth_5_ok_and_6_rejected(db, mock_redis):
    """层级边界：根(1)=第1级，c1..c4=第2~5级；第5级下新增第6级被拒（A0504）"""
    chain = []
    parent = ROOT_ID
    for _ in range(4):
        parent = await _create(db, mock_redis, parent_id=parent)
        chain.append(parent)

    fifth = await _get_dept(db, chain[-1])
    assert fifth is not None
    assert len(fifth.tree_path.split(",")) == 5, "c4 应为第 5 级"

    with pytest.raises(BusinessException) as ei:
        await _create(db, mock_redis, parent_id=chain[-1])
    assert ei.value.code == ResultCode.DATA_BIND_EXISTS
    assert "部门层级不能超过5级" in ei.value.message


# ===== 编辑（T-DPT-018/018a/019/020/021/022/023/024） =====


async def test_update_dept_persists_changes(db, mock_redis):
    dept_id = await _create(db, mock_redis)
    await dept_service.update_dept(
        db, mock_redis, dept_id, {"name": "更名部门", "sort": 9, "status": 0}
    )
    dept = await _get_dept(db, dept_id)
    assert dept is not None
    assert dept.name == "更名部门"
    assert dept.sort == 9
    assert dept.status == 0


async def test_update_dept_name_conflict_rejected_a0501(db, mock_redis):
    """同级名称冲突拒绝（A0501 数据已存在）"""
    other_id = await _create(db, mock_redis)
    dept_id = await _create(db, mock_redis)
    other = await _get_dept(db, other_id)
    assert other is not None
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(
            db, mock_redis, dept_id, {"name": other.name, "parentId": ROOT_ID}
        )
    assert ei.value.code == ResultCode.DATA_EXISTS, (
        f"名称冲突应返回 A0501，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_update_dept_parent_not_found_rejected_a0401(db, mock_redis):
    """父部门不存在拒绝（A0401 资源不存在）"""
    dept_id = await _create(db, mock_redis)
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(db, mock_redis, dept_id, {"parentId": 99999999})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND, (
        f"父部门不存在应返回 A0401，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_update_dept_self_as_parent_rejected_a0503(db, mock_redis):
    """T-DPT-020：不能选择自身作为上级（A0503 操作不允许）"""
    dept_id = await _create(db, mock_redis)
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(db, mock_redis, dept_id, {"parentId": dept_id})
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW, (
        f"自身作为上级应返回 A0503，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_update_dept_cycle_detection_rejected_a0503(db, mock_redis):
    """T-DPT-021/022：A→B→C 链，把 A 移到 C 下形成环必须拒绝（A0503）"""
    a_id = await _create(db, mock_redis, parent_id=ROOT_ID)
    b_id = await _create(db, mock_redis, parent_id=a_id)
    c_id = await _create(db, mock_redis, parent_id=b_id)

    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(db, mock_redis, a_id, {"parentId": c_id})
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW, (
        f"循环引用应返回 A0503，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_update_root_dept_parent_protected_a0503(db, mock_redis):
    """T-DPT-023：根部门不可修改上级。三端实现（Java/Go/Python）均为 A0503
    （OPERATION_NOT_ALLOW），文档 API接口.md §4 写 A0234——文档与实现矛盾，
    建议按三端多数派修订文档（角色模块 A0233→A0503 先例），已报 main 决策"""
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(
            db, mock_redis, ROOT_ID, {"name": "重庆邮电大学", "parentId": CS_ID}
        )
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW, (
        f"根部门修改上级应被拒绝（A0503），实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_update_moved_dept_depth_limit_a0504(db, mock_redis):
    """T-DPT-018a：第2级部门移动到第5级下 → 移动后成第6级，拒绝（A0504）"""
    chain = []
    parent = ROOT_ID
    for _ in range(4):
        parent = await _create(db, mock_redis, parent_id=parent)
        chain.append(parent)
    level2_id = await _create(db, mock_redis, parent_id=ROOT_ID)

    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(db, mock_redis, level2_id, {"parentId": chain[-1]})
    assert ei.value.code == ResultCode.DATA_BIND_EXISTS


async def test_update_moved_subtree_keeps_tree_path_consistent(db, mock_redis):
    """移动非叶子部门后级联更新子部门 tree_path。

    不变量：移动后任意子部门 tree_path == 其父.tree_path + "," + 父.id。
    Java SysDeptServiceImpl.updateDept 缺失级联（三端缺口，Python 已修复，Java/Go 待对齐）。"""
    a_id = await _create(db, mock_redis, parent_id=ROOT_ID)
    b_id = await _create(db, mock_redis, parent_id=a_id)
    p_id = await _create(db, mock_redis, parent_id=ROOT_ID)

    await dept_service.update_dept(db, mock_redis, a_id, {"parentId": p_id})

    a = await _get_dept(db, a_id)
    b = await _get_dept(db, b_id)
    assert a is not None
    assert b is not None
    assert a.parent_id == p_id
    assert a.tree_path == f"0,1,{p_id}"
    assert b.tree_path == f"{a.tree_path},{a_id}", (
        f"移动部门后子部门 tree_path 未级联更新："
        f"B.tree_path={b.tree_path!r}，期望 {a.tree_path},{a_id!r}"
    )


async def test_update_dept_not_found_rejected_a0401(db, mock_redis):
    """编辑不存在的部门拒绝（A0401 资源不存在）"""
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(db, mock_redis, 99999999, {"name": "幽灵"})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND, (
        f"编辑不存在部门应返回 A0401，实际 {ei.value.code}（{ei.value.message}）"
    )


# ===== 删除（T-DPT-028~032/035a） =====


async def test_delete_dept_ok_and_filtered_from_list(db, mock_redis):
    dept_id = await _create(db, mock_redis)
    await dept_service.delete_depts(db, mock_redis, [dept_id])

    tree = await dept_service.get_dept_list(db)
    ids = []

    def _collect(nodes):
        for n in nodes:
            ids.append(n["id"])
            _collect(n.get("children") or [])

    _collect(tree)
    assert dept_id not in ids
    remaining = await _get_dept(db, dept_id)
    assert remaining is None or remaining.deleted != 0


async def test_delete_dept_with_children_rejected_a0502(db, mock_redis):
    parent_id = await _create(db, mock_redis)
    await _create(db, mock_redis, parent_id=parent_id)

    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(db, mock_redis, [parent_id])
    assert ei.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    assert "子部门" in ei.value.message


async def test_delete_dept_with_users_rejected_a0502(db, mock_redis):
    """种子用户均挂 software 学院（id=2）下"""
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(db, mock_redis, [2])
    assert ei.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    assert "用户" in ei.value.message


async def test_delete_root_dept_rejected_a0503(db, mock_redis):
    """T-DPT-031：根部门不可删除。三端实现均为 A0503（OPERATION_NOT_ALLOW），
    文档 API接口.md §4 写 A0234——文档与实现矛盾，随根部门保护口径一并报 main 决策"""
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(db, mock_redis, [ROOT_ID])
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW, (
        f"删除根部门应被拒绝（A0503），实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_delete_dept_not_found_rejected_a0401(db, mock_redis):
    """删除不存在的部门拒绝（A0401，整体失败不部分删除）"""
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(db, mock_redis, [99999999])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND, (
        f"删除不存在部门应返回 A0401，实际 {ei.value.code}（{ei.value.message}）"
    )


async def test_delete_mixed_ids_rejects_instead_of_partial_delete(db, mock_redis):
    """【暴露缺陷】批量删除混合存在/不存在 ID：Java 为整体失败（getById null →
    A0401 事务回滚），Python 静默忽略不存在 ID 并删除其余部门 → 误删风险。
    契约（对齐 Java）：任一 ID 不存在 → A0401 整体拒绝，不发生部分删除。"""
    x_id = await _create(db, mock_redis)

    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(db, mock_redis, [x_id, 99999999])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND

    x = await _get_dept(db, x_id)
    assert x is not None
    assert x.deleted == 0, "混合 ID 批量删除发生部分删除（静默吞掉不存在的 ID）"


# ===== 下拉选项 / 缓存 / 数据权限 / 排序（T-DPT-042c/047/049/045/027） =====


async def test_options_cache_invalidated_after_create(db, mock_redis):
    """T-DPT-027/035：options 有缓存（dept:options，TTL 1h），写操作必须失效"""
    before = await dept_service.get_dept_options(db, mock_redis)
    assert before, "种子部门应非空"

    new_id = await _create(db, mock_redis)
    after = await dept_service.get_dept_options(db, mock_redis)

    def _ids(nodes):
        out = []
        for n in nodes:
            out.append(n["value"])
            out.extend(_ids(n.get("children") or []))
        return out

    assert new_id in _ids(after), "新增部门后 options 未反映（缓存未失效）"


async def test_options_filters_disabled_depts(db, mock_redis):
    """T-DPT-042c：options 仅含启用部门"""
    disabled_id = await _create(db, mock_redis, status=0)

    options = await dept_service.get_dept_options(db, mock_redis)

    def _ids(nodes):
        out = []
        for n in nodes:
            out.append(n["value"])
            out.extend(_ids(n.get("children") or []))
        return out

    assert disabled_id not in _ids(options)


async def test_options_dept_scope_visibility(db, mock_redis):
    """T-DPT-049：data_scope=1（本部门及子部门）用户应仅见本部门子树。
    【暴露缺陷】当前行为：scope 过滤后部门 2/3 的父节点（根 1）不在结果集，
    树构建时既非根（parent_id!=0）又找不到父 → 子树被静默丢弃 → 返回空列表，
    部门管理员下拉选项完全不可用。修复方向：父不在结果集的节点提升为根。"""
    child_id = await _create(db, mock_redis, parent_id=2)
    user = make_user_context(5, username="u", dept_id=2, data_scope=1)

    options = await dept_service.get_dept_options(db, mock_redis, current_user=user)

    def _ids(nodes):
        out = []
        for n in nodes:
            out.append(n["value"])
            out.extend(_ids(n.get("children") or []))
        return out

    visible = set(_ids(options))
    assert 2 in visible, (
        f"数据权限 scope=1 的部门管理员 options 为空（{sorted(visible)}）："
        "数据权限过滤后父节点丢失，树构建静默丢弃本部门子树"
    )
    assert child_id in visible
    assert CS_ID not in visible, "数据权限未隔离：可见了兄弟部门"
    assert ROOT_ID not in visible, "数据权限未隔离：可见了根部门"


async def test_children_ids_excludes_sibling_dept(db, mock_redis):
    """【暴露缺陷】get_children_ids 的 tree_path LIKE 前缀匹配横向越权。

    dept 2 的 tree_path="0,1"，子部门路径为 "0,1,2"。实现用
    LIKE '0,1%' → 同时命中兄弟部门 3（tree_path 同为 "0,1"）及其整个子树。
    data_scope=1 用户的 children_ids 混入兄弟部门 → 行级数据权限横向越权。
    修复方向：条件应为 tree_path == "{tp},{id}" OR LIKE "{tp},{id},%"。"""
    from app.repository.dept_repository import dept_repository

    child_of_2 = await _create(db, mock_redis, parent_id=2)
    child_of_3 = await _create(db, mock_redis, parent_id=CS_ID)

    ids = await dept_repository.get_children_ids(db, 2)

    assert 2 in ids
    assert child_of_2 in ids
    assert CS_ID not in ids, f"get_children_ids(2) 混入兄弟部门 3：{ids}"
    assert child_of_3 not in ids, f"get_children_ids(2) 混入兄弟部门子树：{ids}"


async def test_list_same_level_sorted_by_sort_asc(db, mock_redis):
    """T-DPT-045：同级按 sort 升序"""
    first_id = await _create(db, mock_redis, parent_id=ROOT_ID, sort=1)
    second_id = await _create(db, mock_redis, parent_id=ROOT_ID, sort=5)
    third_id = await _create(db, mock_redis, parent_id=ROOT_ID, sort=3)

    tree = await dept_service.get_dept_list(db)
    root = next(n for n in tree if n["id"] == ROOT_ID)
    child_ids = [c["id"] for c in root["children"]]

    order = [i for i in child_ids if i in (first_id, second_id, third_id)]
    assert order == [first_id, third_id, second_id]
