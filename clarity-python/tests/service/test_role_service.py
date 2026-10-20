"""角色服务层测试（真实 MySQL 测试库 + fakeredis，覆盖软删唯一性/内置角色保护/缓存失效）。

对应角色管理测试用例.md：T-RM-009（编码唯一性含软删行）、T-RM-017（编码只读）、
T-RM-021/028/032/038（缓存刷新）、T-RM-024/031（内置角色保护）、T-RM-025（关联用户保护）、
T-RM-033~037（菜单分配）、T-RM-046（选项列表可见性）。
"""

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import run_after_commit_callbacks
from app.service.role_service import role_service
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.requires_db

# 预置角色 ID（config/sql/data 种子数据，与 SDK 测试 constants 对齐）
ROOT_ROLE_ID = 1
ADMIN_ROLE_ID = 2
USER_ROLE_ID = 5  # 预置用户（user/vip1/vip2/svip）均关联此角色

# 种子菜单 ID（sys_menu.sql：1/2/3 为目录与菜单节点）
SEED_MENU_IDS = [1, 2, 3]


def _role_data(**overrides) -> dict:
    data = {"name": "svc测试角色", "code": "SVC_TEST_ROLE", "dataScope": 1, "sort": 1, "status": 1}
    data.update(overrides)
    return data


async def _create_role(db, mock_redis, **overrides):
    return await role_service.create_role(db, mock_redis, _role_data(**overrides))


async def _flush_after_commit(db):
    """模拟请求事务提交：执行 role_service 登记的提交后回调（缓存失效/会话踢出）"""
    await run_after_commit_callbacks(db)


def _admin_operator():
    """非 root 管理员操作者（持有角色编辑权限，用于权限提升校验通过路径）"""
    return make_user_context(2, username="admin", roles=["ADMIN"], permissions=["sys:role:edit"])


def _root_operator():
    return make_user_context(1, username="root", roles=["ROOT"], permissions=[])


# ===== 新增（T-RM-007/009/010/011/012） =====


async def test_create_role_persists_fields(db, mock_redis):
    role = await _create_role(
        db, mock_redis, name="持久化角色", code="SVC_PERSIST", dataScope=2, sort=7
    )
    assert role.id > 0
    assert role.name == "持久化角色"
    assert role.code == "SVC_PERSIST"
    assert role.data_scope == 2
    assert role.sort == 7
    assert role.status == 1


async def test_create_role_rejects_blank_name_and_code(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await _create_role(db, mock_redis, name="")
    assert ei.value.code == ResultCode.PARAM_ERROR

    with pytest.raises(BusinessException) as ei:
        await _create_role(db, mock_redis, code="")
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_create_role_rejects_missing_data_scope(db, mock_redis):
    data = {"name": "无数据权限角色", "code": "SVC_NO_SCOPE", "sort": 1}
    with pytest.raises(BusinessException) as ei:
        await role_service.create_role(db, mock_redis, data)
    assert ei.value.code == ResultCode.PARAM_ERROR
    assert "数据权限不能为空" in ei.value.message


async def test_create_role_rejects_conflict_with_preset_code(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await _create_role(db, mock_redis, code="ADMIN")
    assert ei.value.code == ResultCode.DATA_EXISTS


async def test_create_role_recreate_same_code_after_delete(db, mock_redis):
    """唯一键含 deleted：删除后重建同编码释放键位，创建成功且软删历史行保留"""
    role = await _create_role(db, mock_redis, name="软删角色", code="SVC_SOFT_DELETED")
    await role_service.delete_roles(db, mock_redis, str(role.id))

    recreated = await _create_role(db, mock_redis, name="软删角色复用", code="SVC_SOFT_DELETED")
    assert recreated.id != role.id
    assert recreated.deleted == 0


async def test_create_role_recreate_same_name_after_delete(db, mock_redis):
    """唯一键含 deleted：删除后重建同名称成功"""
    role = await _create_role(db, mock_redis, name="软删名称占用", code="SVC_NAME_DEL")
    await role_service.delete_roles(db, mock_redis, str(role.id))

    recreated = await _create_role(db, mock_redis, name="软删名称占用", code="SVC_NAME_DEL_2")
    assert recreated.id != role.id
    assert recreated.deleted == 0


async def test_create_role_name_unique_among_active(db, mock_redis):
    """活跃行重名仍拒绝（防裸撞唯一键）"""
    await _create_role(db, mock_redis, name="活跃重名角色", code="SVC_ACTIVE_DUP")
    with pytest.raises(BusinessException) as ei:
        await _create_role(db, mock_redis, name="活跃重名角色", code="SVC_ACTIVE_DUP_2")
    assert ei.value.code == ResultCode.DATA_EXISTS


# ===== 编辑（T-RM-015/017/018/019/021 + 内置角色保护） =====


async def test_update_role_rejects_code_change(db, mock_redis):
    role = await _create_role(db, mock_redis, code="SVC_CODE_LOCK")
    with pytest.raises(BusinessException) as ei:
        await role_service.update_role(
            db, mock_redis, role.id, {"name": role.name, "code": "SVC_CODE_CHANGED"}
        )
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert "编码不可修改" in ei.value.message


async def test_update_role_nonexistent_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await role_service.update_role(db, mock_redis, 99999999, {"name": "任意"})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_update_role_updates_fields_and_clears_cache(db, mock_redis):
    role = await _create_role(db, mock_redis, code="SVC_UPDATE", sort=1)
    cache_key = f"{role_service.ROLE_PERMS_PREFIX}{role.code}"
    await mock_redis.set(cache_key, "1,2,3")

    await role_service.update_role(
        db,
        mock_redis,
        role.id,
        {"name": "更新后名称", "code": role.code, "sort": 9, "status": 0, "dataScope": 3},
    )
    await _flush_after_commit(db)

    updated = await role_service.get_role_by_id(db, role.id)
    assert updated is not None
    assert updated.name == "更新后名称"
    assert updated.sort == 9
    assert updated.status == 0
    assert updated.data_scope == 3
    assert await mock_redis.get(cache_key) is None


async def test_update_builtin_role_keeps_status_and_data_scope(db, mock_redis):
    """内置角色（ADMIN）名称/排序可改，状态与数据权限不可改（与 Java/Go 一致）"""
    admin = await role_service.get_role_by_id(db, ADMIN_ROLE_ID)
    assert admin is not None
    await role_service.update_role(
        db,
        mock_redis,
        ADMIN_ROLE_ID,
        {"name": "系统管理员改", "code": admin.code, "status": 0, "dataScope": 3, "sort": 99},
    )

    updated = await role_service.get_role_by_id(db, ADMIN_ROLE_ID)
    assert updated is not None
    assert updated.name == "系统管理员改"
    assert updated.sort == 99
    assert updated.status == admin.status
    assert updated.data_scope == admin.data_scope


# ===== 删除（T-RM-022/023/024/025/028） =====


async def test_delete_roles_soft_deletes_and_clears_cache(db, mock_redis):
    role = await _create_role(db, mock_redis, code="SVC_DELETE")
    await role_service.assign_menus_to_role(
        db, mock_redis, role.id, SEED_MENU_IDS, _admin_operator()
    )
    cache_key = f"{role_service.ROLE_PERMS_PREFIX}{role.code}"
    await mock_redis.set(cache_key, "1,2")

    await role_service.delete_roles(db, mock_redis, str(role.id))
    await _flush_after_commit(db)

    assert await role_service.get_role_by_id(db, role.id) is None
    assert await role_service.get_role_menu_ids(db, role.id) == []
    assert await mock_redis.get(cache_key) is None


async def test_delete_builtin_role_rejected(db, mock_redis):
    for role_id in (ROOT_ROLE_ID, ADMIN_ROLE_ID):
        with pytest.raises(BusinessException) as ei:
            await role_service.delete_roles(db, mock_redis, str(role_id))
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW


async def test_delete_batch_containing_builtin_rejected(db, mock_redis):
    """批量删除混合内置角色时整体拒绝，普通角色不被误删"""
    role = await _create_role(db, mock_redis, code="SVC_BATCH_MIX")
    with pytest.raises(BusinessException) as ei:
        await role_service.delete_roles(db, mock_redis, f"{ROOT_ROLE_ID},{role.id}")
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert await role_service.get_role_by_id(db, role.id) is not None


async def test_delete_role_with_users_rejected(db, mock_redis):
    """删除仍有关联用户的角色被拒，且角色不被删除（A0500）"""
    with pytest.raises(BusinessException) as ei:
        await role_service.delete_roles(db, mock_redis, str(USER_ROLE_ID))
    assert ei.value.code == ResultCode.BUSINESS_ERROR
    assert await role_service.get_role_by_id(db, USER_ROLE_ID) is not None


async def test_delete_role_with_only_deleted_users_should_succeed(db, mock_redis):
    """已软删用户的角色关联不应阻塞角色删除（暴露缺陷用例：用户删除未清理
    sys_user_role 且 count_users_by_roles 不排除已删用户 → 角色永久无法删除）"""
    from app.models.entity.sys_user import SysUser, SysUserRole

    role = await _create_role(db, mock_redis, code="SVC_DEL_USER_REF")
    user = SysUser(
        username="role_del_ref_user", nickname="待删用户", password="x", dept_id=1, status=1
    )
    db.add(user)
    await db.flush()
    db.add(SysUserRole(user_id=user.id, role_id=role.id))
    await db.flush()
    user.deleted = 1  # 模拟用户已被删除
    await db.flush()

    # 用户已删除 → 角色应可正常删除（当前实现误报"仍有用户关联"）
    await role_service.delete_roles(db, mock_redis, str(role.id))
    assert await role_service.get_role_by_id(db, role.id) is None


async def test_delete_nonexistent_role_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await role_service.delete_roles(db, mock_redis, "99999999")
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 状态管理（T-RM-029/030/031/032） =====


async def test_update_role_status_invalid_value_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await role_service.update_role_status(db, mock_redis, ADMIN_ROLE_ID, 2)
    assert ei.value.code == ResultCode.PARAM_ERROR


async def test_update_role_status_nonexistent_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await role_service.update_role_status(db, mock_redis, 99999999, 1)
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_update_builtin_role_status_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await role_service.update_role_status(db, mock_redis, ADMIN_ROLE_ID, 0)
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW


async def test_update_role_status_toggles(db, mock_redis):
    role = await _create_role(db, mock_redis, code="SVC_STATUS")

    await role_service.update_role_status(db, mock_redis, role.id, 0)
    disabled = await role_service.get_role_by_id(db, role.id)
    assert disabled is not None
    assert disabled.status == 0

    await role_service.update_role_status(db, mock_redis, role.id, 1)
    enabled = await role_service.get_role_by_id(db, role.id)
    assert enabled is not None
    assert enabled.status == 1


async def test_disable_role_kicks_online_users_enable_does_not(db, mock_redis):
    """禁用角色踢出关联在线用户（权限传播）；启用不踢"""
    import json as jsonlib

    from app.models.entity.sys_user import SysUser, SysUserRole

    role = await _create_role(db, mock_redis, code="SVC_DISABLE_KICK")
    user = SysUser(
        username="svc_disable_kick_user", nickname="在线用户", password="x", dept_id=1, status=1
    )
    db.add(user)
    await db.flush()
    db.add(SysUserRole(user_id=user.id, role_id=role.id))
    await db.flush()

    session_payload = jsonlib.dumps(
        {"userId": user.id, "username": user.username, "authorities": []}
    )
    await mock_redis.set("session:kick-disable-1", session_payload)
    await mock_redis.zadd(f"session:user:{user.id}", {"kick-disable-1": 1_700_000_000})

    # 启用（0→1）不踢会话
    await role_service.update_role_status(db, mock_redis, role.id, 1)
    await _flush_after_commit(db)
    assert await mock_redis.get("session:kick-disable-1") is not None
    assert await mock_redis.zrange(f"session:user:{user.id}", 0, -1) == ["kick-disable-1"]

    # 禁用：会话被踢出
    await role_service.update_role_status(db, mock_redis, role.id, 0)
    await _flush_after_commit(db)
    assert await mock_redis.get("session:kick-disable-1") is None
    assert await mock_redis.zrange(f"session:user:{user.id}", 0, -1) == []


async def test_role_options_hide_builtin_roles_for_non_root(db, mock_redis):
    """非 root 下拉隐藏内置角色 ROOT/ADMIN（三端口径一致），root 可见全部"""
    root_opts = await role_service.get_role_options(db, mock_redis, is_root=True)
    root_labels = {opt["label"] for opt in root_opts}
    assert "超级管理员" in root_labels
    assert "系统管理员" in root_labels

    non_root_opts = await role_service.get_role_options(db, mock_redis, is_root=False)
    assert "超级管理员" not in {opt["label"] for opt in non_root_opts}
    assert "系统管理员" not in {opt["label"] for opt in non_root_opts}


# ===== 菜单权限分配（T-RM-033/034/035/036/037/038/039） =====


async def test_assign_menus_replaces_exactly_and_clears_cache(db, mock_redis):
    """全量替换语义：回显与请求完全一致（父节点全选/半选一致性的数据基础）"""
    role = await _create_role(db, mock_redis, code="SVC_ASSIGN")
    cache_key = f"{role_service.ROLE_PERMS_PREFIX}{role.code}"

    await role_service.assign_menus_to_role(
        db, mock_redis, role.id, SEED_MENU_IDS, _admin_operator()
    )
    assert sorted(await role_service.get_role_menu_ids(db, role.id)) == SEED_MENU_IDS
    await _flush_after_commit(db)
    assert await mock_redis.get(cache_key) is None

    # 半选：仅保留子集，回显精确等于请求集合
    await role_service.assign_menus_to_role(db, mock_redis, role.id, [2], _admin_operator())
    await _flush_after_commit(db)
    assert await role_service.get_role_menu_ids(db, role.id) == [2]

    # 清空
    await role_service.assign_menus_to_role(db, mock_redis, role.id, [], _admin_operator())
    await _flush_after_commit(db)
    assert await role_service.get_role_menu_ids(db, role.id) == []


async def test_assign_menus_nonexistent_menu_rejected_and_assignment_intact(db, mock_redis):
    role = await _create_role(db, mock_redis, code="SVC_BAD_MENU")
    await role_service.assign_menus_to_role(db, mock_redis, role.id, [1], _admin_operator())
    await _flush_after_commit(db)

    with pytest.raises(BusinessException) as ei:
        await role_service.assign_menus_to_role(
            db, mock_redis, role.id, [1, 99999999], _admin_operator()
        )
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND
    assert "菜单不存在" in ei.value.message
    # 失败不产生半写：原分配保持不变
    assert await role_service.get_role_menu_ids(db, role.id) == [1]


async def test_assign_menus_nonexistent_role_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await role_service.assign_menus_to_role(db, mock_redis, 99999999, [1], _admin_operator())
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 角色选项列表（T-RM-046） =====


async def test_role_options_exclude_disabled_and_builtin_for_non_root(db, mock_redis):
    await _create_role(db, mock_redis, name="禁用选项角色", code="SVC_DISABLED", status=0)
    await _create_role(db, mock_redis, name="启用选项角色", code="SVC_ENABLED_OPT")
    await _flush_after_commit(db)

    non_root = await role_service.get_role_options(db, mock_redis, is_root=False)
    labels = {opt["label"] for opt in non_root}
    # 内置角色对非 root 不可见，禁用角色不可见
    assert "超级管理员" not in labels
    assert "系统管理员" not in labels
    assert "禁用选项角色" not in labels
    assert "启用选项角色" in labels

    root_opts = await role_service.get_role_options(db, mock_redis, is_root=True)
    root_labels = {opt["label"] for opt in root_opts}
    assert {"超级管理员", "系统管理员"} <= root_labels


# ===== 提交后回调（缓存时机 T-RM-021/028/032/038 的回滚语义）与新增安全/性能行为 =====


async def test_role_options_cache_hit_and_invalidation(db, mock_redis):
    """role:options 缓存 TTL 1h：命中不回源；角色变更（提交后）失效重建"""
    role = await _create_role(db, mock_redis, name="缓存命中角色", code="SVC_OPT_CACHE")
    await _flush_after_commit(db)

    first = await role_service.get_role_options(db, mock_redis, is_root=True)
    assert "缓存命中角色" in {opt["label"] for opt in first}

    # 未失效前：新增角色对缓存不可见（命中缓存）
    await _create_role(db, mock_redis, name="缓存外角色", code="SVC_OPT_CACHED_OUT")
    cached = await role_service.get_role_options(db, mock_redis, is_root=True)
    assert "缓存外角色" not in {opt["label"] for opt in cached}

    # 提交后回调失效缓存：再次查询可见
    await _flush_after_commit(db)
    refreshed = await role_service.get_role_options(db, mock_redis, is_root=True)
    assert "缓存外角色" in {opt["label"] for opt in refreshed}
    assert role.id


async def test_assign_menus_kicks_online_users_with_role(db, mock_redis):
    """菜单分配变更后权限传播：反查活跃用户并踢出其全部在线会话（提交后执行）"""
    import json as jsonlib

    from app.models.entity.sys_user import SysUser, SysUserRole

    role = await _create_role(db, mock_redis, code="SVC_KICK")
    user = SysUser(username="svc_kick_user", nickname="在线用户", password="x", dept_id=1, status=1)
    db.add(user)
    await db.flush()
    db.add(SysUserRole(user_id=user.id, role_id=role.id))
    await db.flush()

    session_id = "sess-kick-1"
    await mock_redis.set(
        f"session:{session_id}",
        jsonlib.dumps({"userId": user.id, "username": user.username, "authorities": []}),
    )
    await mock_redis.zadd(f"session:user:{user.id}", {session_id: 1_700_000_000})

    await role_service.assign_menus_to_role(
        db, mock_redis, role.id, SEED_MENU_IDS, _admin_operator()
    )
    await _flush_after_commit(db)

    assert await mock_redis.get(f"session:{session_id}") is None
    assert await mock_redis.zrange(f"session:user:{user.id}", 0, -1) == []


async def test_assign_menus_with_unheld_perm_rejected(db, mock_redis):
    """权限提升防护（A0301）：操作者不能授予自己未持有的权限标识，失败不产生半写"""
    from sqlalchemy import text

    from app.models.entity.sys_menu import SysMenu

    role = await _create_role(db, mock_redis, code="SVC_ESCALATE")
    db.add(
        SysMenu(
            parent_id=1,
            tree_path=",1,",
            name="越权按钮",
            type=4,
            perm="sys:hidden:danger",
            visible=1,
            status=1,
            sort=1,
            icon="",
        )
    )
    await db.flush()
    menu_id = (await db.execute(text("SELECT MAX(id) FROM sys_menu"))).scalar_one()

    await role_service.assign_menus_to_role(db, mock_redis, role.id, [1], _admin_operator())

    operator = make_user_context(
        2, username="admin", roles=["ADMIN"], permissions=["sys:role:edit"]
    )
    with pytest.raises(BusinessException) as ei:
        await role_service.assign_menus_to_role(db, mock_redis, role.id, [1, menu_id], operator)
    assert ei.value.code == ResultCode.ACCESS_UNAUTHORIZED
    await _flush_after_commit(db)
    # 失败不产生半写：原分配保持不变
    assert await role_service.get_role_menu_ids(db, role.id) == [1]


async def test_assign_menus_with_held_perm_allowed(db, mock_redis):
    """操作者持有被分配的全部权限标识时可正常分配"""
    from sqlalchemy import text

    from app.models.entity.sys_menu import SysMenu

    role = await _create_role(db, mock_redis, code="SVC_HOLD_PERM")
    db.add(
        SysMenu(
            parent_id=1,
            tree_path=",1,",
            name="常规按钮",
            type=4,
            perm="sys:role:edit",
            visible=1,
            status=1,
            sort=1,
            icon="",
        )
    )
    await db.flush()
    menu_id = (await db.execute(text("SELECT MAX(id) FROM sys_menu"))).scalar_one()

    await role_service.assign_menus_to_role(db, mock_redis, role.id, [menu_id], _admin_operator())
    await _flush_after_commit(db)
    assert await role_service.get_role_menu_ids(db, role.id) == [menu_id]


async def test_assign_menus_root_operator_bypasses_perm_check(db, mock_redis):
    """ROOT 忽略权限判断（与 Go FindPermsByRoles / Java SecurityUtils.isRoot 语义一致）"""
    from sqlalchemy import text

    from app.models.entity.sys_menu import SysMenu

    role = await _create_role(db, mock_redis, code="SVC_ROOT_ASSIGN")
    db.add(
        SysMenu(
            parent_id=1,
            tree_path=",1,",
            name="任意按钮",
            type=4,
            perm="sys:any:perm",
            visible=1,
            status=1,
            sort=1,
            icon="",
        )
    )
    await db.flush()
    menu_id = (await db.execute(text("SELECT MAX(id) FROM sys_menu"))).scalar_one()

    await role_service.assign_menus_to_role(db, mock_redis, role.id, [menu_id], _root_operator())
    await _flush_after_commit(db)
    assert await role_service.get_role_menu_ids(db, role.id) == [menu_id]
