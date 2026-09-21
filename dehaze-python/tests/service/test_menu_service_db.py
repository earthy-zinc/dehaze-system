"""菜单服务层测试（真实 MySQL 测试库 + fakeredis，覆盖类型特殊字段/树级联/删除保护/缓存失效）。

对应菜单管理测试用例.md：T-MM-014（外链 component 清空）、目录 Layout 约定、
T-MM-026（移动上级后 tree_path 级联不变量）、T-MM-036/041/042（级联删除与关联清理、
缓存刷新）、T-MM-044（批量删除去重）、T-MM-059/060/061（路由过滤不变量）。
"""

import itertools

import pytest
from sqlalchemy import select

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.repository.menu_repository import menu_repository
from app.service.menu_service import ROUTE_CACHE_KEY, menu_service

pytestmark = pytest.mark.requires_db

USER_ROLE_ID = 5  # 预置普通用户角色（config/sql/data 种子数据）
ADMIN_ROLE_CODE = "ADMIN"  # 种子数据中持有 sys:menu:add（菜单ID 13）的角色

_seq = itertools.count(1)


async def _save(db, mock_redis, **overrides) -> SysMenu:
    n = next(_seq)
    data = {
        "parentId": 0,
        "name": f"svc菜单{n}",
        "type": 2,
        "path": f"/svc{n}",
        "sort": 1,
    }
    data.update(overrides)
    return await menu_service.save_menu(db, mock_redis, data)


async def _fetch(db, menu_id: int) -> SysMenu | None:
    result = await db.execute(select(SysMenu).where(SysMenu.id == menu_id))
    return result.scalar_one_or_none()


# ===== 类型特殊字段（对齐 Java saveMenu） =====


async def test_save_catalog_root_prefixes_path_and_sets_layout(db, mock_redis):
    menu = await _save(db, mock_redis, type=2, path="svccatalog")
    assert menu.path == "/svccatalog"
    assert menu.component == "Layout"


async def test_save_extlink_clears_component(db, mock_redis):
    menu = await _save(db, mock_redis, type=3, path="https://example.com", component="test/index")
    assert menu.component is None


async def test_save_menu_type_component_preserved(db, mock_redis):
    menu = await _save(db, mock_redis, type=1, path="/svcmenu", component="svc/index")
    assert menu.component == "svc/index"
    assert menu.path == "/svcmenu"  # 非 catalog 根级不加 "/" 前缀


# ===== 新增菜单默认关联内置角色 + 权限缓存失效 =====


async def test_new_menu_assigned_to_root_and_admin_roles(db, mock_redis):
    """新增菜单默认分配 ROOT（超级管理员）与 ADMIN（系统管理员），其他角色需手动分配"""
    menu = await _save(db, mock_redis)
    result = await db.execute(select(SysRoleMenu).where(SysRoleMenu.menu_id == menu.id))
    linked_role_ids = {row.role_id for row in result.scalars().all()}
    root_role = await menu_service.role_repository.get_by_code(db, "ROOT")
    admin_role = await menu_service.role_repository.get_by_code(db, "ADMIN")
    assert root_role is not None
    assert admin_role is not None
    assert linked_role_ids == {root_role.id, admin_role.id}


async def test_save_menu_invalidates_role_perm_cache(db, mock_redis):
    """T-MM-034：菜单变更后角色权限缓存立即失效"""
    perms = await menu_service.list_role_perms(db, mock_redis, {ADMIN_ROLE_CODE})
    assert "sys:menu:add" in perms  # 种子菜单 13（菜单新增）分配给 ADMIN
    cache_key = f"role:perms:{ADMIN_ROLE_CODE}"
    assert await mock_redis.get(cache_key) is not None

    await _save(db, mock_redis)

    assert await mock_redis.get(cache_key) is None


# ===== tree_path 级联不变量 =====


async def test_move_subtree_refreshes_descendant_tree_path(db, mock_redis):
    """T-MM-026：移动 A 到 C 下后，其子孙 B 的 tree_path 必须级联更新。

    子路径 = 父路径 + 父ID 前缀是循环引用检测（_is_descendant）与
    级联删除范围计算（get_menu_ids_with_children_batch）的基础不变量。
    """
    a = await _save(db, mock_redis, type=2)
    b = await _save(db, mock_redis, parentId=a.id, type=1, component="svc/index")
    c = await _save(db, mock_redis, type=2)

    await menu_service.save_menu(
        db,
        mock_redis,
        {"id": a.id, "parentId": c.id, "name": a.name, "type": 2, "path": a.path, "sort": a.sort},
    )

    fresh_b = await _fetch(db, b.id)
    assert fresh_b is not None
    assert str(c.id) in fresh_b.tree_path.split(","), (
        f"子孙 tree_path 未级联更新：B.tree_path={fresh_b.tree_path!r}，期望包含父链 {c.id}"
    )


# ===== 删除（级联 / 关联清理 / 去重 / 不存在） =====


async def test_delete_menu_cascades_descendants_and_role_menus(db, mock_redis):
    """T-MM-036/041：删除父菜单级联删除子孙，并清理角色-菜单关联"""
    a = await _save(db, mock_redis)
    b = await _save(db, mock_redis, parentId=a.id, type=1, component="svc/index")
    await menu_repository.save_role_menu(db, USER_ROLE_ID, a.id)

    await menu_service.delete_menu(db, mock_redis, [a.id])

    assert await _fetch(db, a.id) is None
    assert await _fetch(db, b.id) is None
    result = await db.execute(select(SysRoleMenu).where(SysRoleMenu.menu_id.in_([a.id, b.id])))
    assert result.scalars().all() == []


async def test_delete_menu_with_duplicate_ids_deduplicates(db, mock_redis):
    """T-MM-044：批量删除含重复 ID 应去重后成功，不误报菜单不存在"""
    a = await _save(db, mock_redis)

    await menu_service.delete_menu(db, mock_redis, [a.id, a.id])

    assert await _fetch(db, a.id) is None


async def test_delete_missing_menu_rejected(db, mock_redis):
    with pytest.raises(BusinessException) as ei:
        await menu_service.delete_menu(db, mock_redis, [99999999])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


# ===== 可见性 =====


async def test_update_visible_persists(db, mock_redis):
    menu = await _save(db, mock_redis)
    await menu_service.update_menu_visible(db, mock_redis, menu.id, 0)
    fresh = await _fetch(db, menu.id)
    assert fresh is not None
    assert fresh.visible == 0


# ===== 路由过滤不变量与缓存失效链路 =====


def _collect_paths(routes: list[dict]) -> set[str]:
    paths = set()
    for route in routes:
        if route.get("path"):
            paths.add(route["path"])
        paths |= _collect_paths(route.get("children") or [])
    return paths


async def test_routes_filter_by_type_and_mark_hidden(db, mock_redis):
    """T-MM-060/061：按钮/外链类型不生成路由；T-MM-059：隐藏菜单保留路由但
    以 meta.hidden 标记（行为变更：由"不生成"改为"生成但标记"，前端按标记过滤）；
    新建菜单默认分配 ROOT+ADMIN → meta.roles 应含两者。"""
    hidden = await _save(db, mock_redis, type=1, component="svc/index", visible=0)
    extlink = await _save(db, mock_redis, type=3, path="https://example.com/svc")
    button = await _save(db, mock_redis, parentId=hidden.id, type=4, perm=f"svc:route:{next(_seq)}")
    visible_menu = await _save(db, mock_redis, type=1, component="svc/index")

    routes = await menu_service.list_routes(db, mock_redis)
    paths = _collect_paths(routes)

    assert extlink.path not in paths  # 外链不生成路由
    assert button.path not in paths  # 按钮不生成路由

    hidden_route = next(r for r in routes if r["path"] == hidden.path)
    assert hidden_route["meta"]["hidden"] is True

    visible_route = next(r for r in routes if r["path"] == visible_menu.path)
    assert visible_route["meta"]["hidden"] is False
    assert visible_route["meta"]["roles"] == ["ADMIN", "ROOT"]


async def test_menu_change_invalidates_route_cache(db, mock_redis):
    """T-MM-042：保存/删除菜单后路由缓存立即失效，下次查询反映最新数据"""
    routes_before = await menu_service.list_routes(db, mock_redis)
    assert await mock_redis.get(ROUTE_CACHE_KEY) is not None

    menu = await _save(db, mock_redis, type=1, component="svc/index")
    assert await mock_redis.get(ROUTE_CACHE_KEY) is None

    routes_with = await menu_service.list_routes(db, mock_redis)
    assert menu.path in _collect_paths(routes_with)
    assert menu.path not in _collect_paths(routes_before)

    await menu_service.delete_menu(db, mock_redis, [menu.id])
    assert await mock_redis.get(ROUTE_CACHE_KEY) is None

    routes_after = await menu_service.list_routes(db, mock_redis)
    assert menu.path not in _collect_paths(routes_after)


# ===== 预置菜单保护（is_preset，需求规格 3.4.5） =====


async def _create_preset(db, **overrides) -> SysMenu:
    n = next(_seq)
    menu = SysMenu(
        parent_id=0,
        tree_path=",",
        name=f"预置菜单{n}",
        type=2,
        path=f"/preset{n}",
        visible=1,
        sort=1,
        is_preset=1,
    )
    db.add(menu)
    await db.flush()
    return menu


async def test_delete_preset_menu_rejected(db, mock_redis):
    """预置菜单不可删除，整批拒绝且数据保留"""
    preset = await _create_preset(db)

    with pytest.raises(BusinessException) as ei:
        await menu_service.delete_menu(db, mock_redis, [preset.id])
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert "系统预置菜单不可删除" in ei.value.message
    assert await _fetch(db, preset.id) is not None


async def test_delete_mixed_batch_with_preset_rejected(db, mock_redis):
    """批量删除中任一目标为预置菜单时整批拒绝，不做部分删除"""
    preset = await _create_preset(db)
    normal = await _save(db, mock_redis)

    with pytest.raises(BusinessException) as ei:
        await menu_service.delete_menu(db, mock_redis, [normal.id, preset.id])
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert await _fetch(db, preset.id) is not None
    assert await _fetch(db, normal.id) is not None


async def test_update_preset_type_rejected(db, mock_redis):
    """预置菜单不可修改 type（路由生成锚点）"""
    preset = await _create_preset(db)

    with pytest.raises(BusinessException) as ei:
        await menu_service.save_menu(
            db,
            mock_redis,
            {"id": preset.id, "parentId": 0, "name": preset.name, "type": 1, "path": preset.path},
        )
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert "系统预置菜单不可修改类型/权限标识" in ei.value.message


async def test_update_preset_perm_rejected(db, mock_redis):
    """预置菜单（按钮）不可修改 perm（接口鉴权锚点）"""
    preset = await _create_preset(db)
    preset.type = 4
    preset.perm = "sys:preset:old"
    await db.flush()

    with pytest.raises(BusinessException) as ei:
        await menu_service.save_menu(
            db,
            mock_redis,
            {
                "id": preset.id,
                "parentId": 0,
                "name": preset.name,
                "type": 4,
                "perm": "sys:preset:new",
            },
        )
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert "系统预置菜单不可修改类型/权限标识" in ei.value.message


async def test_update_preset_display_attrs_allowed(db, mock_redis):
    """预置菜单展示类属性（名称/图标/排序/可见性）允许修改"""
    preset = await _create_preset(db)

    await menu_service.save_menu(
        db,
        mock_redis,
        {
            "id": preset.id,
            "parentId": 0,
            "name": "预置菜单改名",
            "type": 2,
            "path": preset.path,
            "icon": "setting",
            "sort": 9,
            "visible": 0,
        },
    )

    fresh = await _fetch(db, preset.id)
    assert fresh is not None
    assert fresh.name == "预置菜单改名"
    assert fresh.icon == "setting"
    assert fresh.sort == 9
    assert fresh.visible == 0
    assert fresh.is_preset == 1
