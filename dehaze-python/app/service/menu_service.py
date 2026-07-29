"""
菜单服务

提供菜单 CRUD 功能，支持树形结构
"""

from typing import Any

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_menu import SysMenu
from app.repository.menu_repository import menu_repository
from app.utils.datetime_utils import format_time
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

# 菜单类型枚举（对齐 Java MenuTypeEnum）
MENU_TYPE_MENU = 1      # 菜单
MENU_TYPE_CATALOG = 2   # 目录
MENU_TYPE_EXTLINK = 3   # 外链
MENU_TYPE_BUTTON = 4    # 按钮

# 整数 → 字符串枚举名（用于响应序列化，对齐 Java MenuTypeEnum 的 Jackson 序列化）
MENU_TYPE_TO_NAME = {
    MENU_TYPE_MENU: "MENU",
    MENU_TYPE_CATALOG: "CATALOG",
    MENU_TYPE_EXTLINK: "EXTLINK",
    MENU_TYPE_BUTTON: "BUTTON",
}

# 路由缓存 Key
ROUTE_CACHE_KEY = "menu:routes"


class MenuService:
    """菜单服务（异步版本）"""

    @staticmethod
    async def list_menus(db: AsyncSession, keywords: str | None = None) -> list[dict[str, Any]]:
        """
        获取菜单列表（树形结构）

        对齐 Java listMenus：使用 TreeDataUtils.findRootIds 找出结果集中的根节点
        （父ID不在当前结果集ID中的节点作为根），再从根节点构建树。

        Args:
            db: 数据库会话
            keywords: 搜索关键字（菜单名称）

        Returns:
            菜单列表
        """
        menus = await menu_repository.get_list(db, keyword=keywords)
        if not menus:
            return []

        # 对齐 Java TreeDataUtils.findRootIds：
        # 收集结果集中的所有ID和父ID，父ID不在ID集合中的即为根
        ids = {menu.id for menu in menus}
        root_ids = {menu.parent_id for menu in menus if menu.parent_id not in ids}

        # 构建 children_map（O(N)），避免递归中反复扫描全量列表
        children_map: dict[int, list[SysMenu]] = {}
        for menu in menus:
            children_map.setdefault(menu.parent_id, []).append(menu)

        tree: list[dict[str, Any]] = []
        for root_id in root_ids:
            tree.extend(MenuService._build_menu_tree(root_id, children_map))
        return tree

    @staticmethod
    def _build_menu_tree(
        parent_id: int,
        children_map: dict[int, list[SysMenu]],
    ) -> list[dict[str, Any]]:
        """
        递归构建菜单树（使用预构建的 children_map，O(N)）

        Args:
            parent_id: 父级菜单ID
            children_map: 按 parent_id 分组的菜单字典

        Returns:
            树形菜单列表
        """
        tree = []
        for menu in children_map.get(parent_id, []):
            menu_dict: dict[str, Any] = {
                "id": menu.id,
                "parentId": menu.parent_id,
                "name": menu.name,
                "type": MENU_TYPE_TO_NAME.get(menu.type, str(menu.type)),
                "path": menu.path,
                "component": menu.component,
                "perm": menu.perm,
                "visible": menu.visible,
                "sort": menu.sort,
                "icon": menu.icon,
                "redirect": menu.redirect,
                "alwaysShow": menu.always_show,
                "keepAlive": menu.keep_alive,
                "createTime": format_time(menu.create_time),
            }

            # 递归查找子菜单
            children = MenuService._build_menu_tree(menu.id, children_map)
            if children:
                menu_dict["children"] = children

            tree.append(menu_dict)

        return tree

    @staticmethod
    async def list_menu_options(db: AsyncSession) -> list[dict[str, Any]]:
        """
        获取菜单下拉选项列表

        Args:
            db: 数据库会话

        Returns:
            菜单下拉选项列表
        """
        menus = await menu_repository.get_list(db)
        if not menus:
            return []

        # 构建 children_map（O(N)）
        children_map: dict[int, list[SysMenu]] = {}
        for menu in menus:
            children_map.setdefault(menu.parent_id, []).append(menu)

        return MenuService._build_menu_options(0, children_map)

    @staticmethod
    def _build_menu_options(
        parent_id: int,
        children_map: dict[int, list[SysMenu]],
    ) -> list[dict[str, Any]]:
        """
        递归构建菜单下拉选项（使用预构建的 children_map，O(N)）

        Args:
            parent_id: 父级菜单ID
            children_map: 按 parent_id 分组的菜单字典

        Returns:
            菜单下拉选项列表
        """
        options = []
        for menu in children_map.get(parent_id, []):
            # 按钮类型不显示在下拉选项中
            if menu.type == MENU_TYPE_BUTTON:
                continue

            option: dict[str, Any] = {"value": menu.id, "label": menu.name}

            # 递归查找子菜单选项
            children = MenuService._build_menu_options(menu.id, children_map)
            if children:
                option["children"] = children

            options.append(option)

        return options

    @staticmethod
    async def save_menu(db: AsyncSession, redis: Redis, data: dict[str, Any]) -> SysMenu:
        """
        保存菜单（新增/修改）

        对齐 Java saveMenu：不做额外业务校验，使用 saveOrUpdate 语义
        （ID 存在则更新，不存在则新增）。

        Args:
            db: 数据库会话
            redis: Redis 客户端
            data: 菜单数据

        Returns:
            保存的菜单对象
        """
        menu_id = data.get("id")

        # saveOrUpdate 语义：ID 存在则查询已有记录，不存在则新建
        menu = None
        if menu_id:
            menu = await menu_repository.get_by_id(db, menu_id)
            # 修改时检查菜单是否存在
            if menu is None:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在")
        is_new = menu is None
        if is_new:
            menu = SysMenu()
            if menu_id:
                menu.id = menu_id

        # 设置菜单属性
        menu.parent_id = data.get("parentId", 0)
        menu.name = data.get("name", "")
        menu.type = data.get("type", MENU_TYPE_MENU)
        menu.path = data.get("path", "")
        menu.component = data.get("component")
        menu.perm = data.get("perm")
        menu.visible = data.get("visible", 1)
        menu.sort = data.get("sort", 1)
        menu.icon = data.get("icon", "")
        menu.redirect = data.get("redirect")
        menu.always_show = data.get("alwaysShow", 0)
        menu.keep_alive = data.get("keepAlive", 0)

        # 生成树路径
        tree_path = await MenuService._generate_menu_tree_path(db, menu.parent_id)
        menu.tree_path = tree_path

        # 根据类型处理特殊字段（对齐 Java saveMenu）
        if menu.type == MENU_TYPE_CATALOG:
            # 目录类型：根目录补全路径前缀，设置 component 为 "Layout"
            if menu.parent_id == 0 and menu.path and not menu.path.startswith("/"):
                menu.path = "/" + menu.path
            menu.component = "Layout"
        elif menu.type == MENU_TYPE_EXTLINK:
            # 外链类型：清空 component
            menu.component = None

        if is_new:
            merged = await menu_repository.create_menu(db, menu)
            # 新增菜单默认分配给超级管理员角色
            from app.repository.role_repository import ROOT_ROLE_CODE, role_repository
            root_role = await role_repository.get_by_code(db, ROOT_ROLE_CODE)
            if root_role:
                await menu_repository.save_role_menu(db, root_role.id, merged.id)
        else:
            merged = await menu_repository.update_menu(db, menu)

        # 清除缓存
        await MenuService._clear_menu_cache(db, redis)

        return merged

    @staticmethod
    async def _generate_menu_tree_path(db: AsyncSession, parent_id: int) -> str:
        """
        生成菜单树路径

        Args:
            db: 数据库会话
            parent_id: 父级菜单ID

        Returns:
            树路径，格式如 ",1,2,"
        """
        if parent_id == 0:
            return ","

        parent_menu = await menu_repository.get_by_id(db, parent_id)

        if parent_menu and parent_menu.tree_path is not None:
            return f"{parent_menu.tree_path}{parent_id},"
        else:
            return f",{parent_id},"

    @staticmethod
    async def list_routes(db: AsyncSession, redis: Redis) -> list[dict[str, Any]]:
        """
        获取路由列表（带缓存）

        Args:
            db: 数据库会话
            redis: Redis 客户端

        Returns:
            路由列表
        """
        cache = CacheService(redis)

        # 尝试从缓存获取
        cached = await cache.get_json(ROUTE_CACHE_KEY)
        if cached is not None:
            return cached

        # 从数据库获取
        menus = await menu_repository.get_route_menus(db)
        if not menus:
            return []

        # 构建 children_map（O(N)）
        children_map: dict[int, list[SysMenu]] = {}
        for menu in menus:
            children_map.setdefault(menu.parent_id, []).append(menu)

        routes = MenuService._build_routes(0, children_map)

        # 写入缓存
        await cache.set_json(ROUTE_CACHE_KEY, routes, CACHE_TTL_HOUR)

        return routes

    @staticmethod
    def _build_routes(
        parent_id: int,
        children_map: dict[int, list[SysMenu]],
    ) -> list[dict[str, Any]]:
        """
        递归构建路由列表（使用预构建的 children_map，O(N)）

        Args:
            parent_id: 父级菜单ID
            children_map: 按 parent_id 分组的菜单字典

        Returns:
            路由列表
        """
        routes = []
        for menu in children_map.get(parent_id, []):
            # 构建路由对象
            route = MenuService._to_route_vo(menu)

            # 递归查找子路由
            children = MenuService._build_routes(menu.id, children_map)
            if children:
                route["children"] = children

            routes.append(route)

        return routes

    @staticmethod
    def _to_route_vo(menu: SysMenu) -> dict[str, Any]:
        """
        将菜单转换为路由对象

        Args:
            menu: 菜单对象

        Returns:
            路由对象
        """
        # 路由name需要驼峰命名，首字母大写
        route_name = "".join(word.capitalize()
                             for word in menu.path.replace("-", "_").split("_") if word)

        route: dict[str, Any] = {
            "name": route_name,
            "path": menu.path,
            "redirect": menu.redirect,
            "component": menu.component,
        }

        # 构建meta信息
        meta: dict[str, Any] = {"title": menu.name, "icon": menu.icon,
                                "hidden": menu.visible == 0}

        # 【菜单】是否开启页面缓存
        if menu.type == MENU_TYPE_MENU and menu.keep_alive == 1:
            meta["keepAlive"] = True

        # 【目录】只有一个子路由是否始终显示
        if menu.type == MENU_TYPE_CATALOG and menu.always_show == 1:
            meta["alwaysShow"] = True

        route["meta"] = meta
        return route

    @staticmethod
    async def update_menu_visible(
        db: AsyncSession,
        redis: Redis,
        menu_id: int,
        visible: int,
    ) -> None:
        """
        更新菜单显示状态

        Args:
            db: 数据库会话
            redis: Redis 客户端
            menu_id: 菜单ID
            visible: 显示状态（1:显示; 0:隐藏）

        Raises:
            BusinessException: 显示状态无效或菜单不存在
        """
        if visible not in [0, 1]:
            raise BusinessException(ResultCode.PARAM_ERROR, "显示状态只能为0或1")

        menu = await menu_repository.get_by_id(db, menu_id)

        if not menu:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在")

        menu.visible = visible

        # 清除缓存
        await MenuService._clear_menu_cache(db, redis)

    @staticmethod
    async def list_role_perms(db: AsyncSession, redis: Redis, roles: set[str]) -> set[str]:
        """
        获取角色权限集合

        统一权限缓存策略：逐角色判断缺失 → 回源 → 回填 → 设 TTL（30min）。
        使用 SingleFlight 防止缓存击穿（CacheService.get_json_with_loader 内置）。

        Args:
            db: 数据库会话
            redis: Redis 客户端
            roles: 角色编码集合

        Returns:
            权限集合
        """
        if not roles:
            return set()

        cache = CacheService(redis)
        all_perms: set[str] = set()

        for role_code in roles:
            cache_key = f"role:perms:{role_code}"
            perms = await cache.get_json_with_loader(
                cache_key,
                loader=lambda rc=role_code: _load_role_perms(db, rc),
                ttl=1800,  # 30 分钟
                default=[],
            )
            if perms:
                if isinstance(perms, list) and len(perms) == 2 and isinstance(perms[0], str) and isinstance(perms[1], list):
                    perms = perms[1]
                all_perms.update(perms)

        return all_perms

    @staticmethod
    async def get_menu_form(db: AsyncSession, menu_id: int) -> dict[str, Any]:
        """
        获取菜单表单数据

        Args:
            db: 数据库会话
            menu_id: 菜单ID

        Returns:
            菜单表单数据

        Raises:
            BusinessException: 菜单不存在时抛出 RESOURCE_NOT_FOUND
        """
        menu = await menu_repository.get_by_id(db, menu_id)

        if not menu:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在")

        return {
            "id": menu.id,
            "parentId": menu.parent_id,
            "name": menu.name,
            "type": MENU_TYPE_TO_NAME.get(menu.type, str(menu.type)),
            "path": menu.path,
            "component": menu.component,
            "perm": menu.perm,
            "visible": menu.visible,
            "sort": menu.sort,
            "icon": menu.icon,
            "redirect": menu.redirect,
            "alwaysShow": menu.always_show,
            "keepAlive": menu.keep_alive,
        }

    @staticmethod
    async def delete_menu(db: AsyncSession, redis: Redis, menu_ids: list[int]) -> None:
        """
        批量删除菜单（级联删除子孙菜单，并清理角色-菜单关联）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            menu_ids: 菜单ID集合

        Raises:
            BusinessException: 任意一个菜单不存在时抛出 RESOURCE_NOT_FOUND
        """
        if not menu_ids:
            return

        # 校验所有传入的菜单ID都存在
        exist_count = await menu_repository.count_by_ids(db, menu_ids)
        if exist_count != len(menu_ids):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在")

        # 一次性查询所有待删除菜单ID（传入ID + 子孙），合并去重
        all_menu_ids = await menu_repository.get_menu_ids_with_children_batch(db, menu_ids)
        if not all_menu_ids:
            return

        # 1. 删除角色-菜单关联
        await menu_repository.delete_role_menus_by_menu_ids(db, all_menu_ids)

        # 2. 删除菜单
        await menu_repository.delete_menus_by_ids(db, all_menu_ids)

        # 3. 清除缓存
        await MenuService._clear_menu_cache(db, redis)

    @staticmethod
    async def _clear_menu_cache(db: AsyncSession, redis: Redis) -> None:
        """清除菜单相关缓存

        精确删除所有角色的权限缓存（按 roleCode 逐个删除，禁止通配符）。
        """
        cache = CacheService(redis)
        await cache.delete(ROUTE_CACHE_KEY)
        # 精确删除所有角色权限缓存（禁止通配符 delete_pattern）
        from app.repository.role_repository import role_repository
        role_codes = await role_repository.get_all_active_codes(db)
        for role_code in role_codes:
            await redis.delete(f"role:perms:{role_code}")


async def _load_role_perms(db: AsyncSession, role_code: str) -> list[str]:
    """加载角色权限的 loader 函数（返回 list 以支持 JSON 序列化）"""
    perms = await menu_repository.get_role_perms(db, [role_code])
    return list(perms) if perms else []
