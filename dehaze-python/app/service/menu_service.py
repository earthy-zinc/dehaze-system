"""
菜单服务

提供菜单 CRUD 功能，支持树形结构
"""

from typing import Any

from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.entity.sys_menu import SysMenu
from app.repository.menu_repository import menu_repository
from app.utils.datetime_utils import format_time
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

# 菜单类型枚举（与设计文档保持一致）
MENU_TYPE_CATALOG = 1  # 目录
MENU_TYPE_MENU = 2     # 菜单
MENU_TYPE_EXTLINK = 3  # 外链
MENU_TYPE_BUTTON = 4   # 按钮

# 路由缓存 Key
ROUTE_CACHE_KEY = "menu:routes"


class MenuService:
    """菜单服务（异步版本）"""

    @staticmethod
    async def list_menus(db: AsyncSession, keywords: str | None = None) -> list[dict[str, Any]]:
        """
        获取菜单列表（树形结构）

        Args:
            db: 数据库会话
            keywords: 搜索关键字（菜单名称）

        Returns:
            菜单列表
        """
        menus = await menu_repository.get_list(db, keyword=keywords)
        return MenuService._build_menu_tree(0, menus)

    @staticmethod
    def _build_menu_tree(parent_id: int, menus: list[SysMenu]) -> list[dict[str, Any]]:
        """
        递归构建菜单树

        Args:
            parent_id: 父级菜单ID
            menus: 菜单列表

        Returns:
            树形菜单列表
        """
        tree = []
        for menu in menus:
            if menu.parent_id == parent_id:
                menu_dict: dict[str, Any] = {
                    "id": menu.id,
                    "parentId": menu.parent_id,
                    "name": menu.name,
                    "type": menu.type,
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
                children = MenuService._build_menu_tree(menu.id, menus)
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
        return MenuService._build_menu_options(0, menus)

    @staticmethod
    def _build_menu_options(parent_id: int, menus: list[SysMenu]) -> list[dict[str, Any]]:
        """
        递归构建菜单下拉选项

        Args:
            parent_id: 父级菜单ID
            menus: 菜单列表

        Returns:
            菜单下拉选项列表
        """
        options = []
        for menu in menus:
            if menu.parent_id == parent_id:
                # 按钮类型不显示在下拉选项中
                if menu.type == MENU_TYPE_BUTTON:
                    continue

                option: dict[str, Any] = {"value": menu.id, "label": menu.name}

                # 递归查找子菜单选项
                children = MenuService._build_menu_options(menu.id, menus)
                if children:
                    option["children"] = children

                options.append(option)

        return options

    @staticmethod
    async def _validate_menu_data(db: AsyncSession, data: dict[str, Any], is_update: bool = False) -> None:
        """
        校验菜单数据

        Args:
            db: 数据库会话
            data: 菜单数据
            is_update: 是否为更新操作

        Raises:
            BusinessException: 校验失败
        """
        name = data.get("name", "")
        parent_id = data.get("parentId", 0)
        menu_type = data.get("type")
        path = data.get("path", "")
        component = data.get("component")
        perm = data.get("perm")
        menu_id = data.get("id")

        # 1. 检查菜单名称唯一性（同一父级下）
        if name:
            exclude_id = menu_id if is_update else None
            if await menu_repository.check_name_exists(db, name, parent_id, exclude_id=exclude_id):
                raise BusinessException("同一父级下菜单名称已存在")

        # 2. 校验父级菜单类型（按钮和外链不能作为父级）
        if parent_id != 0:
            parent_menu = await menu_repository.get_by_id(db, parent_id)
            if not parent_menu:
                raise BusinessException("父级菜单不存在")
            if parent_menu.type == MENU_TYPE_BUTTON:
                raise BusinessException("按钮类型不能作为父级菜单")
            if parent_menu.type == MENU_TYPE_EXTLINK:
                raise BusinessException("外链类型不能作为父级菜单")

        # 3. 根据菜单类型校验必填字段
        if menu_type == MENU_TYPE_MENU:
            # 菜单类型必须配置路由地址
            if not path:
                raise BusinessException("菜单类型必须配置路由地址")
            # 菜单类型必须配置组件路径
            if not component:
                raise BusinessException("菜单类型必须配置组件路径")

        elif menu_type == MENU_TYPE_BUTTON:
            # 按钮类型必须配置权限标识
            if not perm:
                raise BusinessException("按钮类型必须配置权限标识")

        elif menu_type == MENU_TYPE_EXTLINK:
            # 外链类型必须配置路由地址
            if not path:
                raise BusinessException("外链类型必须配置路由地址")

    @staticmethod
    async def save_menu(db: AsyncSession, redis: Redis, data: dict[str, Any]) -> SysMenu:
        """
        保存菜单（新增/修改）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            data: 菜单数据

        Returns:
            保存的菜单对象

        Raises:
            BusinessException: 校验失败或菜单不存在（更新时）
        """
        menu_id = data.get("id")
        is_update = bool(menu_id)

        # 业务规则校验
        await MenuService._validate_menu_data(db, data, is_update)

        # 检查菜单是否存在（更新时）
        if is_update and menu_id is not None:
            menu = await menu_repository.get_by_id(db, menu_id)
            if not menu:
                raise BusinessException("菜单不存在")
        else:
            menu = SysMenu()

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

        # 根据类型处理特殊字段
        if menu.type == MENU_TYPE_CATALOG:
            # 目录类型：根目录补全路径前缀，设置 component 为 "Layout"
            if menu.parent_id == 0 and menu.path and not menu.path.startswith("/"):
                menu.path = "/" + menu.path
            menu.component = "Layout"
        elif menu.type == MENU_TYPE_EXTLINK:
            # 外链类型：清空 component
            menu.component = None

        if is_update:
            # 更新
            merged = await menu_repository.update_menu(db, menu)
        else:
            # 新增
            created = await menu_repository.create_menu(db, menu)
            merged = created

        # 清除缓存
        await MenuService._clear_menu_cache(redis)

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
        routes = MenuService._build_routes(0, menus)

        # 写入缓存
        await cache.set_json(ROUTE_CACHE_KEY, routes, CACHE_TTL_HOUR)

        return routes

    @staticmethod
    def _build_routes(parent_id: int, menus: list[SysMenu]) -> list[dict[str, Any]]:
        """
        递归构建路由列表

        Args:
            parent_id: 父级菜单ID
            menus: 菜单列表

        Returns:
            路由列表
        """
        routes = []
        for menu in menus:
            if menu.parent_id == parent_id:
                # 构建路由对象
                route = MenuService._to_route_vo(menu)

                # 递归查找子路由
                children = MenuService._build_routes(menu.id, menus)
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
            raise BusinessException("显示状态只能为0或1")

        menu = await menu_repository.get_by_id(db, menu_id)

        if not menu:
            raise BusinessException("菜单不存在")

        menu.visible = visible

        # 清除缓存
        await MenuService._clear_menu_cache(redis)

    @staticmethod
    async def list_role_perms(db: AsyncSession, roles: set[str]) -> set[str]:
        """
        获取角色权限集合

        Args:
            db: 数据库会话
            roles: 角色编码集合

        Returns:
            权限集合
        """
        return await menu_repository.get_role_perms(db, list(roles))

    @staticmethod
    async def get_menu_form(db: AsyncSession, menu_id: int) -> dict[str, Any] | None:
        """
        获取菜单表单数据

        Args:
            db: 数据库会话
            menu_id: 菜单ID

        Returns:
            菜单表单数据
        """
        menu = await menu_repository.get_by_id(db, menu_id)

        if not menu:
            return None

        return {
            "id": menu.id,
            "parentId": menu.parent_id,
            "name": menu.name,
            "type": menu.type,
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
    async def delete_menu(db: AsyncSession, redis: Redis, menu_id: int) -> None:
        """
        删除菜单（级联删除子菜单和角色关联）

        Args:
            db: 数据库会话
            redis: Redis 客户端
            menu_id: 菜单ID

        Raises:
            BusinessException: 菜单不存在
        """
        menu = await menu_repository.get_by_id(db, menu_id)

        if not menu:
            raise BusinessException("菜单不存在")

        # 1. 删除角色-菜单关联
        await menu_repository.delete_role_menus_by_menu_id(db, menu_id)

        # 2. 删除菜单及其子菜单
        await menu_repository.delete_menu_and_children(db, menu_id)

        # 3. 清除缓存
        await MenuService._clear_menu_cache(redis)

    @staticmethod
    async def _clear_menu_cache(redis: Redis) -> None:
        """清除菜单相关缓存"""
        cache = CacheService(redis)
        await cache.delete(ROUTE_CACHE_KEY)
        # 清除所有角色权限缓存
        await cache.delete_pattern("role:perms:*")
