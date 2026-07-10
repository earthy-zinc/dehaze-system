"""
角色服务

提供角色 CRUD 功能，支持菜单分配和数据权限管理
"""

import re
from typing import Any

from app.core.exceptions import BusinessException
from app.models.entity.sys_user import SysRole
from app.repository.base import escape_like
from app.repository.role_repository import role_repository
from app.repository.user_repository import user_repository
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession


class RoleService:
    """角色服务（异步版本）"""

    # 缓存常量
    ROLE_PERMS_PREFIX = "role:perms:"
    CACHE_TTL_PERMS = 1800  # 30分钟

    # 超级管理员角色编码
    ROOT_ROLE_CODE = "ROOT"

    @staticmethod
    async def get_role_list(
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
    ) -> tuple[list[SysRole], int]:
        """
        获取角色分页列表

        Args:
            db: 异步数据库会话
            page: 页码
            page_size: 每页数量
            keywords: 搜索关键字（角色名称或编码）

        Returns:
            tuple: (角色列表, 总数)
        """
        filters = {"deleted": 0}
        search_fields = None

        if keywords:
            search_fields = [
                (SysRole.name, "like", f"%{escape_like(keywords)}%"),
                (SysRole.code, "like", f"%{escape_like(keywords)}%"),
            ]

        return await role_repository.get_list(
            db,
            filters=filters,
            search_fields=search_fields,
            order_by="sort",
            page=page,
            page_size=page_size,
        )

    @staticmethod
    async def get_role_options(db: AsyncSession, *, is_root: bool = False) -> list[dict[str, Any]]:
        """
        获取角色下拉选项列表

        Args:
            db: 异步数据库会话
            is_root: 当前用户是否为超级管理员（非超级管理员不显示 ROOT 角色）

        Returns:
            角色下拉选项列表
        """
        return await role_repository.get_role_options(db, is_root=is_root)

    @staticmethod
    async def get_role_by_id(db: AsyncSession, role_id: int) -> SysRole | None:
        """
        根据ID获取角色信息

        Args:
            db: 异步数据库会话
            role_id: 角色ID

        Returns:
            角色对象，如果未找到返回None
        """
        return await role_repository.get_by_id(db, role_id)

    @staticmethod
    async def create_role(
        db: AsyncSession,
        redis: Redis,
        data: dict[str, Any],
    ) -> SysRole:
        """
        创建角色

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            data: 角色数据

        Returns:
            创建的角色对象

        Raises:
            BusinessException: 参数为空、编码格式错误或角色已存在
        """
        name = data.get("name")
        code = data.get("code")

        if not name or not code:
            raise BusinessException("角色名称和编码不能为空")

        # 校验角色编码格式：大写字母、数字、下划线
        if not re.match(r"^[A-Z][A-Z0-9_]*$", code):
            raise BusinessException("角色编码格式错误，必须以大写字母开头，只能包含大写字母、数字和下划线")

        # 检查角色名称是否已存在
        if await role_repository.check_name_exists(db, name):
            raise BusinessException("角色名称已存在")

        # 检查角色编码是否已存在
        if await role_repository.check_code_exists(db, code):
            raise BusinessException("角色编码已存在")

        role = SysRole(
            name=name,
            code=code,
            sort=data.get("sort", 0),
            status=data.get("status", 1),
            data_scope=data.get("dataScope", data.get("data_scope", 1)),
        )

        created = await role_repository.create(db, role)

        return created

    @staticmethod
    async def update_role(
        db: AsyncSession,
        redis: Redis,
        role_id: int,
        data: dict[str, Any],
    ) -> None:
        """
        更新角色信息

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            role_id: 角色ID
            data: 角色数据

        Raises:
            BusinessException: 角色不存在、名称为空或名称已存在
        """
        role = await RoleService.get_role_by_id(db, role_id)
        if not role:
            raise BusinessException("角色不存在")

        name = data.get("name")

        if not name:
            raise BusinessException("角色名称不能为空")

        # 检查角色名称是否已存在（排除自己）
        if await role_repository.check_name_exists(db, name, exclude_id=role_id):
            raise BusinessException("角色名称已存在")

        # 超级管理员角色保护：不可修改状态和数据权限
        update_data = {
            "name": name,
            "sort": data.get("sort", role.sort),
        }

        # 非超级管理员角色可以修改状态和数据权限
        if role.code != RoleService.ROOT_ROLE_CODE:
            update_data["status"] = data.get("status", role.status)
            update_data["data_scope"] = data.get(
                "dataScope", data.get("data_scope", role.data_scope))

        # 更新角色信息（不更新 code，编码创建后不可修改）
        await role_repository.update_by_id(db, role_id, update_data)

        # 清除角色权限缓存
        if role.code is None:
            raise ValueError("角色编码不能为空")
        await RoleService._clear_role_perms_cache(redis, role.code)

    @staticmethod
    async def delete_roles(
        db: AsyncSession,
        redis: Redis,
        ids: str,
    ) -> None:
        """
        删除角色（支持批量删除）

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            ids: 角色ID，多个以英文逗号分隔

        Raises:
            BusinessException: 角色不存在、为超级管理员或已分配给用户
        """
        role_ids = [int(id) for id in ids.split(",")]

        for role_id in role_ids:
            role = await RoleService.get_role_by_id(db, role_id)
            if not role:
                raise BusinessException(f"角色ID {role_id} 不存在")

            # 超级管理员角色保护：code='ROOT' 的角色不能删除
            if role.code == RoleService.ROOT_ROLE_CODE:
                raise BusinessException("超级管理员角色不可删除")

            # 检查角色是否已分配给用户
            user_count = await user_repository.count_users_by_role(db, role_id)
            if user_count > 0:
                raise BusinessException(f"角色【{role.name}】已分配给用户，请先解除关联后删除")

            # 删除角色-菜单关联记录（物理删除）
            await role_repository.delete_role_menus(db, role_id)

            # 逻辑删除角色
            await role_repository.delete(db, role_id)

            # 清除角色权限缓存
            if role.code is None:
                raise ValueError("角色编码不能为空")
            await RoleService._clear_role_perms_cache(redis, role.code)

    @staticmethod
    async def update_role_status(
        db: AsyncSession,
        redis: Redis,
        role_id: int,
        status: int,
    ) -> None:
        """
        更新角色状态

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            role_id: 角色ID
            status: 状态（1-启用，0-禁用）

        Raises:
            BusinessException: 状态值无效、角色不存在或为超级管理员
        """
        if status not in [0, 1]:
            raise BusinessException("状态值只能为0或1")

        role = await RoleService.get_role_by_id(db, role_id)
        if not role:
            raise BusinessException("角色不存在")

        # 超级管理员角色保护：code='ROOT' 的角色不能修改状态
        if role.code == RoleService.ROOT_ROLE_CODE:
            raise BusinessException("超级管理员角色不可禁用")

        await role_repository.update_by_id(db, role_id, {"status": status})

    @staticmethod
    async def get_role_menu_ids(db: AsyncSession, role_id: int) -> list[int]:
        """
        获取角色的菜单ID集合

        Args:
            db: 异步数据库会话
            role_id: 角色ID

        Returns:
            菜单ID列表
        """
        return await role_repository.get_role_menu_ids(db, role_id)

    @staticmethod
    async def get_maximum_data_scope(db: AsyncSession, roles: list[str]) -> int | None:
        """
        获取最大范围的数据权限

        Args:
            db: 异步数据库会话
            roles: 角色编码集合

        Returns:
            数据权限范围
        """
        return await role_repository.get_maximum_data_scope(db, roles)

    @staticmethod
    async def assign_menus_to_role(
        db: AsyncSession,
        redis: Redis,
        role_id: int,
        menu_ids: list[int],
    ) -> None:
        """
        分配菜单给角色

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            role_id: 角色ID
            menu_ids: 菜单ID列表

        Raises:
            BusinessException: 角色不存在
        """
        role = await RoleService.get_role_by_id(db, role_id)
        if not role:
            raise BusinessException("角色不存在")

        # 使用 repository 替换角色菜单
        await role_repository.replace_role_menus(db, role_id, menu_ids)

        # 清除角色权限缓存
        if role.code is None:
            raise ValueError("角色编码不能为空")
        await RoleService._clear_role_perms_cache(redis, role.code)

    @staticmethod
    async def _clear_role_perms_cache(redis: Redis, role_code: str):
        """
        清除角色权限缓存

        Args:
            redis: Redis 异步客户端
            role_code: 角色编码
        """
        cache_key = f"{RoleService.ROLE_PERMS_PREFIX}{role_code}"
        await redis.delete(cache_key)
