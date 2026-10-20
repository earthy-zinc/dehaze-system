"""
角色服务

提供角色 CRUD 功能，支持菜单分配和数据权限管理
"""

from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import defer_after_commit
from app.dependencies.auth import UserContext
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService
from app.models.base import get_current_user_id
from app.models.entity.sys_user import SysRole
from app.repository.menu_repository import menu_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.role_repository import role_repository
from app.repository.user_repository import user_repository


class RoleService:
    """角色服务"""

    # 缓存常量
    ROLE_PERMS_PREFIX = "role:perms:"
    ROLE_OPTIONS_CACHE_KEY = "role:options"
    ROLE_OPTIONS_CACHE_TTL = CACHE_TTL_HOUR

    # 内置不可删除的角色编码（与 Java/Go 一致）
    BUILTIN_ROLE_CODES = {"ROOT", "ADMIN"}

    async def get_role_list(
        self,
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
                ("name", "like", f"%{keywords}%"),
                ("code", "like", f"%{keywords}%"),
            ]

        return await role_repository.get_list(
            db,
            filters=filters,
            search_fields=search_fields,
            order_by="sort",
            page=page,
            page_size=page_size,
        )

    async def get_role_options(
        self, db: AsyncSession, redis: Redis, *, is_root: bool = False
    ) -> list[dict[str, Any]]:
        """
        获取角色下拉选项列表（role:options 缓存 TTL 1h，角色增删改后失效）

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            is_root: 当前用户是否为超级管理员（非超级管理员不显示内置角色）

        Returns:
            角色下拉选项列表
        """
        cache = CacheService(redis)
        options = (
            await cache.get_json_with_loader(
                self.ROLE_OPTIONS_CACHE_KEY,
                lambda: role_repository.get_role_options(db),
                ttl=self.ROLE_OPTIONS_CACHE_TTL,
            )
            or []
        )
        if not is_root:
            options = [opt for opt in options if opt.get("code") not in self.BUILTIN_ROLE_CODES]
        return [{"value": opt["value"], "label": opt["label"]} for opt in options]

    async def get_role_by_id(self, db: AsyncSession, role_id: int) -> SysRole | None:
        """
        根据ID获取角色信息

        Args:
            db: 异步数据库会话
            role_id: 角色ID

        Returns:
            角色对象，如果未找到返回None
        """
        return await role_repository.get_by_id(db, role_id)

    async def create_role(
        self,
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
            raise BusinessException(ResultCode.PARAM_ERROR, "角色名称和编码不能为空")

        # 新增时 dataScope 必填（T-RM-012：数据权限未选择报"数据权限不能为空" A0400）
        if data.get("dataScope") is None:
            raise BusinessException(ResultCode.PARAM_ERROR, "数据权限不能为空")

        # 检查角色名称是否已存在（含软删记录）
        if await role_repository.check_name_exists(db, name):
            raise BusinessException(ResultCode.DATA_EXISTS, "角色名称已存在")

        # 检查角色编码是否已存在（含软删记录）
        if await role_repository.check_code_exists(db, code):
            raise BusinessException(ResultCode.DATA_EXISTS, "角色编码已存在")

        role = SysRole(
            name=name,
            code=code,
            sort=data.get("sort", 0),
            status=data.get("status", 1),
            data_scope=data["dataScope"],
        )

        created = await role_repository.create(db, role)

        self._defer_invalidate_role_options(db, redis)

        return created

    async def update_role(
        self,
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
        role = await self.get_role_by_id(db, role_id)
        if not role:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在")

        name = data.get("name")
        code = data.get("code")

        if not name:
            raise BusinessException(ResultCode.PARAM_ERROR, "角色名称不能为空")

        # 角色编码创建后不可修改（与 Go/Java 一致，优先检查）
        if code and code != role.code:
            raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "角色编码不可修改")

        # 检查角色名称是否已存在（排除自己，含软删记录）
        if await role_repository.check_name_exists(db, name, exclude_id=role_id):
            raise BusinessException(ResultCode.DATA_EXISTS, "角色名称已存在")

        # 检查角色编码是否已存在（排除自己，含软删记录）
        if code and await role_repository.check_code_exists(db, code, exclude_id=role_id):
            raise BusinessException(ResultCode.DATA_EXISTS, "角色编码已存在")

        # 超级管理员角色保护：不可修改状态和数据权限
        update_data = {
            "name": name,
            "code": code or role.code,
            "sort": data.get("sort", role.sort),
        }

        # 内置角色不可修改状态和数据权限（与 Java/Go 一致）
        if role.code not in self.BUILTIN_ROLE_CODES:
            update_data["status"] = data.get("status", role.status)
            # dataScope 去除 schema 默认值后可为 None（未随请求提交），此时保持原值
            data_scope = data.get("dataScope")
            if data_scope is not None:
                update_data["data_scope"] = data_scope

        # 更新角色信息（不更新 code，编码创建后不可修改）
        if role.code is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "角色编码不能为空")
        await role_repository.update_by_id(db, role_id, update_data)

        self._defer_clear_role_perms(db, redis, role.code)
        self._defer_invalidate_role_options(db, redis)

    async def delete_roles(
        self,
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
        ids = ids.strip()
        try:
            role_ids = [int(part) for part in ids.split(",") if part.strip()]
        except ValueError:
            raise BusinessException(
                ResultCode.PARAM_ERROR, "角色ID格式错误，应为英文逗号分隔的数字"
            ) from None
        if not role_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "角色ID不能为空")

        # 批量查询角色（避免 N+1）
        roles = await role_repository.get_by_ids(db, role_ids)
        roles_map = {int(r.id): r for r in roles}

        for role_id in role_ids:
            role = roles_map.get(role_id)
            if not role:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"角色ID {role_id} 不存在")

            # 内置角色保护：与 Java/Go 一致，ROOT 和 ADMIN 均不可删除
            if role.code in self.BUILTIN_ROLE_CODES:
                raise BusinessException(
                    ResultCode.OPERATION_NOT_ALLOW, f"内置角色 '{role.code}' 不可删除"
                )

            if role.code is None:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "角色编码不能为空")

        # 批量检查角色是否已分配给用户（避免 N+1）
        user_counts = await user_repository.count_users_by_roles(db, role_ids)
        for role_id in role_ids:
            role = roles_map[role_id]
            count = user_counts.get(role_id, 0)
            if count > 0:
                raise BusinessException(ResultCode.BUSINESS_ERROR, "该角色仍有用户关联，请先解绑")

        # 批量清理角色-菜单关联 + 批量软删除角色（2 条 SQL，替代 2N 条）
        await role_repository.delete_role_menus_by_role_ids(db, role_ids)
        await role_repository.delete_by_ids(db, role_ids)

        # 提交后清理：权限缓存 + 选项缓存
        # （无需踢出关联用户：删除前置校验已确保无活跃用户关联）
        for role in roles:
            self._defer_clear_role_perms(db, redis, role.code)
        self._defer_invalidate_role_options(db, redis)

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="role",
            target_id=ids,
            action="delete",
            module="role",
        )

    async def update_role_status(
        self,
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
            raise BusinessException(ResultCode.PARAM_ERROR, "状态值只能为0或1")

        role = await self.get_role_by_id(db, role_id)
        if not role:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在")

        # 内置角色不可修改状态（与 Java/Go 一致）
        if role.code in self.BUILTIN_ROLE_CODES:
            raise BusinessException(
                ResultCode.OPERATION_NOT_ALLOW, f"内置角色 '{role.code}' 不可修改状态"
            )

        await role_repository.update_by_id(db, role_id, {"status": status})

        # 状态变更同步失效角色权限缓存（与 Java refreshRolePermsCache 对齐）
        if role.code:
            self._defer_clear_role_perms(db, redis, role.code)
        self._defer_invalidate_role_options(db, redis)

        # 禁用角色时踢出关联在线用户（权限传播）；启用不踢
        if status == 0:
            self._defer_kick_role_users(db, redis, [role_id])

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="role",
            target_id=role_id,
            action="status_change",
            module="role",
            after_value=status,
        )

    async def get_role_menu_ids(self, db: AsyncSession, role_id: int) -> list[int]:
        """
        获取角色的菜单ID集合

        Args:
            db: 异步数据库会话
            role_id: 角色ID

        Returns:
            菜单ID列表
        """
        return await role_repository.get_role_menu_ids(db, role_id)

    async def assign_menus_to_role(
        self,
        db: AsyncSession,
        redis: Redis,
        role_id: int,
        menu_ids: list[int],
        operator: UserContext,
    ) -> None:
        """
        分配菜单给角色

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            role_id: 角色ID
            menu_ids: 菜单ID列表
            operator: 操作者上下文（权限提升校验用）

        Raises:
            BusinessException: 角色不存在、菜单不存在或操作者试图授予自身未持有的权限
        """
        role = await self.get_role_by_id(db, role_id)
        if not role:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在")

        if role.code is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "角色编码不能为空")

        # 校验分配的菜单（含按钮/接口节点）必须真实存在
        # （T-RM-036：分配不存在的菜单报"菜单不存在" A0401）
        if menu_ids:
            exist_count = await menu_repository.count_by_ids(db, menu_ids)
            if exist_count != len(menu_ids):
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "菜单不存在")

        # 权限提升防护（A0301）：操作者不能授予自己未持有的权限标识；ROOT 忽略权限判断
        if menu_ids and not operator.is_root:
            assigned_perms = await menu_repository.get_perms_by_ids(db, menu_ids)
            missing = assigned_perms - set(operator.permissions)
            if missing:
                raise BusinessException(
                    ResultCode.ACCESS_UNAUTHORIZED,
                    "无权分配超出自身权限范围的权限标识",
                )

        # 使用 repository 替换角色菜单
        await role_repository.replace_role_menus(db, role_id, menu_ids)

        self._defer_clear_role_perms(db, redis, role.code)
        self._defer_kick_role_users(db, redis, [role_id])

        mongo_audit_log_repository.create_audit_async(
            operator_id=get_current_user_id(),
            target_type="role",
            target_id=role_id,
            action="update",
            module="role",
            after_value=menu_ids,
        )

    def _defer_clear_role_perms(
        self, db: AsyncSession, redis: Redis, role_code: str | None
    ) -> None:
        """登记角色权限缓存失效（必须经 CacheService 双清 L1+L2，且在事务提交后执行）。

        角色 code 为空属异常（各写路径均已前置校验非空）；此时无有效缓存键可清理，
        若静默跳过会致权限缓存陈旧，故显式报错暴露，而非默默略过。
        """
        if role_code is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "角色编码不能为空，无法清理权限缓存")
        cache_key = f"{self.ROLE_PERMS_PREFIX}{role_code}"
        defer_after_commit(db, lambda: CacheService(redis).delete(cache_key))

    def _defer_invalidate_role_options(self, db: AsyncSession, redis: Redis) -> None:
        """登记角色选项缓存失效（走 CacheService，与 role:perms 同为提交后执行）"""
        defer_after_commit(db, lambda: CacheService(redis).delete(self.ROLE_OPTIONS_CACHE_KEY))

    def _defer_kick_role_users(self, db: AsyncSession, redis: Redis, role_ids: list[int]) -> None:
        """登记踢出关联角色的在线用户（权限传播，事务提交后反查活跃用户执行）"""

        async def _kick():
            from app.service.auth_service import auth_service

            user_ids = await role_repository.list_user_ids_by_roles(db, role_ids)
            await auth_service.kick_user_sessions_batch(redis, user_ids)

        defer_after_commit(db, _kick)


role_service = RoleService()
