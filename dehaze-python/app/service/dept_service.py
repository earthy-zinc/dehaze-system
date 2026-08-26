"""
部门服务

提供部门 CRUD 功能，支持树形结构
"""

import re
from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService, DeptCacheKeys
from app.models.entity.sys_dept import SysDept
from app.repository.dept_repository import dept_repository
from app.repository.user_repository import user_repository
from app.utils.datetime_utils import format_time

# 根部门 ID（系统内置，不可修改/删除）
ROOT_DEPT_ID = 1

# 部门最大层级深度（T-DPT-014/018a：超出 5 级报 A0504"部门层级不能超过5级"）
MAX_DEPT_LEVEL = 5

# XSS 危险模式：HTML 标签起始、javascript 协议、事件处理器（onXxx=）
# 匹配 Java XssUtils 的安全防护意图，拦截 XSS 注入
_XSS_PATTERN = re.compile(
    r"<\s*/?\s*[a-zA-Z]|javascript:\s*|on\w+\s*=",
    re.IGNORECASE,
)


class DeptService:
    """部门服务"""

    def _build_dept_tree(self, dept_list: list[SysDept]) -> list[dict[str, Any]]:
        """构建部门树形结构"""
        if not dept_list:
            return []

        dept_dict = {
            dept.id: {
                "id": dept.id,
                "name": dept.name,
                "parentId": dept.parent_id,
                "treePath": dept.tree_path,
                "sort": dept.sort,
                "status": dept.status,
                "deleted": dept.deleted,
                "createTime": format_time(dept.create_time),
                "updateTime": format_time(dept.update_time),
                "children": [],
            }
            for dept in dept_list
        }

        root_depts = []
        for dept in dept_dict.values():
            if dept["parentId"] == 0:
                root_depts.append(dept)
            else:
                parent = dept_dict.get(dept["parentId"])
                if parent:
                    parent["children"].append(dept)

        return root_depts

    async def get_dept_list(
        self,
        db: AsyncSession,
        keywords: str | None = None,
        status: int | None = None,
        current_user=None,
    ) -> list[dict[str, Any]]:
        """获取部门列表（树形结构，按 current_user 行级数据权限过滤）"""
        dept_list = await dept_repository.get_dept_list(
            db, keywords=keywords, status=status, current_user=current_user
        )
        return self._build_dept_tree(dept_list)

    async def get_dept_options(
        self,
        db: AsyncSession,
        redis: Redis,
        current_user=None,
    ) -> list[dict[str, Any]]:
        """获取部门下拉选项（树形结构，带缓存）

        缓存仅对全量视图（ROOT / 全部数据权限）生效；行级过滤结果因人而异，不读写缓存。
        """
        # ROOT 或 data_scope 为空/0 时无行级过滤，全量结果可共享缓存
        cacheable = (
            current_user is None
            or current_user.is_root
            or current_user.data_scope is None
            or current_user.data_scope == 0
        )
        cache = CacheService(redis)

        if cacheable:
            cached = await cache.get_json(DeptCacheKeys.OPTIONS)
            if cached is not None:
                return cached

        options = await dept_repository.get_dept_options_tree(db, current_user=current_user)

        if cacheable:
            await cache.set_json(DeptCacheKeys.OPTIONS, options, CACHE_TTL_HOUR)

        return options

    async def get_dept_form(self, db: AsyncSession, dept_id: int) -> dict[str, Any] | None:
        """获取部门表单数据"""
        return await dept_repository.get_dept_form(db, dept_id)

    async def _calculate_depth(self, tree_path: str) -> int:
        """计算部门层级深度"""
        if not tree_path or tree_path == "0":
            return 1
        # tree_path 格式: "0,1,2,3"
        return len(tree_path.split(","))

    async def _assert_max_dept_depth(self, tree_path: str) -> None:
        """校验部门层级不超过 5 级（T-DPT-014/018a：超出报 A0504"部门层级不能超过5级"）"""
        depth = await self._calculate_depth(tree_path)
        if depth > MAX_DEPT_LEVEL:
            raise BusinessException(ResultCode.DATA_BIND_EXISTS, "部门层级不能超过5级")

    def _validate_name_safety(self, name: str) -> None:
        """
        校验部门名称安全性，拦截 XSS 攻击

        检测 HTML 标签起始、javascript 协议、事件处理器等危险模式，
        匹配 Java XssUtils 的安全防护意图。

        Args:
            name: 部门名称

        Raises:
            BusinessException: 名称包含 XSS 攻击模式（PARAM_ERROR）
        """
        if name and _XSS_PATTERN.search(name):
            raise BusinessException(ResultCode.PARAM_ERROR, "部门名称包含不安全的字符")

    async def create_dept(
        self,
        db: AsyncSession,
        redis: Redis,
        data: dict[str, Any],
    ) -> int:
        """
        新增部门（匹配 Java SysDeptServiceImpl.saveDept 逻辑）

        Args:
            db: 异步数据库会话
            redis: Redis 客户端
            data: 部门数据

        Returns:
            创建的部门ID

        Raises:
            BusinessException: 业务校验失败
        """
        name = data.get("name")
        parent_id = data.get("parentId", 0)

        if not name:
            raise BusinessException("部门名称不能为空")

        self._validate_name_safety(name)

        if await dept_repository.check_name_exists(db, name):
            raise BusinessException("部门名称已存在")

        if parent_id != 0:
            parent_dept = await dept_repository.get_by_id(db, parent_id)
            if not parent_dept:
                raise BusinessException("父部门不存在")

        tree_path = await dept_repository.generate_tree_path(db, parent_id)

        await self._assert_max_dept_depth(tree_path)

        dept = SysDept(
            name=name,
            parent_id=parent_id,
            status=data.get("status", 1),
            sort=data.get("sort", 1),
            tree_path=tree_path,
        )

        db.add(dept)
        await db.flush()
        await db.refresh(dept)

        await self._clear_cache(redis)

        return dept.id

    async def update_dept(
        self,
        db: AsyncSession,
        redis: Redis,
        dept_id: int,
        data: dict[str, Any],
    ) -> int:
        """
        更新部门（匹配 Java SysDeptServiceImpl.updateDept 逻辑）

        Args:
            db: 异步数据库会话
            redis: Redis 客户端
            dept_id: 部门ID
            data: 部门数据

        Returns:
            更新的部门ID

        Raises:
            BusinessException: 业务校验失败
        """
        dept = await dept_repository.get_by_id(db, dept_id)
        if not dept:
            raise BusinessException("部门不存在")

        # 1. 校验部门名称是否存在（全局，匹配 Java）
        name = data.get("name")
        if name:
            self._validate_name_safety(name)
            if await dept_repository.check_name_exists(db, name, exclude_id=dept_id):
                raise BusinessException("部门名称已存在")

        # 2. 循环引用校验：不能将部门移动到自身或其子部门下（匹配 Java）
        if "parentId" in data:
            new_parent_id = data["parentId"]
            if new_parent_id == dept_id:
                raise BusinessException("不能将部门设置为自己的上级部门")

            if new_parent_id != 0:
                new_parent = await dept_repository.get_by_id(db, new_parent_id)
                if not new_parent:
                    raise BusinessException("父部门不存在")
                # 父部门的 tree_path 包含当前部门ID → 父部门是当前部门的子部门 → 循环引用
                if new_parent.tree_path:
                    tree_path_with_commas = f",{new_parent.tree_path},"
                    if f",{dept_id}," in tree_path_with_commas:
                        raise BusinessException("不能将部门移动到其子部门下，存在循环引用")

            new_tree_path = await dept_repository.generate_tree_path(db, new_parent_id)
            # 移动后层级校验（T-DPT-018a：移动至超深层级报 A0504"部门层级不能超过5级"）
            await self._assert_max_dept_depth(new_tree_path)
            dept.tree_path = new_tree_path
            dept.parent_id = new_parent_id

        if "name" in data:
            dept.name = data["name"]
        if "status" in data:
            dept.status = data["status"]
        if "sort" in data:
            dept.sort = data["sort"]

        await self._clear_cache(redis)

        return dept.id

    async def delete_depts(
        self,
        db: AsyncSession,
        redis: Redis,
        dept_ids: list[int],
    ) -> None:
        """
        删除部门（有子部门/关联用户则禁止删除，不级联删除，匹配 T-DPT-029/030）

        Args:
            db: 异步数据库会话
            redis: Redis 客户端
            dept_ids: 部门ID列表

        Raises:
            BusinessException: 有子部门（A0502）或有关联用户（A0502）时禁止删除
        """
        if not dept_ids:
            raise BusinessException("未指定要删除的部门")

        # 1. 根部门保护
        if ROOT_DEPT_ID in dept_ids:
            raise BusinessException("根部门不可删除")

        # 批量预取部门信息（避免错误路径逐条查询触发 N+1）
        depts_map = {int(d.id): d for d in await dept_repository.get_by_ids(db, dept_ids)}

        # 2. 子部门检查：有子部门禁止删除（T-DPT-030，不级联删除，A0502）
        child_counts = await dept_repository.count_children_by_parents(db, dept_ids)
        for dept_id in dept_ids:
            if child_counts.get(dept_id, 0) > 0:
                dept = depts_map.get(dept_id)
                dept_name = dept.name if dept else f"ID={dept_id}"
                raise BusinessException(
                    ResultCode.DATA_STATE_NOT_ALLOW, "该部门下存在子部门，请先删除子部门"
                )

        # 3. 关联用户检查：有用户禁止删除（T-DPT-029，A0502）
        user_counts = await user_repository.count_users_by_depts(db, dept_ids)
        for dept_id in dept_ids:
            count = user_counts.get(dept_id, 0)
            if count > 0:
                dept = depts_map.get(dept_id)
                dept_name = dept.name if dept else f"ID={dept_id}"
                raise BusinessException(
                    ResultCode.DATA_STATE_NOT_ALLOW, f"该部门下存在用户，无法删除"
                )

        # 4. 逻辑删除指定部门（不含子部门，子部门已被前置校验拦截）
        deleted_count = await dept_repository.soft_delete_by_ids(db, dept_ids)
        if deleted_count == 0:
            raise BusinessException("部门删除失败")

        await self._clear_cache(redis)

    async def _clear_cache(self, redis: Redis) -> None:
        """清除部门相关缓存"""
        cache = CacheService(redis)
        for pattern in DeptCacheKeys.all_patterns():
            await cache.delete_pattern(pattern)


dept_service = DeptService()
