"""
部门服务

提供部门 CRUD 功能，支持树形结构
"""

from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import (CACHE_TTL_HOUR, CacheService,
                                            DeptCacheKeys)
from app.models.entity.sys_dept import SysDept
from app.repository.dept_repository import dept_repository
from app.repository.user_repository import user_repository
from app.utils.datetime_utils import format_time

# 部门层级深度限制（设计文档要求：不超过 5 级）
MAX_DEPT_DEPTH = getattr(settings, "DEPT_MAX_DEPTH", 5)

# 根部门 ID（系统内置，不可修改/删除）
ROOT_DEPT_ID = 1


class DeptService:
    """部门服务（异步版本）"""

    @staticmethod
    def _build_dept_tree(dept_list: list[SysDept]) -> list[dict[str, Any]]:
        """构建部门树形结构"""
        if not dept_list:
            return []

        # 构建部门字典
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

        # 构建父子关系
        root_depts = []
        for dept in dept_dict.values():
            if dept["parentId"] == 0:
                root_depts.append(dept)
            else:
                parent = dept_dict.get(dept["parentId"])
                if parent:
                    parent["children"].append(dept)

        return root_depts

    @staticmethod
    async def get_dept_list(
        db: AsyncSession,
        keywords: str | None = None,
        status: int | None = None,
    ) -> list[dict[str, Any]]:
        """获取部门列表（树形结构）"""
        dept_list = await dept_repository.get_dept_list(db, keywords=keywords, status=status)
        return DeptService._build_dept_tree(dept_list)

    @staticmethod
    async def get_dept_options(
        db: AsyncSession,
        redis: Redis,
    ) -> list[dict[str, Any]]:
        """获取部门下拉选项（树形结构，带缓存）"""
        cache = CacheService(redis)

        # 尝试从缓存获取
        cached = await cache.get_json(DeptCacheKeys.OPTIONS)
        if cached is not None:
            return cached

        # 从数据库获取
        options = await dept_repository.get_dept_options_tree(db)

        # 写入缓存
        await cache.set_json(DeptCacheKeys.OPTIONS, options, CACHE_TTL_HOUR)

        return options

    @staticmethod
    async def get_dept_form(db: AsyncSession, dept_id: int) -> dict[str, Any] | None:
        """获取部门表单数据"""
        return await dept_repository.get_dept_form(db, dept_id)

    @staticmethod
    async def _calculate_depth(tree_path: str) -> int:
        """计算部门层级深度"""
        if not tree_path or tree_path == "0":
            return 1
        # tree_path 格式: "0,1,2,3"
        return len(tree_path.split(","))

    @staticmethod
    async def create_dept(
        db: AsyncSession,
        redis: Redis,
        data: dict[str, Any],
    ) -> int:
        """
        新增部门

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

        # 1. 检查部门名称是否已存在（同层级内）
        if await dept_repository.check_name_exists(db, name, parent_id=parent_id):
            raise BusinessException("同一层级下部门名称已存在")

        # 2. 检查上级部门存在性
        if parent_id != 0:
            parent_dept = await dept_repository.get_by_id(db, parent_id)
            if not parent_dept:
                raise BusinessException("上级部门不存在")

            # 3. 检查层级深度限制
            new_depth = await DeptService._calculate_depth(parent_dept.tree_path) + 1
            if new_depth > MAX_DEPT_DEPTH:
                raise BusinessException(f"部门层级不能超过 {MAX_DEPT_DEPTH} 级")

        # 生成 tree_path
        tree_path = await dept_repository.generate_tree_path(db, parent_id)

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

        # 清除缓存
        await DeptService._clear_cache(redis)

        return dept.id

    @staticmethod
    async def update_dept(
        db: AsyncSession,
        redis: Redis,
        dept_id: int,
        data: dict[str, Any],
    ) -> int:
        """
        更新部门

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

        # 1. 根部门保护：不可修改 parentId
        if dept_id == ROOT_DEPT_ID and "parentId" in data and data["parentId"] != 0:
            raise BusinessException("根部门不可修改上级部门")

        # 2. 检查部门名称唯一性（同层级内）
        name = data.get("name")
        if name:
            parent_id = data.get("parentId", dept.parent_id)
            if await dept_repository.check_name_exists(db, name, exclude_id=dept_id, parent_id=parent_id):
                raise BusinessException("同一层级下部门名称已存在")

        # 3. 循环引用检测
        if "parentId" in data:
            new_parent_id = data["parentId"]
            if new_parent_id != dept.parent_id:
                # 不能将部门移动到自身下
                if new_parent_id == dept_id:
                    raise BusinessException("不能将部门移动到自身下")

                # 不能将部门移动到子部门下
                child_ids = await dept_repository.get_children_ids(db, dept_id)
                if new_parent_id in child_ids:
                    raise BusinessException("不能将部门移动到子部门下")

                # 检查上级部门存在性
                if new_parent_id != 0:
                    new_parent = await dept_repository.get_by_id(db, new_parent_id)
                    if not new_parent:
                        raise BusinessException("上级部门不存在")

                    # 检查层级深度限制
                    current_child_depth = await DeptService._calculate_depth(dept.tree_path or "0")
                    new_parent_depth = await DeptService._calculate_depth(new_parent.tree_path or "0")
                    max_child_depth = await dept_repository.get_max_child_depth(db, dept_id)
                    depth_diff = max_child_depth - current_child_depth if max_child_depth > 0 else 0

                    if new_parent_depth + depth_diff + 1 > MAX_DEPT_DEPTH:
                        raise BusinessException(
                            f"移动后部门层级将超过 {MAX_DEPT_DEPTH} 级限制")

                # 更新 tree_path
                dept.tree_path = await dept_repository.generate_tree_path(db, new_parent_id)
                dept.parent_id = new_parent_id

        # 更新其他字段
        if "name" in data:
            dept.name = data["name"]
        if "status" in data:
            dept.status = data["status"]
        if "sort" in data:
            dept.sort = data["sort"]

        # 清除缓存
        await DeptService._clear_cache(redis)

        return dept.id

    @staticmethod
    async def delete_depts(
        db: AsyncSession,
        redis: Redis,
        dept_ids: list[int],
    ) -> None:
        """
        删除部门

        Args:
            db: 异步数据库会话
            redis: Redis 客户端
            dept_ids: 部门ID列表

        Raises:
            BusinessException: 业务校验失败
        """
        if not dept_ids:
            raise BusinessException("未指定要删除的部门")

        # 1. 根部门保护
        if ROOT_DEPT_ID in dept_ids:
            raise BusinessException("根部门不可删除")

        # 2. 检查是否存在关联用户
        for dept_id in dept_ids:
            user_count = await user_repository.count_users_by_dept(db, dept_id)
            if user_count > 0:
                dept = await dept_repository.get_by_id(db, dept_id)
                dept_name = dept.name if dept else f"ID={dept_id}"
                raise BusinessException(
                    f"部门【{dept_name}】下存在 {user_count} 个用户，无法删除")

        # 3. 检查是否存在子部门（设计文档要求：暂不实现级联删除）
        for dept_id in dept_ids:
            child_count = await dept_repository.count_children(db, dept_id)
            if child_count > 0:
                dept = await dept_repository.get_by_id(db, dept_id)
                dept_name = dept.name if dept else f"ID={dept_id}"
                raise BusinessException(f"部门【{dept_name}】下存在子部门，请先删除子部门")

        # 执行删除
        await dept_repository.delete_depts(db, dept_ids)

        # 清除缓存
        await DeptService._clear_cache(redis)

    @staticmethod
    async def _clear_cache(redis: Redis) -> None:
        """清除部门相关缓存"""
        cache = CacheService(redis)
        for pattern in DeptCacheKeys.all_patterns():
            await cache.delete_pattern(pattern)
