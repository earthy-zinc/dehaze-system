"""
部门服务

提供部门 CRUD 功能，支持树形结构
"""

from typing import Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.cache.cache import CACHE_TTL_HOUR, CacheService, DeptCacheKeys
from app.models.entity.sys_dept import SysDept
from app.models.schema.common import validate_no_xss
from app.repository.dept_repository import dept_repository
from app.repository.user_repository import user_repository
from app.utils.datetime_utils import format_time

# 根部门 ID（系统内置，不可删除/修改上级）
ROOT_DEPT_ID = 1

# 部门最大层级深度（T-DPT-014/018a：超出 5 级报 A0504"部门层级不能超过5级"）
MAX_DEPT_LEVEL = 5


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
                "sort": dept.sort,
                "status": dept.status,
                "createTime": format_time(dept.create_time),
                "updateTime": format_time(dept.update_time),
                "children": [],
            }
            for dept in dept_list
        }

        root_depts = []
        for dept in dept_dict.values():
            parent = dept_dict.get(dept["parentId"])
            # 数据权限过滤后父节点不在可见集时，提升为可见子树的根节点
            if dept["parentId"] == 0 or parent is None:
                root_depts.append(dept)
            else:
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

    def _assert_max_dept_depth(self, tree_path: str) -> None:
        """校验部门层级不超过 5 级（T-DPT-014/018a：超出报 A0504"部门层级不能超过5级"）"""
        # tree_path 格式: "0" 为第 1 级，"0,1" 为第 2 级
        if len(tree_path.split(",")) > MAX_DEPT_LEVEL:
            raise BusinessException(ResultCode.DATA_BIND_EXISTS, "部门层级不能超过5级")

    def _validate_name_safety(self, name: str) -> None:
        """部门名称安全性校验（复用公共 XSS 校验，报 A0400）"""
        try:
            validate_no_xss(name)
        except ValueError as e:
            raise BusinessException(ResultCode.PARAM_ERROR, str(e)) from e

    async def create_dept(
        self,
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
            raise BusinessException(ResultCode.PARAM_ERROR, "部门名称不能为空")
        self._validate_name_safety(name)

        if parent_id != 0:
            parent_dept = await dept_repository.get_by_id(db, parent_id)
            if not parent_dept:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "父部门不存在")

        if await dept_repository.check_name_exists(db, name, parent_id=parent_id):
            raise BusinessException(ResultCode.DATA_EXISTS, "部门名称已存在")

        tree_path = await dept_repository.generate_tree_path(db, parent_id)
        self._assert_max_dept_depth(tree_path)

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
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "部门不存在")

        # 1. 上级部门校验：根部门保护、自身引用、循环引用（不能移动到自身子树下）
        new_parent_id = data.get("parentId", dept.parent_id)
        if new_parent_id != dept.parent_id:
            if dept_id == ROOT_DEPT_ID:
                raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "根部门不可修改上级")
            if new_parent_id == dept_id:
                raise BusinessException(
                    ResultCode.OPERATION_NOT_ALLOW, "不能将部门设置为自己的上级部门"
                )
            if new_parent_id != 0:
                new_parent = await dept_repository.get_by_id(db, new_parent_id)
                if not new_parent:
                    raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "父部门不存在")
                # 父部门的 tree_path 包含当前部门ID → 父部门是当前部门的子部门 → 循环引用
                if new_parent.tree_path:
                    tree_path_with_commas = f",{new_parent.tree_path},"
                    if f",{dept_id}," in tree_path_with_commas:
                        raise BusinessException(
                            ResultCode.OPERATION_NOT_ALLOW,
                            "不能将部门移动到其子部门下，存在循环引用",
                        )

            new_tree_path = await dept_repository.generate_tree_path(db, new_parent_id)
            # 移动后层级校验（T-DPT-018a：移动至超深层级报 A0504"部门层级不能超过5级"）
            self._assert_max_dept_depth(new_tree_path)
            old_tree_path = dept.tree_path
            dept.tree_path = new_tree_path
            dept.parent_id = new_parent_id
            # 级联平移子树路径，保持不变量：子.tree_path == 父.tree_path + "," + 父.id
            await dept_repository.update_subtree_tree_path(
                db, f"{old_tree_path},{dept_id}", f"{new_tree_path},{dept_id}"
            )

        # 2. 名称唯一性：同一上级部门下唯一（含已删除记录，删除后名称不可复用）
        name = data.get("name")
        if name:
            if await dept_repository.check_name_exists(
                db, name, parent_id=new_parent_id, exclude_id=dept_id
            ):
                raise BusinessException(ResultCode.DATA_EXISTS, "部门名称已存在")
            dept.name = name

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
            BusinessException: 部门不存在（A0401）、根部门（A0503）、
                有子部门（A0502）或有关联用户（A0502）时禁止删除
        """
        if not dept_ids:
            raise BusinessException(ResultCode.PARAM_ERROR, "未指定要删除的部门")

        # 批量预取并加行锁，防止并发删除与关联校验间的 TOCTOU
        depts_map = {
            int(d.id): d for d in await dept_repository.get_by_ids(db, dept_ids, for_update=True)
        }
        for dept_id in dept_ids:
            if dept_id not in depts_map:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "部门不存在")
            # 根部门保护
            if dept_id == ROOT_DEPT_ID:
                raise BusinessException(ResultCode.OPERATION_NOT_ALLOW, "根部门不可删除")

        # 子部门检查：有子部门禁止删除（T-DPT-030，不级联删除，A0502）
        child_counts = await dept_repository.count_children_by_parents(db, dept_ids)
        if any(child_counts.get(dept_id, 0) > 0 for dept_id in dept_ids):
            raise BusinessException(
                ResultCode.DATA_STATE_NOT_ALLOW, "该部门下存在子部门，请先删除子部门"
            )

        # 关联用户检查：有用户禁止删除（T-DPT-029，A0502）
        user_counts = await user_repository.count_users_by_depts(db, dept_ids)
        if any(user_counts.get(dept_id, 0) > 0 for dept_id in dept_ids):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "该部门下存在用户，无法删除")

        # 逻辑删除指定部门（不含子部门，子部门已被前置校验拦截）
        deleted_count = await dept_repository.soft_delete_by_ids(db, dept_ids)
        if deleted_count == 0:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "部门不存在")

        await self._clear_cache(redis)

    async def _clear_cache(self, redis: Redis) -> None:
        """清除部门相关缓存"""
        cache = CacheService(redis)
        for pattern in DeptCacheKeys.all_patterns():
            await cache.delete_pattern(pattern)


dept_service = DeptService()
