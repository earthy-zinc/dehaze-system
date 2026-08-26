"""
行级数据权限过滤助手

与 Java（MyBatis-Plus DataPermissionInterceptor）和 Go（GORM DataScopePlugin）对齐，
采用显式过滤方案：在需要数据权限的 Repository 查询中显式调用 apply_data_scope，
按当前用户 data_scope 追加 WHERE 条件。

data_scope 取值（与 sys_role.data_scope 注释、Go DataScope* 常量一致）：
    0  全部数据（DataScopeAll）         → 原样返回
    1  部门及子部门数据（DataScopeDeptTree） → WHERE dept_field IN (本部门及子部门ID)
    2  本部门数据（DataScopeDept）       → WHERE dept_field == 本部门ID
    3  本人数据（DataScopeSelf）         → WHERE creator_field == 当前用户ID

ROOT 用户（roles 含 "ROOT"）跳过过滤。

说明：
- 异步 ORM 下无法可靠地在 event 回调中获取当前请求用户上下文（ContextVar 不可靠），
  因此采用显式调用而非自动 SQL 改写。
- 对于无 dept_id 字段的业务表（如订单、反馈），通过 create_by 关联用户实现"本人"过滤；
  "本部门"过滤需调用方在查询中 JOIN sys_user 取 dept_id，此处不自动注入 JOIN。
"""

from typing import Any

from sqlalchemy import Select, false
from sqlalchemy.orm import InstrumentedAttribute

from app.dependencies.auth import UserContext


async def apply_data_scope(
    stmt: Select,
    user: UserContext,
    db,
    *,
    dept_field: InstrumentedAttribute[Any] | None = None,
    creator_field: InstrumentedAttribute[Any] | None = None,
    children_ids: list[int] | None = None,
) -> Select:
    """
    按用户 data_scope 为查询追加行级过滤条件（纯函数，不依赖 dept_repository）

    Args:
        stmt: 原始查询语句
        user: 当前用户上下文
        db: 异步数据库会话（保留以兼容签名，当前未直接使用）
        dept_field: 部门字段表达式（如 SysUser.dept_id），无部门字段的表传 None
        creator_field: 创建人字段表达式（如 SysOrder.create_by），用于"本人"过滤
        children_ids: data_scope=1（部门及子部门）时由调用方注入的部门子树 ID 列表；
            避免本模块反向依赖 dept_repository 形成循环引用

    Returns:
        附加条件后的查询语句；ROOT 或 data_scope=0 时原样返回

    Raises:
        ValueError: data_scope 取值需要 dept_field 但未提供时
    """
    if user.is_root:
        return stmt

    data_scope = user.data_scope
    if data_scope is None or data_scope == 0:
        return stmt

    if data_scope == 3:
        # 本人数据
        if creator_field is None:
            raise ValueError("data_scope=3（本人）需要提供 creator_field")
        return stmt.where(creator_field == user.id)

    if data_scope == 2:
        # 本部门数据
        if dept_field is None:
            raise ValueError("data_scope=2（本部门）需要提供 dept_field")
        if user.dept_id is None:
            return stmt.where(false())  # 无部门则返回空集
        return stmt.where(dept_field == user.dept_id)

    if data_scope == 1:
        # 部门及子部门数据
        if dept_field is None:
            raise ValueError("data_scope=1（部门及子部门）需要提供 dept_field")
        if user.dept_id is None:
            return stmt.where(false())
        if not children_ids:
            return stmt.where(false())
        return stmt.where(dept_field.in_(children_ids))

    # 未知 data_scope 取值，保守返回空集
    return stmt.where(false())
