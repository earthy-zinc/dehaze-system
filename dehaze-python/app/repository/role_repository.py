"""
角色数据访问层
"""

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_audit_update_values
from app.models.entity.sys_menu import SysRoleMenu
from app.models.entity.sys_user import SysRole
from app.repository.base import BaseRepository

# 内置不可删除的角色编码（与 Java/Go 一致）
BUILTIN_ROLE_CODES = {"ROOT", "ADMIN"}


class RoleRepository(BaseRepository[SysRole]):
    """角色数据访问层"""

    model = SysRole

    async def get_list(
        self,
        db: AsyncSession,
        *,
        filters: dict | None = None,
        search_fields: list | None = None,
        order_by: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> tuple[list[SysRole], int]:
        """获取角色分页列表"""
        stmt = select(SysRole)

        if filters:
            for key, value in filters.items():
                if hasattr(SysRole, key):
                    stmt = stmt.where(getattr(SysRole, key) == value)

        if search_fields:
            from sqlalchemy import or_
            conditions = []
            for field, op, val in search_fields:
                col = getattr(SysRole, field, None)
                if col is not None:
                    if op == "like":
                        conditions.append(col.like(val))
            if conditions:
                stmt = stmt.where(or_(*conditions))

        if order_by and hasattr(SysRole, order_by):
            stmt = stmt.order_by(getattr(SysRole, order_by))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())

        return items, total

    async def update_by_id(
        self,
        db: AsyncSession,
        role_id: int,
        data: dict,
    ) -> SysRole | None:
        """根据 ID 更新角色"""
        stmt = select(SysRole).where(SysRole.id == role_id)
        result = await db.execute(stmt)
        role = result.scalar_one_or_none()
        if role:
            for key, value in data.items():
                if hasattr(role, key):
                    setattr(role, key, value)
            await db.flush()
            await db.refresh(role)
        return role

    async def delete(
        self,
        db: AsyncSession,
        role_id: int,
    ) -> None:
        """软删除角色"""
        values = {"deleted": 1}
        values.update(get_audit_update_values())
        stmt = update(SysRole).where(SysRole.id == role_id).values(**values)
        await db.execute(stmt)
        await db.flush()

    async def delete_by_ids(
        self,
        db: AsyncSession,
        role_ids: list[int],
    ) -> int:
        """批量软删除角色（1 条 SQL 替代 N 条，避免 N+1）"""
        if not role_ids:
            return 0
        values = {"deleted": 1}
        values.update(get_audit_update_values())
        stmt = update(SysRole).where(SysRole.id.in_(role_ids)).values(**values)
        result = await db.execute(stmt)
        return result.rowcount

    async def get_role_options(
        self,
        db: AsyncSession,
        *,
        is_root: bool = False,
    ) -> list[dict]:
        """获取角色下拉选项列表（仅启用状态，非 root 用户排除内置角色）"""
        stmt = (
            select(SysRole.id, SysRole.name)
            .where(SysRole.deleted == 0, SysRole.status == 1)
            .order_by(SysRole.sort)
        )
        if not is_root:
            stmt = stmt.where(SysRole.code.notin_(BUILTIN_ROLE_CODES))
        result = await db.execute(stmt)
        return [{"value": row[0], "label": row[1]} for row in result.fetchall()]

    async def get_role_menu_ids(
        self,
        db: AsyncSession,
        role_id: int,
    ) -> list[int]:
        """获取角色的菜单 ID 集合"""
        stmt = select(SysRoleMenu.menu_id).where(SysRoleMenu.role_id == role_id)
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def replace_role_menus(
        self,
        db: AsyncSession,
        role_id: int,
        menu_ids: list[int],
    ) -> None:
        """替换角色菜单（先删后增）"""
        await db.execute(
            delete(SysRoleMenu).where(SysRoleMenu.role_id == role_id)
        )
        if menu_ids:
            role_menus = [
                SysRoleMenu(role_id=role_id, menu_id=menu_id) for menu_id in menu_ids
            ]
            db.add_all(role_menus)
        await db.flush()

    async def delete_role_menus_by_role_ids(
        self,
        db: AsyncSession,
        role_ids: list[int],
    ) -> int:
        """批量删除多个角色的菜单关联记录（物理删除，避免 N+1）"""
        if not role_ids:
            return 0
        stmt = delete(SysRoleMenu).where(SysRoleMenu.role_id.in_(role_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def get_maximum_data_scope(
        self,
        db: AsyncSession,
        role_codes: list[str],
    ) -> int | None:
        """获取最大范围的数据权限（返回最小的 data_scope 值）"""
        if not role_codes:
            return None
        stmt = (
            select(SysRole.data_scope)
            .where(
                SysRole.code.in_(role_codes),
                SysRole.deleted == 0,
                SysRole.status == 1,
            )
            .order_by(SysRole.data_scope)
        )
        result = await db.execute(stmt)
        data_scopes = [row[0] for row in result.fetchall()]
        return min(data_scopes) if data_scopes else None

    async def check_name_exists(
        self,
        db: AsyncSession,
        name: str,
        *,
        exclude_id: int | None = None,
    ) -> bool:
        """检查角色名称是否已存在（含软删记录，命中→报"已被历史记录占用"）"""
        stmt = select(func.count()).select_from(SysRole).where(
            SysRole.name == name,
        )
        if exclude_id:
            stmt = stmt.where(SysRole.id != exclude_id)
        result = await db.execute(stmt)
        return (result.scalar() or 0) > 0

    async def check_code_exists(
        self,
        db: AsyncSession,
        code: str,
        *,
        exclude_id: int | None = None,
    ) -> bool:
        """检查角色编码是否已存在（含软删记录，命中→报"已被历史记录占用"）"""
        stmt = select(func.count()).select_from(SysRole).where(
            SysRole.code == code,
        )
        if exclude_id:
            stmt = stmt.where(SysRole.id != exclude_id)
        result = await db.execute(stmt)
        return (result.scalar() or 0) > 0

    async def get_all_active_codes(self, db: AsyncSession) -> list[str]:
        """获取所有未删除角色的编码列表（用于权限缓存失效时精确删除）"""
        stmt = select(SysRole.code).where(
            SysRole.deleted == 0,
            SysRole.code.isnot(None),
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall() if row[0]]

    async def get_by_code(self, db: AsyncSession, code: str) -> SysRole | None:
        """根据角色编码查询角色（含软删记录）"""
        stmt = select(SysRole).where(SysRole.code == code)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_user_role_by_role_ids(
        self,
        db: AsyncSession,
        role_ids: list[int],
    ) -> int:
        """批量物理删除角色的用户关联记录（软删角色时同步清理）"""
        from app.models.entity.sys_user import SysUserRole
        if not role_ids:
            return 0
        stmt = delete(SysUserRole).where(SysUserRole.role_id.in_(role_ids))
        result = await db.execute(stmt)
        return result.rowcount


# 单例
role_repository = RoleRepository()
