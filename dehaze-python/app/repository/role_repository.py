"""
角色数据访问层
"""

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_menu import SysRoleMenu
from app.models.entity.sys_user import SysRole
from app.repository.base import BaseRepository

# 超级管理员角色编码
ROOT_ROLE_CODE = "ROOT"


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
        stmt = update(SysRole).where(SysRole.id == role_id).values(deleted=1)
        await db.execute(stmt)
        await db.flush()

    async def get_role_options(
        self,
        db: AsyncSession,
        *,
        is_root: bool = False,
    ) -> list[dict]:
        """获取角色下拉选项列表（仅启用状态，非 root 用户排除 ROOT 角色）"""
        stmt = (
            select(SysRole.id, SysRole.name)
            .where(SysRole.deleted == 0, SysRole.status == 1)
            .order_by(SysRole.sort)
        )
        if not is_root:
            stmt = stmt.where(SysRole.code != ROOT_ROLE_CODE)
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

    async def delete_role_menus(
        self,
        db: AsyncSession,
        role_id: int,
    ) -> int:
        """删除角色的菜单关联记录（物理删除）"""
        stmt = delete(SysRoleMenu).where(SysRoleMenu.role_id == role_id)
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
        """检查角色名称是否已存在"""
        stmt = select(func.count()).select_from(SysRole).where(
            SysRole.deleted == 0,
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
        """检查角色编码是否已存在"""
        stmt = select(func.count()).select_from(SysRole).where(
            SysRole.deleted == 0,
            SysRole.code == code,
        )
        if exclude_id:
            stmt = stmt.where(SysRole.id != exclude_id)
        result = await db.execute(stmt)
        return (result.scalar() or 0) > 0

    async def check_name_or_code_exists(
        self,
        db: AsyncSession,
        name: str,
        code: str,
        *,
        exclude_id: int | None = None,
    ) -> bool:
        """
        检查角色名称或编码是否已存在

        注意：此方法用于创建时检查，更新时请分别调用 check_name_exists 和 check_code_exists
        """
        stmt = select(func.count()).select_from(SysRole).where(
            SysRole.deleted == 0,
            (SysRole.name == name) | (SysRole.code == code),
        )
        if exclude_id:
            stmt = stmt.where(SysRole.id != exclude_id)
        result = await db.execute(stmt)
        return (result.scalar() or 0) > 0


# 单例
role_repository = RoleRepository()
