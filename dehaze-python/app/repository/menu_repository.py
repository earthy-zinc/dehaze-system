"""
菜单数据访问层
"""

from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.models.entity.sys_user import SysRole
from app.repository.base import BaseRepository, escape_like
from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession


class MenuRepository(BaseRepository[SysMenu]):
    """菜单数据访问层"""

    model = SysMenu

    async def get_list(
        self,
        db: AsyncSession,
        keyword: str | None = None,
    ) -> list[SysMenu]:
        """获取菜单列表（按排序字段排序）"""
        stmt = select(SysMenu).order_by(SysMenu.sort)
        if keyword:
            escaped = escape_like(keyword)
            stmt = stmt.where(SysMenu.name.like(f"%{escaped}%", escape="\\"))
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_route_menus(self, db: AsyncSession) -> list[SysMenu]:
        """获取路由菜单列表（类型为目录或菜单，且可见）"""
        stmt = (
            select(SysMenu)
            .where(
                and_(
                    SysMenu.type.in_([1, 2]),  # 目录或菜单类型
                    SysMenu.visible == 1,
                )
            )
            .order_by(SysMenu.sort)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def create_menu(
        self,
        db: AsyncSession,
        menu: SysMenu,
    ) -> SysMenu:
        """创建菜单"""
        db.add(menu)
        await db.flush()
        await db.refresh(menu)
        return menu

    async def update_menu(
        self,
        db: AsyncSession,
        menu: SysMenu,
    ) -> SysMenu:
        """更新菜单"""
        merged = await db.merge(menu)
        await db.flush()
        return merged

    async def delete_menu_and_children(
        self,
        db: AsyncSession,
        menu_id: int,
    ) -> int:
        """删除菜单及其子菜单（使用 tree_path 匹配所有子节点）"""
        stmt = delete(SysMenu).where(
            or_(
                SysMenu.id == menu_id,
                func.concat(",", SysMenu.tree_path, ",").like(
                    f"%,{menu_id},%"),
            )
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_role_menus_by_menu_id(
        self,
        db: AsyncSession,
        menu_id: int,
    ) -> int:
        """删除菜单的角色关联记录"""
        # 先获取所有要删除的菜单ID（包含子菜单）
        menu_ids = await self._get_menu_ids_with_children(db, menu_id)

        stmt = delete(SysRoleMenu).where(SysRoleMenu.menu_id.in_(menu_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def _get_menu_ids_with_children(
        self,
        db: AsyncSession,
        menu_id: int,
    ) -> list[int]:
        """获取菜单ID及其所有子菜单ID"""
        stmt = select(SysMenu.id).where(
            or_(
                SysMenu.id == menu_id,
                func.concat(",", SysMenu.tree_path, ",").like(
                    f"%,{menu_id},%"),
            )
        )
        result = await db.execute(stmt)
        return [int(row[0]) for row in result.fetchall()]

    async def get_role_perms(
        self,
        db: AsyncSession,
        role_codes: list[str],
    ) -> set[str]:
        """获取角色权限集合（通过角色编码）"""
        if not role_codes:
            return set()
        stmt = (
            select(SysMenu.perm)
            .select_from(SysRoleMenu)
            .join(SysMenu, SysRoleMenu.menu_id == SysMenu.id)
            .join(SysRole, SysRoleMenu.role_id == SysRole.id)
            .where(
                SysRole.code.in_(role_codes),
                SysRole.deleted == 0,
                SysMenu.perm.isnot(None),
                SysMenu.perm != "",
            )
        )
        result = await db.execute(stmt)
        perms = result.scalars().all()
        return {p for p in perms if p is not None and p != ""}



# 单例
menu_repository = MenuRepository()
