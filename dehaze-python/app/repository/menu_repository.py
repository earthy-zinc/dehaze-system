"""
菜单数据访问层
"""

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_menu import SysMenu, SysRoleMenu
from app.models.entity.sys_user import SysRole
from app.repository.base import BaseRepository, escape_like


class MenuRepository(BaseRepository[SysMenu]):
    """菜单数据访问层"""

    model = SysMenu

    async def get_list(
        self,
        db: AsyncSession,
        keyword: str | None = None,
        type: int | None = None,
        visible: int | None = None,
    ) -> list[SysMenu]:
        """获取菜单列表（按排序字段排序，支持关键字/类型/显示状态筛选）

        type/visible 用于 T-MM-009/010 列表筛选。
        """
        stmt = select(SysMenu).where(SysMenu.deleted == 0).order_by(SysMenu.sort)
        if keyword:
            escaped = escape_like(keyword)
            stmt = stmt.where(SysMenu.name.like(f"%{escaped}%", escape="\\"))
        if type is not None:
            stmt = stmt.where(SysMenu.type == type)
        if visible is not None:
            stmt = stmt.where(SysMenu.visible == visible)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_route_menus(self, db: AsyncSession) -> list[SysMenu]:
        """获取路由菜单列表（类型为目录或菜单，且可见）"""
        stmt = (
            select(SysMenu)
            .where(
                and_(
                    SysMenu.deleted == 0,
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

    async def count_by_ids(
        self,
        db: AsyncSession,
        menu_ids: list[int],
    ) -> int:
        """统计给定ID集合中存在的菜单数量"""
        if not menu_ids:
            return 0
        stmt = select(func.count()).select_from(SysMenu).where(SysMenu.id.in_(menu_ids))
        result = await db.execute(stmt)
        return int(result.scalar() or 0)

    async def get_menu_ids_with_children_batch(
        self,
        db: AsyncSession,
        menu_ids: list[int],
    ) -> list[int]:
        """获取所有传入菜单ID及其子孙菜单ID（合并去重）

        条件：id IN (menu_ids) OR tree_path LIKE '%,id,%'（对每个 id 做 OR）
        """
        if not menu_ids:
            return []
        conditions = [SysMenu.id.in_(menu_ids)]
        for menu_id in menu_ids:
            conditions.append(func.concat(",", SysMenu.tree_path, ",").like(f"%,{menu_id},%"))
        stmt = select(SysMenu.id).where(or_(*conditions))
        result = await db.execute(stmt)
        return list({int(row[0]) for row in result.fetchall()})

    async def delete_menus_by_ids(
        self,
        db: AsyncSession,
        menu_ids: list[int],
    ) -> int:
        """根据ID集合批量删除菜单"""
        if not menu_ids:
            return 0
        stmt = delete(SysMenu).where(SysMenu.id.in_(menu_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def delete_role_menus_by_menu_ids(
        self,
        db: AsyncSession,
        menu_ids: list[int],
    ) -> int:
        """根据菜单ID集合批量删除角色-菜单关联记录"""
        if not menu_ids:
            return 0
        stmt = delete(SysRoleMenu).where(SysRoleMenu.menu_id.in_(menu_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def save_role_menu(
        self,
        db: AsyncSession,
        role_id: int,
        menu_id: int,
    ) -> None:
        """新增角色-菜单关联"""
        db.add(SysRoleMenu(role_id=role_id, menu_id=menu_id))
        await db.flush()

    async def exists_by_name(
        self,
        db: AsyncSession,
        parent_id: int,
        name: str,
        exclude_id: int | None = None,
    ) -> bool:
        """同级菜单名称是否已存在（含软删记录）

        用于 T-MM-015/027 同级重名校验。exclude_id 用于修改时排除自身。
        """
        stmt = select(func.count()).select_from(SysMenu).where(
            SysMenu.parent_id == parent_id,
            SysMenu.name == name,
        )
        if exclude_id is not None:
            stmt = stmt.where(SysMenu.id != exclude_id)
        result = await db.execute(stmt)
        return int(result.scalar() or 0) > 0

    async def exists_by_perm(
        self,
        db: AsyncSession,
        perm: str,
        exclude_id: int | None = None,
    ) -> bool:
        """权限标识是否已存在（含软删记录，全局唯一）

        用于 T-MM-016 权限标识唯一性校验。exclude_id 用于修改时排除自身。
        """
        stmt = (
            select(func.count())
            .select_from(SysMenu)
            .where(SysMenu.perm == perm, SysMenu.perm.isnot(None), SysMenu.perm != "")
        )
        if exclude_id is not None:
            stmt = stmt.where(SysMenu.id != exclude_id)
        result = await db.execute(stmt)
        return int(result.scalar() or 0) > 0

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
                SysMenu.deleted == 0,
                SysMenu.perm.isnot(None),
                SysMenu.perm != "",
            )
        )
        result = await db.execute(stmt)
        perms = result.scalars().all()
        return {p for p in perms if p is not None and p != ""}


# 单例
menu_repository = MenuRepository()
