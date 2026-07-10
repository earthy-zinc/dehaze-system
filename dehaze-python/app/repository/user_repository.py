"""
用户数据访问层
"""

from datetime import datetime, timedelta

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_user import SysRole, SysUser, SysUserRole
from app.repository.base import BaseRepository
from app.repository.base import escape_like


class UserRepository(BaseRepository[SysUser]):
    """用户数据访问层"""

    model = SysUser

    def _get_user_deleted_column(self):
        """获取 user deleted 列"""
        return getattr(SysUser, "deleted")

    def _get_role_deleted_column(self):
        """获取 role deleted 列"""
        return getattr(SysRole, "deleted")

    def _get_menu_deleted_column(self):
        """获取 menu deleted 列"""
        from app.models.entity.sys_menu import SysMenu
        return getattr(SysMenu, "deleted")

    async def get_by_username(
        self,
        db: AsyncSession,
        username: str,
    ) -> SysUser | None:
        """根据用户名查询"""
        deleted_column = self._get_user_deleted_column()
        stmt = select(SysUser).where(
            SysUser.username == username,
            deleted_column == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_roles(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[SysRole]:
        """查询用户的角色列表"""
        role_deleted_column = self._get_role_deleted_column()
        stmt = (
            select(SysRole)
            .join(SysUserRole, SysRole.id == SysUserRole.role_id)
            .where(
                SysUserRole.user_id == user_id,
                role_deleted_column == 0,
                SysRole.status == 1,
            )
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_user_role_ids(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[int]:
        """查询用户的角色 ID 列表"""
        stmt = select(SysUserRole.role_id).where(SysUserRole.user_id == user_id)
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def replace_user_roles(
        self,
        db: AsyncSession,
        user_id: int,
        role_ids: list[int],
    ) -> None:
        """替换用户角色（先删后增）"""
        await db.execute(
            delete(SysUserRole).where(SysUserRole.user_id == user_id)
        )
        if role_ids:
            role_links = [
                SysUserRole(user_id=user_id, role_id=rid) for rid in role_ids
            ]
            db.add_all(role_links)
        await db.flush()

    async def check_username_exists(
        self,
        db: AsyncSession,
        username: str,
        *,
        exclude_id: int | None = None,
    ) -> bool:
        """检查用户名是否已存在"""
        deleted_column = self._get_user_deleted_column()
        stmt = select(SysUser).where(
            SysUser.username == username,
            deleted_column == 0,
        )
        if exclude_id:
            stmt = stmt.where(SysUser.id != exclude_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def get_user_list(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        keywords: str | None = None,
        status: int | None = None,
        dept_ids: list[int] | None = None,
        create_time_start: str | None = None,
        create_time_end: str | None = None,
    ) -> tuple[list[dict], int]:
        """
        分页查询用户列表（含部门名称、角色名称）

        Args:
            db: 数据库会话
            page: 页码
            page_size: 每页数量
            keywords: 关键词（用户名/昵称/手机号）
            status: 状态
            dept_ids: 部门 ID 列表（包含子部门）
            create_time_start: 创建时间开始
            create_time_end: 创建时间结束

        Returns:
            (用户列表字典, 总数)
        """
        from app.models.entity.sys_dept import SysDept

        base_query = (
            select(
                SysUser.id,
                SysUser.username,
                SysUser.nickname,
                SysUser.mobile,
                SysUser.avatar,
                SysUser.status,
                SysUser.email,
                SysUser.gender,
                SysUser.create_time,
                SysDept.name.label("deptName"),
                func.group_concat(SysRole.name).label("roleNames"),
            )
            .outerjoin(SysDept, SysUser.dept_id == SysDept.id)
            .outerjoin(SysUserRole, SysUser.id == SysUserRole.user_id)
            .outerjoin(SysRole, (SysUserRole.role_id == SysRole.id) & (SysRole.deleted == 0))
            .where(SysUser.deleted == 0, SysUser.username != "root")
            .group_by(SysUser.id)
        )

        # 关键词搜索
        if keywords:
            escaped = escape_like(keywords)
            base_query = base_query.where(
                or_(
                    SysUser.username.like(f"%{escaped}%", escape="\\"),
                    SysUser.nickname.like(f"%{escaped}%", escape="\\"),
                    SysUser.mobile.like(f"%{escaped}%", escape="\\"),
                )
            )

        # 状态筛选
        if status is not None:
            base_query = base_query.where(SysUser.status == status)

        # 部门筛选
        if dept_ids:
            base_query = base_query.where(SysUser.dept_id.in_(dept_ids))

        # 创建时间范围
        if create_time_start:
            try:
                start_dt = datetime.strptime(create_time_start, "%Y-%m-%d")
                base_query = base_query.where(SysUser.create_time >= start_dt)
            except ValueError:
                pass

        if create_time_end:
            try:
                end_dt = datetime.strptime(create_time_end, "%Y-%m-%d") + timedelta(days=1)
                base_query = base_query.where(SysUser.create_time < end_dt)
            except ValueError:
                pass

        # 排序并分页
        base_query = base_query.order_by(SysUser.create_time.desc())
        return await BaseRepository.paginate_rows(db, base_query, page, page_size)

    async def get_protected_user_ids(
        self,
        db: AsyncSession,
        user_ids: list[int],
    ) -> list[int]:
        """获取受保护的用户 ID（超级管理员）"""
        stmt = select(SysUser.id).where(
            SysUser.id.in_(user_ids),
            SysUser.username == "root",
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]

    async def count_users_by_role(
        self,
        db: AsyncSession,
        role_id: int,
    ) -> int:
        """统计关联某角色的用户数量"""
        stmt = select(func.count()).select_from(SysUserRole).where(
            SysUserRole.role_id == role_id
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def count_users_by_dept(
        self,
        db: AsyncSession,
        dept_id: int,
    ) -> int:
        """统计某部门下的用户数量"""
        stmt = select(func.count()).select_from(SysUser).where(
            SysUser.dept_id == dept_id,
            SysUser.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar() or 0

    async def get_user_role_codes(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[str]:
        """获取用户角色代码列表"""

        stmt = (
            select(SysRole.code)
            .join(SysUserRole, SysUserRole.role_id == SysRole.id)
            .where(
                SysUserRole.user_id == user_id,
                SysRole.deleted == 0,
                SysRole.status == 1,
            )
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall() if row[0]]

    async def create_user(
        self,
        db: AsyncSession,
        user: SysUser,
        role_ids: list[int] | None = None,
    ) -> SysUser:
        """创建用户并关联角色"""
        db.add(user)
        await db.flush()
        await db.refresh(user)

        if role_ids:
            user_roles = [
                SysUserRole(user_id=user.id, role_id=rid) for rid in role_ids
            ]
            db.add_all(user_roles)
            await db.flush()

        return user

    async def update_user(
        self,
        db: AsyncSession,
        user: SysUser,
    ) -> SysUser:
        """更新用户"""
        merged = await db.merge(user)
        await db.flush()
        return merged

    async def get_user_permissions(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> list[str]:
        """获取用户权限列表（通过角色关联的菜单权限）"""
        from app.models.entity.sys_menu import SysMenu, SysRoleMenu

        stmt = (
            select(SysMenu.perm)
            .distinct()
            .join(SysRoleMenu, SysRoleMenu.menu_id == SysMenu.id)
            .join(SysRole, SysRole.id == SysRoleMenu.role_id)
            .join(SysUserRole, SysUserRole.role_id == SysRole.id)
            .where(
                SysUserRole.user_id == user_id,
                SysMenu.perm.isnot(None),
                SysMenu.perm != "",
                SysMenu.status == 1,
                SysMenu.visible == 1,
                SysRole.deleted == 0,
                SysRole.status == 1,
            )
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall() if row[0]]


# 单例
user_repository = UserRepository()
