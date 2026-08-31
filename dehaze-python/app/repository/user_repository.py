"""
用户数据访问层
"""

from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_user import SysRole, SysUser, SysUserRole
from app.repository.base import BaseRepository
from app.repository.dept_repository import dept_repository


class UserRepository(BaseRepository[SysUser]):
    """用户数据访问层"""

    model = SysUser

    async def list_active_admin_ids(self, db: AsyncSession) -> list[int]:
        """全部活跃管理员的用户 ID（ROOT/ADMIN 角色，去重；低分告警等定向通知用）"""
        stmt = (
            select(SysUser.id)
            .join(SysUserRole, SysUser.id == SysUserRole.user_id)
            .join(SysRole, SysUserRole.role_id == SysRole.id)
            .where(
                SysUser.deleted == 0,
                SysUser.status == 1,
                SysRole.code.in_(["ROOT", "ADMIN"]),
                SysRole.deleted == 0,
            )
            .distinct()
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_username(
        self,
        db: AsyncSession,
        username: str,
    ) -> SysUser | None:
        """根据用户名查询（仅活跃用户，默认行为）"""
        stmt = select(SysUser).where(
            SysUser.username == username,
            SysUser.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_username_include_deleted(
        self,
        db: AsyncSession,
        username: str,
    ) -> SysUser | None:
        """根据用户名查询（查全表，含软删行，用于注册/改名查重）"""
        stmt = select(SysUser).where(
            SysUser.username == username,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_display_names(
        self,
        db: AsyncSession,
        user_ids: set[int] | list[int],
    ) -> dict[int, str]:
        """批量查询用户展示名（昵称优先，缺省回退用户名；含已软删用户，审计视角需追溯历史归属）"""
        if not user_ids:
            return {}
        stmt = (
            select(SysUser.id, func.coalesce(SysUser.nickname, SysUser.username))
            .where(SysUser.id.in_(user_ids))
            .execution_options(include_deleted=True)
        )
        rows = (await db.execute(stmt)).all()
        return {row[0]: row[1] for row in rows if row[1]}

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
        await db.execute(delete(SysUserRole).where(SysUserRole.user_id == user_id))
        if role_ids:
            role_links = [SysUserRole(user_id=user_id, role_id=rid) for rid in role_ids]
            db.add_all(role_links)
        await db.flush()

    async def get_existing_usernames(
        self,
        db: AsyncSession,
        usernames: list[str],
    ) -> set[str]:
        """批量查询已存在的用户名（避免导入时 N+1，含软删行）"""
        if not usernames:
            return set()
        stmt = (
            select(SysUser.username)
            .where(
                SysUser.username.in_(usernames),
            )
            .execution_options(include_deleted=True)
        )
        result = await db.execute(stmt)
        return {row[0] for row in result.fetchall() if row[0]}

    async def check_username_exists(
        self,
        db: AsyncSession,
        username: str,
        *,
        exclude_id: int | None = None,
    ) -> bool:
        """检查用户名是否已存在（查全表，含软删行）"""
        stmt = select(SysUser).where(
            SysUser.username == username,
        )
        if exclude_id:
            stmt = stmt.where(SysUser.id != exclude_id)
        stmt = stmt.execution_options(include_deleted=True)
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
        current_user=None,
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
            current_user: 当前登录用户（用于行级数据权限过滤）

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

        # 关键词搜索（与 Java 后端一致：不转义特殊字符，直接 LIKE）
        if keywords:
            base_query = base_query.where(
                or_(
                    SysUser.username.like(f"%{keywords}%"),
                    SysUser.nickname.like(f"%{keywords}%"),
                    SysUser.mobile.like(f"%{keywords}%"),
                )
            )

        if status is not None:
            base_query = base_query.where(SysUser.status == status)

        if dept_ids:
            base_query = base_query.where(SysUser.dept_id.in_(dept_ids))

        if current_user is not None:
            from app.repository.data_scope import apply_data_scope

            children_ids = (
                await dept_repository.get_children_ids(db, current_user.dept_id)
                if current_user.data_scope == 1 and current_user.dept_id is not None
                else None
            )
            base_query = await apply_data_scope(
                base_query,
                current_user,
                db,
                dept_field=SysUser.dept_id,
                creator_field=SysUser.create_by,
                children_ids=children_ids,
            )

        if create_time_start:
            start_dt = datetime.strptime(create_time_start, "%Y-%m-%d")
            base_query = base_query.where(SysUser.create_time >= start_dt)

        if create_time_end:
            end_dt = datetime.strptime(create_time_end, "%Y-%m-%d") + timedelta(days=1)
            base_query = base_query.where(SysUser.create_time < end_dt)

        # 排序并分页（按 id 升序，与 Java 后端一致）
        base_query = base_query.order_by(SysUser.id.asc())
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

    async def count_users_by_roles(
        self,
        db: AsyncSession,
        role_ids: list[int],
    ) -> dict[int, int]:
        """批量统计多个角色关联的用户数量（避免 N+1）"""
        if not role_ids:
            return {}
        stmt = (
            select(SysUserRole.role_id, func.count().label("cnt"))
            .where(SysUserRole.role_id.in_(role_ids))
            .group_by(SysUserRole.role_id)
        )
        result = await db.execute(stmt)
        return {int(row.role_id): int(row.cnt) for row in result}

    async def count_users_by_depts(
        self,
        db: AsyncSession,
        dept_ids: list[int],
    ) -> dict[int, int]:
        """批量统计多个部门下的用户数量（避免 N+1）"""
        if not dept_ids:
            return {}
        stmt = (
            select(SysUser.dept_id, func.count().label("cnt"))
            .where(SysUser.dept_id.in_(dept_ids), SysUser.deleted == 0)
            .group_by(SysUser.dept_id)
        )
        result = await db.execute(stmt)
        return {int(row.dept_id): int(row.cnt) for row in result}

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

    async def get_credits_balance_and_version(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> tuple[Decimal, int] | None:
        """读取用户余额与乐观锁版本号（billing 余额 CAS 前置读取）"""
        stmt = select(SysUser.credits_balance, SysUser.credits_version).where(
            SysUser.id == user_id, SysUser.deleted == 0
        )
        row = (await db.execute(stmt)).first()
        if row is None:
            return None
        return Decimal(row.credits_balance), row.credits_version

    async def deduct_balance_cas(
        self,
        db: AsyncSession,
        user_id: int,
        amount: Decimal,
        current_version: int,
    ) -> bool:
        """CAS 乐观锁扣减积分余额，返回是否成功"""
        result = await db.execute(
            update(SysUser)
            .where(
                SysUser.id == user_id,
                SysUser.credits_version == current_version,
            )
            .values(
                credits_balance=SysUser.credits_balance - amount,
                credits_version=SysUser.credits_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    async def increase_balance_cas(
        self,
        db: AsyncSession,
        user_id: int,
        amount: Decimal,
        current_version: int,
    ) -> bool:
        """CAS 乐观锁增加积分余额，返回是否成功"""
        result = await db.execute(
            update(SysUser)
            .where(
                SysUser.id == user_id,
                SysUser.credits_version == current_version,
            )
            .values(
                credits_balance=SysUser.credits_balance + amount,
                credits_version=SysUser.credits_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

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
            user_roles = [SysUserRole(user_id=user.id, role_id=rid) for rid in role_ids]
            db.add_all(user_roles)
            await db.flush()

        return user

user_repository = UserRepository()
