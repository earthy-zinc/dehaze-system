from datetime import datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member import QUOTA_TASK_TYPES, SysMember
from app.models.entity.sys_user import SysUser
from app.repository.base import BaseRepository, escape_like


def parse_expire_time(s: str, *, is_end: bool) -> datetime:
    fmt = "%Y-%m-%d %H:%M:%S" if " " in s else "%Y-%m-%d"
    dt = datetime.strptime(s, fmt)
    if " " not in s:
        return (
            dt.replace(hour=23, minute=59, second=59)
            if is_end
            else dt.replace(hour=0, minute=0, second=0)
        )
    return dt


class MemberRepository(BaseRepository[SysMember]):
    model = SysMember

    async def get_by_user_id(self, db: AsyncSession, user_id: int) -> SysMember | None:
        stmt = select(SysMember).where(
            SysMember.user_id == user_id,
            SysMember.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_or_init_member(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> SysMember:
        """确保会员记录存在：已存在且未删除时直接返回（保留全部数据）；
        软删记录复活（重置 deleted=0、降级 level_0、清空成长值与月度配额，保留
        total_consumption）；不存在时初始化 level_0 记录"""
        # 含软删行一起查，避免全局软删过滤器遮蔽待复活记录
        existing = (
            await db.execute(
                select(SysMember)
                .where(SysMember.user_id == user_id)
                .execution_options(include_deleted=True)
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.deleted == 0:
                return existing
            existing.deleted = 0
            existing.level_code = "level_0"
            existing.level_source = "growth"
            existing.growth_value = 0
            existing.status = 1
            for task_type in QUOTA_TASK_TYPES:
                setattr(existing, f"monthly_{task_type}_quota", 0)
                setattr(existing, f"monthly_{task_type}_used", 0)
            await db.flush()
            return existing

        member = SysMember(
            user_id=user_id,
            level_code="level_0",
            level_source="growth",
            growth_value=0,
            total_consumption=0,
            status=1,
            **{f"monthly_{t}_quota": 0 for t in QUOTA_TASK_TYPES},
            **{f"monthly_{t}_used": 0 for t in QUOTA_TASK_TYPES},
        )
        db.add(member)
        await db.flush()
        return member

    async def increase_used_conditional(
        self, db: AsyncSession, user_id: int, quota_type: str, quota: int
    ) -> bool:
        """条件更新当月已用 +1（已用 < 生效配额才允许），防止并发超扣。

        Args:
            quota: 该任务的生效配额（已合并会员卡覆盖与等级权益）

        Returns:
            True 表示更新成功；False 表示配额已用尽（条件不满足）
        """
        from sqlalchemy import update

        used_field = f"monthly_{quota_type}_used"
        stmt = (
            update(SysMember)
            .where(
                SysMember.user_id == user_id,
                SysMember.deleted == 0,
                getattr(SysMember, used_field) < quota,
            )
            .values(**{used_field: getattr(SysMember, used_field) + 1})
        )
        result = await db.execute(stmt)
        return result.rowcount > 0

    async def extend_expire_days(self, db: AsyncSession, user_id: int, days: int) -> None:
        """会员卡到期时间顺延（解冻补回用）：expire_time += days"""
        from sqlalchemy import update

        stmt = (
            update(SysMember)
            .where(SysMember.user_id == user_id, SysMember.deleted == 0)
            .values(expire_time=SysMember.expire_time + timedelta(days=days))
        )
        await db.execute(stmt)

    async def list_active_by_level(
        self,
        db: AsyncSession,
        level_code: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ) -> list[SysMember]:
        """分页扫描指定等级的活跃会员（未删除、未冻结），用于 VIP 月度赠送等批量任务"""
        stmt = (
            select(SysMember)
            .where(
                SysMember.level_code == level_code,
                SysMember.deleted == 0,
                SysMember.status == 1,
            )
            .order_by(SysMember.id.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_with_user(self, db: AsyncSession, user_id: int) -> dict | None:
        stmt = (
            select(
                SysMember,
                SysUser.username.label("username"),
                SysUser.nickname.label("nickname"),
                SysUser.avatar.label("avatar"),
            )
            .outerjoin(SysUser, SysMember.user_id == SysUser.id)
            .where(SysMember.user_id == user_id, SysMember.deleted == 0)
        )
        result = await db.execute(stmt)
        row = result.first()
        if not row:
            return None
        member = row[0]
        return {
            "member": member,
            "username": row[1],
            "nickname": row[2],
            "avatar": row[3],
        }

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        keywords: str | None = None,
        level_code: str | None = None,
        status: int | None = None,
        expire_time_start: str | None = None,
        expire_time_end: str | None = None,
        growth_min: int | None = None,
        growth_max: int | None = None,
    ) -> tuple[list[dict], int]:
        stmt = (
            select(
                SysMember,
                SysUser.username.label("username"),
                SysUser.nickname.label("nickname"),
            )
            .outerjoin(SysUser, SysMember.user_id == SysUser.id)
            .where(SysMember.deleted == 0)
        )

        if keywords:
            escaped = escape_like(keywords)
            like_pattern = f"%{escaped}%"
            stmt = stmt.where(
                or_(
                    SysUser.username.like(like_pattern, escape="\\"),
                    SysUser.nickname.like(like_pattern, escape="\\"),
                    SysUser.mobile.like(like_pattern, escape="\\"),
                )
            )

        if level_code:
            stmt = stmt.where(SysMember.level_code == level_code)
        if status is not None:
            stmt = stmt.where(SysMember.status == status)
        if expire_time_start:
            start_dt = parse_expire_time(expire_time_start, is_end=False)
            stmt = stmt.where(SysMember.expire_time >= start_dt)
        if expire_time_end:
            end_dt = parse_expire_time(expire_time_end, is_end=True)
            stmt = stmt.where(SysMember.expire_time <= end_dt)
        if growth_min is not None:
            stmt = stmt.where(SysMember.growth_value >= growth_min)
        if growth_max is not None:
            stmt = stmt.where(SysMember.growth_value <= growth_max)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysMember.become_member_time.desc(), SysMember.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "member": row[0],
                "username": row[1],
                "nickname": row[2],
            }
            for row in rows
        ]
        return items, total


member_repository = MemberRepository()
