from datetime import datetime
from typing import Optional

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_member import SysMember
from app.models.entity.sys_user import SysUser
from app.repository.base import BaseRepository, escape_like


class MemberRepository(BaseRepository[SysMember]):
    model = SysMember

    async def get_by_user_id(self, db: AsyncSession, user_id: int) -> Optional[SysMember]:
        stmt = select(SysMember).where(
            SysMember.user_id == user_id,
            SysMember.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_user(self, db: AsyncSession, user_id: int) -> Optional[dict]:
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
        keywords: Optional[str] = None,
        level_code: Optional[str] = None,
        status: Optional[int] = None,
        expire_time_start: Optional[str] = None,
        expire_time_end: Optional[str] = None,
        growth_min: Optional[int] = None,
        growth_max: Optional[int] = None,
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
            start_dt = datetime.strptime(expire_time_start, "%Y-%m-%d %H:%M:%S")
            stmt = stmt.where(SysMember.expire_time >= start_dt)
        if expire_time_end:
            end_dt = datetime.strptime(expire_time_end, "%Y-%m-%d %H:%M:%S")
            stmt = stmt.where(SysMember.expire_time <= end_dt)
        if growth_min is not None:
            stmt = stmt.where(SysMember.growth_value >= growth_min)
        if growth_max is not None:
            stmt = stmt.where(SysMember.growth_value <= growth_max)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysMember.create_time.desc(), SysMember.id.desc())
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
