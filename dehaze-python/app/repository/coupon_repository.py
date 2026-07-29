from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_coupon import SysCoupon
from app.models.entity.sys_user_coupon import SysUserCoupon
from app.repository.base import BaseRepository, escape_like


class CouponRepository(BaseRepository[SysCoupon]):
    model = SysCoupon

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
        status: Optional[int] = None,
    ) -> tuple[list[SysCoupon], int]:
        stmt = select(SysCoupon).where(SysCoupon.deleted == 0)

        if name:
            escaped = escape_like(name)
            stmt = stmt.where(SysCoupon.name.like(f"%{escaped}%", escape="\\"))
        if type:
            stmt = stmt.where(SysCoupon.type == type)
        if status is not None:
            stmt = stmt.where(SysCoupon.status == status)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysCoupon.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def increment_issued_qty(self, db: AsyncSession, coupon_id: int) -> bool:
        stmt = (
            update(SysCoupon)
            .where(
                SysCoupon.id == coupon_id,
                SysCoupon.deleted == 0,
                SysCoupon.status == 1,
            )
            .values(issued_qty=SysCoupon.issued_qty + 1)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def increment_issued_qty_with_limit(self, db: AsyncSession, coupon_id: int) -> bool:
        stmt = (
            update(SysCoupon)
            .where(
                SysCoupon.id == coupon_id,
                SysCoupon.deleted == 0,
                SysCoupon.status == 1,
            )
            .values(issued_qty=SysCoupon.issued_qty + 1)
            .execution_options(synchronize_session=False)
        )
        stmt = stmt.where(
            (SysCoupon.total_qty == -1) | (SysCoupon.issued_qty < SysCoupon.total_qty)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def increment_used_qty(self, db: AsyncSession, coupon_id: int) -> None:
        stmt = (
            update(SysCoupon)
            .where(SysCoupon.id == coupon_id)
            .values(used_qty=SysCoupon.used_qty + 1)
        )
        await db.execute(stmt)
        await db.flush()


coupon_repository = CouponRepository()


class UserCouponRepository(BaseRepository[SysUserCoupon]):
    model = SysUserCoupon

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        status: Optional[int] = None,
    ) -> list[SysUserCoupon]:
        stmt = select(SysUserCoupon).where(
            SysUserCoupon.user_id == user_id,
            SysUserCoupon.deleted == 0,
        )
        if status is not None:
            stmt = stmt.where(SysUserCoupon.status == status)
        stmt = stmt.order_by(SysUserCoupon.id.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_user_and_coupon(
        self,
        db: AsyncSession,
        user_id: int,
        coupon_id: int,
    ) -> int:
        stmt = select(func.count()).where(
            SysUserCoupon.user_id == user_id,
            SysUserCoupon.coupon_id == coupon_id,
            SysUserCoupon.deleted == 0,
        )
        return (await db.execute(stmt)).scalar() or 0

    async def lock_coupon(self, db: AsyncSession, user_coupon_id: int) -> bool:
        stmt = (
            update(SysUserCoupon)
            .where(
                SysUserCoupon.id == user_coupon_id,
                SysUserCoupon.status == 1,
                SysUserCoupon.deleted == 0,
            )
            .values(status=4)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def consume_coupon(self, db: AsyncSession, user_coupon_id: int, order_id: int) -> bool:
        stmt = (
            update(SysUserCoupon)
            .where(
                SysUserCoupon.id == user_coupon_id,
                SysUserCoupon.status == 4,
            )
            .values(status=2, used_time=datetime.now(), used_order_id=order_id)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def release_coupon(self, db: AsyncSession, user_coupon_id: int) -> bool:
        stmt = (
            update(SysUserCoupon)
            .where(
                SysUserCoupon.id == user_coupon_id,
                SysUserCoupon.status == 4,
            )
            .values(status=1)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount > 0

    async def expire_coupons(self, db: AsyncSession) -> int:
        stmt = (
            update(SysUserCoupon)
            .where(
                SysUserCoupon.status == 1,
                SysUserCoupon.expire_time < datetime.now(),
                SysUserCoupon.deleted == 0,
            )
            .values(status=3)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount

    async def count_used_by_coupon_ids(self, db: AsyncSession, coupon_ids: list[int]) -> int:
        if not coupon_ids:
            return 0
        stmt = select(func.count()).where(
            SysUserCoupon.coupon_id.in_(coupon_ids),
            SysUserCoupon.status == 2,
            SysUserCoupon.deleted == 0,
        )
        return (await db.execute(stmt)).scalar() or 0

    async def soft_delete_unused_by_coupon_ids(self, db: AsyncSession, coupon_ids: list[int]) -> None:
        if not coupon_ids:
            return
        stmt = (
            update(SysUserCoupon)
            .where(
                SysUserCoupon.coupon_id.in_(coupon_ids),
                SysUserCoupon.status == 1,
                SysUserCoupon.deleted == 0,
            )
            .values(deleted=1)
        )
        await db.execute(stmt)
        await db.flush()


user_coupon_repository = UserCouponRepository()
