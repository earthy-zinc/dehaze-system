from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_user import SysUser
from app.repository.base import BaseRepository, escape_like
from app.repository.dept_repository import dept_repository

ORDER_STATUS_MAP = {
    "pending": 1,
    "paid": 2,
    "completed": 3,
    "cancelled": 4,
    "refunding": 5,
    "refunded": 6,
}

ORDER_STATUS_REVERSE_MAP = {v: k for k, v in ORDER_STATUS_MAP.items()}


class OrderRepository(BaseRepository[SysOrder]):
    model = SysOrder

    async def get_by_order_no(self, db: AsyncSession, order_no: str) -> SysOrder | None:
        stmt = select(SysOrder).where(
            SysOrder.order_no == order_no,
            SysOrder.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_user(self, db: AsyncSession, order_no: str) -> dict | None:
        stmt = (
            select(
                SysOrder,
                SysUser.username.label("username"),
            )
            .outerjoin(SysUser, SysOrder.user_id == SysUser.id)
            .where(SysOrder.order_no == order_no, SysOrder.deleted == 0)
        )
        result = await db.execute(stmt)
        row = result.first()
        if not row:
            return None
        return {"order": row[0], "username": row[1]}

    async def get_my_page(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        page_size: int,
        *,
        status: str | None = None,
    ) -> tuple[list[SysOrder], int]:
        stmt = select(SysOrder).where(
            SysOrder.user_id == user_id,
            SysOrder.deleted == 0,
        )
        if status and status in ORDER_STATUS_MAP:
            stmt = stmt.where(SysOrder.status == ORDER_STATUS_MAP[status])

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysOrder.create_time.desc(), SysOrder.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        order_no: str | None = None,
        keywords: str | None = None,
        status: str | None = None,
        package_type: str | None = None,
        pay_method: str | None = None,
        amount_min: int | None = None,
        amount_max: int | None = None,
        paid_time_start: str | None = None,
        paid_time_end: str | None = None,
        current_user=None,
    ) -> tuple[list[dict], int]:
        stmt = (
            select(
                SysOrder,
                SysUser.username.label("username"),
            )
            .outerjoin(SysUser, SysOrder.user_id == SysUser.id)
            .where(SysOrder.deleted == 0)
        )

        # 行级数据权限过滤（订单表无 dept_id，通过已 JOIN 的 sys_user.dept_id 过滤部门范围）
        if current_user is not None:
            from app.repository.data_scope import apply_data_scope

            children_ids = (
                await dept_repository.get_children_ids(db, current_user.dept_id)
                if current_user.data_scope == 1 and current_user.dept_id is not None
                else None
            )
            stmt = await apply_data_scope(
                stmt,
                current_user,
                db,
                dept_field=SysUser.dept_id,
                creator_field=SysOrder.user_id,
                children_ids=children_ids,
            )

        if order_no:
            stmt = stmt.where(SysOrder.order_no == order_no)
        if keywords:
            escaped = escape_like(keywords)
            like_pattern = f"%{escaped}%"
            stmt = stmt.where(
                or_(
                    SysUser.username.like(like_pattern, escape="\\"),
                    SysUser.nickname.like(like_pattern, escape="\\"),
                )
            )
        if status and status in ORDER_STATUS_MAP:
            stmt = stmt.where(SysOrder.status == ORDER_STATUS_MAP[status])
        if package_type:
            stmt = stmt.where(SysOrder.package_type == package_type)
        if pay_method:
            stmt = stmt.where(SysOrder.pay_method == pay_method)
        if amount_min is not None:
            stmt = stmt.where(SysOrder.payable_amount >= amount_min)
        if amount_max is not None:
            stmt = stmt.where(SysOrder.payable_amount <= amount_max)
        if paid_time_start:
            stmt = stmt.where(
                SysOrder.paid_time >= datetime.strptime(paid_time_start, "%Y-%m-%d %H:%M:%S")
            )
        if paid_time_end:
            stmt = stmt.where(
                SysOrder.paid_time <= datetime.strptime(paid_time_end, "%Y-%m-%d %H:%M:%S")
            )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysOrder.create_time.desc(), SysOrder.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "order": row[0],
                "username": row[1],
            }
            for row in rows
        ]
        return items, total

    async def has_paid_order(self, db: AsyncSession, user_id: int) -> bool:
        """用户是否存在已支付订单（status: 2 已支付 / 3 已完成），用于新用户专享可用性判断"""
        stmt = select(func.count()).select_from(SysOrder).where(
            SysOrder.user_id == user_id,
            SysOrder.deleted == 0,
            SysOrder.status.in_([2, 3]),
        )
        return ((await db.execute(stmt)).scalar() or 0) > 0

    async def list_expired_pending(self, db: AsyncSession) -> list[SysOrder]:
        stmt = select(SysOrder).where(
            SysOrder.status == 1,
            SysOrder.expire_time < datetime.now(),
            SysOrder.deleted == 0,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_completed_expiring(self, db: AsyncSession) -> list[SysOrder]:
        stmt = select(SysOrder).where(
            SysOrder.status == 2,
            SysOrder.package_expire_time < datetime.now(),
            SysOrder.deleted == 0,
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_stats(
        self,
        db: AsyncSession,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict:
        base = select(SysOrder).where(SysOrder.deleted == 0)
        if start_time:
            base = base.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            base = base.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )

        total_stmt = select(func.count()).select_from(base.subquery())
        total_orders = (await db.execute(total_stmt)).scalar() or 0

        revenue_stmt = select(func.coalesce(func.sum(SysOrder.paid_amount), 0)).where(
            SysOrder.deleted == 0,
            SysOrder.status.in_([2, 3]),
        )
        if start_time:
            revenue_stmt = revenue_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            revenue_stmt = revenue_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        total_revenue = int((await db.execute(revenue_stmt)).scalar() or 0)

        refund_stmt = select(func.coalesce(func.sum(SysOrder.paid_amount), 0)).where(
            SysOrder.deleted == 0,
            SysOrder.status == 6,
        )
        if start_time:
            refund_stmt = refund_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            refund_stmt = refund_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        total_refund = int((await db.execute(refund_stmt)).scalar() or 0)

        status_stmt = (
            select(SysOrder.status, func.count())
            .where(SysOrder.deleted == 0)
            .group_by(SysOrder.status)
        )
        if start_time:
            status_stmt = status_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            status_stmt = status_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        status_rows = (await db.execute(status_stmt)).all()
        status_distribution = {
            ORDER_STATUS_REVERSE_MAP.get(s, "unknown"): c for s, c in status_rows
        }

        pay_method_stmt = (
            select(SysOrder.pay_method, func.count())
            .where(
                SysOrder.deleted == 0,
                SysOrder.pay_method.isnot(None),
            )
            .group_by(SysOrder.pay_method)
        )
        if start_time:
            pay_method_stmt = pay_method_stmt.where(
                SysOrder.create_time >= datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            )
        if end_time:
            pay_method_stmt = pay_method_stmt.where(
                SysOrder.create_time <= datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            )
        pay_method_rows = (await db.execute(pay_method_stmt)).all()
        pay_method_distribution = {pm: c for pm, c in pay_method_rows}

        return {
            "total_orders": total_orders,
            "total_revenue": total_revenue,
            "total_refund": total_refund,
            "status_distribution": status_distribution,
            "pay_method_distribution": pay_method_distribution,
        }


order_repository = OrderRepository()
