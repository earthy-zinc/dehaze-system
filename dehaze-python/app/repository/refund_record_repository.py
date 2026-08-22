from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_order import SysOrder
from app.models.entity.sys_refund_record import SysRefundRecord
from app.models.entity.sys_user import SysUser
from app.repository.base import BaseRepository, escape_like

REFUND_STATUS_MAP = {
    "refunding": 1,
    "refunded": 2,
    "refund_failed": 3,
}

REFUND_STATUS_REVERSE_MAP = {v: k for k, v in REFUND_STATUS_MAP.items()}


class RefundRecordRepository(BaseRepository[SysRefundRecord]):
    model = SysRefundRecord

    async def get_by_order_id(self, db: AsyncSession, order_id: int) -> SysRefundRecord | None:
        stmt = select(SysRefundRecord).where(
            SysRefundRecord.order_id == order_id,
            SysRefundRecord.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_page(
        self,
        db: AsyncSession,
        page: int,
        page_size: int,
        *,
        order_no: str | None = None,
        keywords: str | None = None,
        status: str | None = None,
        apply_time_start: str | None = None,
        apply_time_end: str | None = None,
    ) -> tuple[list[dict], int]:
        stmt = (
            select(
                SysRefundRecord,
                SysOrder.order_no.label("order_no"),
                SysUser.username.label("username"),
            )
            .join(SysOrder, SysRefundRecord.order_id == SysOrder.id)
            .outerjoin(SysUser, SysRefundRecord.user_id == SysUser.id)
            .where(SysRefundRecord.deleted == 0)
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
        if status and status in REFUND_STATUS_MAP:
            stmt = stmt.where(SysRefundRecord.status == REFUND_STATUS_MAP[status])
        if apply_time_start:
            stmt = stmt.where(
                SysRefundRecord.apply_time
                >= datetime.strptime(apply_time_start, "%Y-%m-%d %H:%M:%S")
            )
        if apply_time_end:
            stmt = stmt.where(
                SysRefundRecord.apply_time <= datetime.strptime(apply_time_end, "%Y-%m-%d %H:%M:%S")
            )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        stmt = stmt.order_by(SysRefundRecord.apply_time.desc(), SysRefundRecord.id.desc())
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(stmt)
        rows = result.all()

        items = [
            {
                "refund": row[0],
                "order_no": row[1],
                "username": row[2],
            }
            for row in rows
        ]
        return items, total


refund_record_repository = RefundRecordRepository()
