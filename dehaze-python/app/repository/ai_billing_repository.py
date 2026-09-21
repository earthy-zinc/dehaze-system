from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_billing import SysAiBilling
from app.repository.base import BaseRepository

# chat 类计费类型：token 相关统计仅基于 chat 类记录（asr/tts 的 input_tokens 存储
# 秒数/字符数，与 token 计量口径不同）；chat_subagent（子 Agent 实报实销）与主图
# chat 记录同口径聚合，保证消耗统计反映真实扣减。
# KB embedding/rerank 的 input_tokens 虽为 token 口径，但消耗归属知识库动作而非
# 对话消息（检索/文档向量化按 bill_type 区分），不纳入 chat token 统计
CHAT_BILL_TYPES = ("chat", "chat_subagent")


def _bill_type_filter(bill_type: str | Sequence[str] | None):
    """构造 bill_type 过滤条件（单类型为等值，多类型为 IN）"""
    if not bill_type:
        return None
    col = SysAiBilling.bill_type
    return col == bill_type if isinstance(bill_type, str) else col.in_(bill_type)


class AiBillingRepository(BaseRepository[SysAiBilling]):
    model = SysAiBilling

    async def create_billing(
        self,
        db: AsyncSession,
        **kwargs: Any,
    ) -> SysAiBilling:
        billing = SysAiBilling(**kwargs)
        return await self.create(db, billing)

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        *,
        conversation_id: int | None = None,
        bill_type: str | None = None,
        model_id: str | None = None,
        date_start: datetime | None = None,
        date_end: datetime | None = None,
    ) -> tuple[list[SysAiBilling], int]:
        stmt = select(SysAiBilling).where(SysAiBilling.user_id == user_id)
        if conversation_id is not None:
            stmt = stmt.where(SysAiBilling.conversation_id == conversation_id)
        if bill_type:
            stmt = stmt.where(SysAiBilling.bill_type == bill_type)
        if model_id:
            stmt = stmt.where(SysAiBilling.model == model_id)
        if date_start:
            stmt = stmt.where(SysAiBilling.create_time >= date_start)
        if date_end:
            stmt = stmt.where(SysAiBilling.create_time <= date_end)

        stmt = stmt.order_by(SysAiBilling.create_time.desc(), SysAiBilling.id.desc())
        return await self.paginate(db, stmt, page, size)

    async def list_by_conversation(
        self,
        db: AsyncSession,
        conversation_id: int,
    ) -> list[SysAiBilling]:
        stmt = (
            select(SysAiBilling)
            .where(SysAiBilling.conversation_id == conversation_id)
            .order_by(SysAiBilling.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_message(
        self,
        db: AsyncSession,
        message_id: int,
    ) -> list[SysAiBilling]:
        stmt = (
            select(SysAiBilling)
            .where(SysAiBilling.message_id == message_id)
            .order_by(SysAiBilling.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_request_id(
        self,
        db: AsyncSession,
        request_id: str,
    ) -> list[SysAiBilling]:
        stmt = (
            select(SysAiBilling)
            .where(SysAiBilling.request_id == request_id)
            .order_by(SysAiBilling.id.asc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def sum_credits_by_user_and_period(
        self,
        db: AsyncSession,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """统计用户某时间段内的积分消耗汇总"""
        stmt = select(
            func.coalesce(func.sum(SysAiBilling.credits), 0),
            func.coalesce(func.sum(SysAiBilling.credits_saved), 0),
            func.count(SysAiBilling.id),
        ).where(
            SysAiBilling.user_id == user_id,
            SysAiBilling.create_time >= start,
            SysAiBilling.create_time <= end,
        )
        row = (await db.execute(stmt)).one()
        return {
            "total_credits": int(row[0]),
            "total_credits_saved": int(row[1]),
            "total_count": int(row[2]),
        }

    async def sum_credits_by_user_group_by_bill_type(
        self,
        db: AsyncSession,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                SysAiBilling.bill_type,
                func.sum(SysAiBilling.credits),
                func.sum(SysAiBilling.input_tokens),
                func.sum(SysAiBilling.output_tokens),
                func.sum(SysAiBilling.credits_saved),
                func.count(SysAiBilling.id),
            )
            .where(
                SysAiBilling.user_id == user_id,
                SysAiBilling.create_time >= start,
                SysAiBilling.create_time <= end,
            )
            .group_by(SysAiBilling.bill_type)
        )
        rows = (await db.execute(stmt)).all()
        return [
            {
                "bill_type": r[0],
                "credits": int(r[1] or 0),
                "input_tokens": int(r[2] or 0),
                "output_tokens": int(r[3] or 0),
                "credits_saved": int(r[4] or 0),
                "count": int(r[5]),
            }
            for r in rows
        ]

    async def sum_credits_by_user_group_by_model(
        self,
        db: AsyncSession,
        user_id: int,
        start: datetime,
        end: datetime,
        *,
        bill_type: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                SysAiBilling.model,
                func.sum(SysAiBilling.credits),
                func.sum(SysAiBilling.input_tokens),
                func.sum(SysAiBilling.output_tokens),
                func.sum(SysAiBilling.credits_saved),
                func.sum(func.if_(SysAiBilling.actual_model.isnot(None), 1, 0)),
                func.count(SysAiBilling.id),
            )
            .where(
                SysAiBilling.user_id == user_id,
                SysAiBilling.create_time >= start,
                SysAiBilling.create_time <= end,
            )
            .group_by(SysAiBilling.model)
        )
        cond = _bill_type_filter(bill_type)
        if cond is not None:
            stmt = stmt.where(cond)
        rows = (await db.execute(stmt)).all()
        return [
            {
                "model": r[0],
                "credits": int(r[1] or 0),
                "input_tokens": int(r[2] or 0),
                "output_tokens": int(r[3] or 0),
                "credits_saved": int(r[4] or 0),
                "degradation_count": int(r[5] or 0),
                "count": int(r[6]),
            }
            for r in rows
        ]

    async def sum_credits_by_user_group_by_period(
        self,
        db: AsyncSession,
        user_id: int,
        start: datetime,
        end: datetime,
        period: str,
        *,
        bill_type: str | Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按日/月聚合用户消耗趋势（summary 数据源）"""
        fmt = "%Y-%m-%d" if period == "day" else "%Y-%m"
        date_col = func.date_format(SysAiBilling.create_time, fmt)
        stmt = (
            select(
                date_col,
                func.sum(SysAiBilling.credits),
                func.sum(SysAiBilling.input_tokens),
                func.sum(SysAiBilling.output_tokens),
                func.sum(SysAiBilling.credits_saved),
                func.sum(SysAiBilling.cached_input_tokens),
            )
            .where(
                SysAiBilling.user_id == user_id,
                SysAiBilling.create_time >= start,
                SysAiBilling.create_time <= end,
            )
            .group_by(date_col)
            .order_by(date_col)
        )
        cond = _bill_type_filter(bill_type)
        if cond is not None:
            stmt = stmt.where(cond)
        rows = (await db.execute(stmt)).all()
        return [
            {
                "date": r[0],
                "credits": int(r[1] or 0),
                "input_tokens": int(r[2] or 0),
                "output_tokens": int(r[3] or 0),
                "credits_saved": int(r[4] or 0),
                "cached_input_tokens": int(r[5] or 0),
            }
            for r in rows
        ]

    async def stats_by_dimension(
        self,
        db: AsyncSession,
        group_by: str,
        *,
        user_id: int | None = None,
        model_id: str | None = None,
        bill_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """按维度聚合计费统计（管理员），group_by: user/model/billType/day

        token 相关列仅基于 chat 类记录（asr/tts 的 input_tokens 存储秒数/字符数，与 token
        计量口径不同，混入会使 token 统计失真）。
        """
        if group_by == "user":
            dim_col = SysAiBilling.user_id
        elif group_by == "model":
            dim_col = SysAiBilling.model
        elif group_by == "billType":
            dim_col = SysAiBilling.bill_type
        elif group_by == "day":
            dim_col = func.date_format(SysAiBilling.create_time, "%Y-%m-%d")
        else:
            raise ValueError(f"不支持的统计维度: {group_by}")

        _is_chat = SysAiBilling.bill_type.in_(CHAT_BILL_TYPES)
        stmt = (
            select(
                dim_col,
                func.sum(SysAiBilling.credits),
                func.sum(func.if_(_is_chat, SysAiBilling.input_tokens, 0)),
                func.sum(func.if_(_is_chat, SysAiBilling.output_tokens, 0)),
                func.sum(func.if_(_is_chat, SysAiBilling.cached_input_tokens, 0)),
                func.sum(SysAiBilling.credits_saved),
                func.sum(func.if_(SysAiBilling.actual_model.isnot(None), 1, 0)),
            )
            .group_by(dim_col)
            .order_by(dim_col)
        )
        if user_id is not None:
            stmt = stmt.where(SysAiBilling.user_id == user_id)
        if model_id:
            stmt = stmt.where(SysAiBilling.model == model_id)
        if bill_type:
            stmt = stmt.where(SysAiBilling.bill_type == bill_type)
        if start:
            stmt = stmt.where(SysAiBilling.create_time >= start)
        if end:
            stmt = stmt.where(SysAiBilling.create_time <= end)

        rows = (await db.execute(stmt)).all()
        return [
            {
                "dimension": str(r[0]),
                "total_credits": int(r[1] or 0),
                "total_input_tokens": int(r[2] or 0),
                "total_output_tokens": int(r[3] or 0),
                "chat_cached_tokens": int(r[4] or 0),
                "credits_saved": int(r[5] or 0),
                "degradation_count": int(r[6] or 0),
            }
            for r in rows
        ]

    async def distinct_user_ids(
        self,
        db: AsyncSession,
        start: datetime,
        end: datetime,
    ) -> list[int]:
        """查询某时间段内有计费记录的去重用户 ID 列表"""
        stmt = (
            select(SysAiBilling.user_id)
            .where(
                SysAiBilling.create_time >= start,
                SysAiBilling.create_time <= end,
            )
            .distinct()
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.all()]


ai_billing_repository = AiBillingRepository()
