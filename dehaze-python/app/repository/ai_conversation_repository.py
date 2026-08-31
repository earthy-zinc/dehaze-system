from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_billing import SysAiBilling
from app.models.entity.sys_ai_billing_anomaly import SysAiBillingAnomaly
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.repository.base import BaseRepository


class AiConversationRepository(BaseRepository[SysAiConversation]):
    model = SysAiConversation

    async def paginate_user_conversations(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        status: int | None = None,
    ) -> tuple[list[SysAiConversation], int]:
        stmt = select(SysAiConversation).where(
            SysAiConversation.user_id == user_id,
            SysAiConversation.deleted == 0,
        )
        if status is not None:
            stmt = stmt.where(SysAiConversation.status == status)
        stmt = stmt.order_by(
            SysAiConversation.pinned.desc(),
            SysAiConversation.pinned_at.desc(),
            SysAiConversation.last_message_at.desc(),
            SysAiConversation.id.desc(),
        )
        return await self.paginate(db, stmt, page, size)

    async def paginate_all_conversations(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        status: int | None = None,
    ) -> tuple[list[SysAiConversation], int]:
        """管理端审计视角：全量用户会话分页（不过滤 user_id）"""
        stmt = select(SysAiConversation).where(SysAiConversation.deleted == 0)
        if status is not None:
            stmt = stmt.where(SysAiConversation.status == status)
        stmt = stmt.order_by(
            SysAiConversation.pinned.desc(),
            SysAiConversation.pinned_at.desc(),
            SysAiConversation.last_message_at.desc(),
            SysAiConversation.id.desc(),
        )
        return await self.paginate(db, stmt, page, size)

    async def paginate_all_with_keyword(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        keyword: str,
        status: int | None = None,
    ) -> tuple[list[SysAiConversation], int]:
        """管理端审计视角：全量会话按标题关键词过滤（DB like，审计范围不引 ES 全量检索）"""
        stmt = select(SysAiConversation).where(
            SysAiConversation.deleted == 0,
            SysAiConversation.title.like(f"%{keyword}%"),
        )
        if status is not None:
            stmt = stmt.where(SysAiConversation.status == status)
        stmt = stmt.order_by(
            SysAiConversation.pinned.desc(),
            SysAiConversation.pinned_at.desc(),
            SysAiConversation.last_message_at.desc(),
            SysAiConversation.id.desc(),
        )
        return await self.paginate(db, stmt, page, size)

    async def paginate_trash(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        before_date: datetime,
    ) -> tuple[list[SysAiConversation], int]:
        """回收站列表：已软删且未超过 30 天恢复窗口，按 delete_time 倒序"""
        stmt = select(SysAiConversation).where(
            SysAiConversation.user_id == user_id,
            SysAiConversation.deleted == 1,
            SysAiConversation.delete_time >= before_date,
        )
        stmt = stmt.order_by(
            SysAiConversation.delete_time.desc(),
            SysAiConversation.id.desc(),
        )
        # 回收站需查已软删记录，用 include_deleted 绕过全局 deleted=0 过滤
        stmt = stmt.execution_options(include_deleted=True)
        return await self.paginate(db, stmt, page, size)

    async def get_in_trash(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        before_date: datetime,
    ) -> SysAiConversation | None:
        """查询回收站中未超 30 天窗口的会话（供单条恢复）"""
        stmt = select(SysAiConversation).where(
            SysAiConversation.id == conv_id,
            SysAiConversation.user_id == user_id,
            SysAiConversation.deleted == 1,
            SysAiConversation.delete_time >= before_date,
        )
        # 恢复路径需查已软删记录，用 include_deleted 绕过全局 deleted=0 过滤
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_active_pinned(self, db: AsyncSession, user_id: int) -> int:
        """统计当前用户置顶且未删除的会话数"""
        stmt = (
            select(func.count())
            .select_from(SysAiConversation)
            .where(
                SysAiConversation.user_id == user_id,
                SysAiConversation.deleted == 0,
                SysAiConversation.pinned == 1,
            )
        )
        return (await db.execute(stmt)).scalar() or 0

    async def set_pinned(
        self,
        db: AsyncSession,
        conv_id: int,
        pinned: int,
        pinned_at: datetime | None,
    ) -> None:
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id == conv_id)
            .values(pinned=pinned, pinned_at=pinned_at)
        )
        await db.execute(stmt)

    async def mark_read(self, db: AsyncSession, conv_id: int, message_id: int) -> None:
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id == conv_id)
            .values(last_read_message_id=message_id)
        )
        await db.execute(stmt)

    async def soft_delete_by_ids(
        self,
        db: AsyncSession,
        ids: list[int],
    ) -> int:
        """软删除并记录软删时间（delete_time）"""
        if not ids:
            return 0
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id.in_(ids))
            .values(deleted=1, delete_time=datetime.now())
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def restore_by_ids(self, db: AsyncSession, ids: list[int]) -> int:
        """恢复软删除会话，清除软删时间标记"""
        if not ids:
            return 0
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id.in_(ids))
            .values(deleted=0, delete_time=None)
        )
        result = await db.execute(stmt)
        return result.rowcount

    async def list_soft_deleted_before(
        self,
        db: AsyncSession,
        before_date: datetime,
    ) -> list[int]:
        """查询软删超过 30 天的会话 ID（供物理清理）"""
        stmt = select(SysAiConversation.id).where(
            SysAiConversation.deleted == 1,
            SysAiConversation.delete_time < before_date,
        )
        # 物理清理需查已软删记录，用 include_deleted 绕过全局 deleted=0 过滤
        stmt = stmt.execution_options(include_deleted=True)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_ids(
        self,
        db: AsyncSession,
        user_id: int,
        ids: list[int],
    ) -> list[SysAiConversation]:
        """按 ID 列表查询会话（保持传入顺序），用于 ES 检索结果回查"""
        if not ids:
            return []
        stmt = select(SysAiConversation).where(
            SysAiConversation.id.in_(ids),
            SysAiConversation.user_id == user_id,
            SysAiConversation.deleted == 0,
        )
        result = await db.execute(stmt)
        convs = list(result.scalars().all())
        order = {cid: i for i, cid in enumerate(ids)}
        convs.sort(key=lambda c: order.get(c.id, len(ids)))
        return convs

    async def get_by_id_and_user(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> SysAiConversation | None:
        stmt = select(SysAiConversation).where(
            SysAiConversation.id == conv_id,
            SysAiConversation.user_id == user_id,
            SysAiConversation.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        db: AsyncSession,
        ids: list[int],
        status: int,
    ) -> None:
        """批量更新会话状态（归档/恢复活跃）"""
        if not ids:
            return
        stmt = update(SysAiConversation).where(SysAiConversation.id.in_(ids)).values(status=status)
        await db.execute(stmt)

    async def update_last_message(
        self,
        db: AsyncSession,
        conv_id: int,
        message_id: int,
        time: datetime,
        message_delta: int = 1,
    ) -> None:
        """更新会话最后消息指针并递增 message_count（创建消息后计数收敛于此单点）"""
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id == conv_id)
            .values(
                last_message_at=time,
                current_branch_message_id=message_id,
                message_count=SysAiConversation.message_count + message_delta,
            )
        )
        await db.execute(stmt)

    async def update_current_branch(
        self,
        db: AsyncSession,
        conv_id: int,
        message_id: int,
    ) -> None:
        """切换当前激活分支（仅更新分支指针，不改动最后消息时间）"""
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id == conv_id)
            .values(current_branch_message_id=message_id)
        )
        await db.execute(stmt)

    async def update_title(
        self,
        db: AsyncSession,
        conv_id: int,
        title: str,
        title_source: str = "auto",
    ) -> None:
        """更新会话标题及其来源"""
        stmt = (
            update(SysAiConversation)
            .where(SysAiConversation.id == conv_id)
            .values(title=title, title_source=title_source)
        )
        await db.execute(stmt)

    async def sum_consumption_by_conversation(
        self,
        db: AsyncSession,
        conv_ids: list[int],
    ) -> dict[int, dict[str, int]]:
        """按会话聚合计费消耗：{conv_id: {"token": 输入+输出Token, "credits": 积分}}

        只读消费 sys_ai_billing（跨模块只读查询，不改动计费模块）：与计费明细同源，
        覆盖 tool_llm/kb_inject 等无消息对应的计费项。
        """
        if not conv_ids:
            return {}
        stmt = (
            select(
                SysAiBilling.conversation_id,
                func.coalesce(func.sum(SysAiBilling.input_tokens + SysAiBilling.output_tokens), 0),
                func.coalesce(func.sum(SysAiBilling.credits), 0),
            )
            .where(SysAiBilling.conversation_id.in_(conv_ids))
            .group_by(SysAiBilling.conversation_id)
        )
        rows = (await db.execute(stmt)).all()
        return {
            row[0]: {"token": int(row[1]), "credits": int(row[2])} for row in rows if row[0]
        }

    async def list_quota_anomaly_conversation_ids(
        self,
        db: AsyncSession,
        conv_ids: list[int],
    ) -> set[int]:
        """存在"连续配额不足"异常的会话ID（审计视角异常标注的配额数据源）"""
        if not conv_ids:
            return set()
        stmt = (
            select(SysAiBilling.conversation_id)
            .join(SysAiBillingAnomaly, SysAiBillingAnomaly.billing_id == SysAiBilling.id)
            .where(
                SysAiBilling.conversation_id.in_(conv_ids),
                SysAiBillingAnomaly.anomaly_type == "consecutive_quota_fail",
            )
            .distinct()
        )
        rows = (await db.execute(stmt)).all()
        return {row[0] for row in rows if row[0]}

    async def archive_inactive(
        self,
        db: AsyncSession,
        before_date: datetime,
    ) -> list[tuple[int, int]]:
        """归档超期不活跃会话，返回 [(conv_id, user_id), ...] 供缓存清理"""
        stmt = select(SysAiConversation.id, SysAiConversation.user_id).where(
            SysAiConversation.status == 1,
            SysAiConversation.last_message_at < before_date,
        )
        result = await db.execute(stmt)
        rows = [(row[0], row[1]) for row in result.all()]
        if rows:
            await db.execute(
                update(SysAiConversation)
                .where(SysAiConversation.id.in_([r[0] for r in rows]))
                .values(status=2)
            )
        return rows


ai_conversation_repository = AiConversationRepository()
