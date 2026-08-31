from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.repository.base import BaseRepository, escape_like


class AiMessageRepository(BaseRepository[SysAiMessage]):
    model = SysAiMessage

    async def list_by_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        page: int,
        size: int,
        *,
        order: str = "asc",
    ) -> tuple[list[SysAiMessage], int]:
        """会话消息分页。

        order="desc" 时按时间倒序（pageNum=1 返回最新一页，供历史会话加载
        最近消息，前端展示时反转回正序）；其余调用方默认 asc 保持时间正序。
        """
        stmt = select(SysAiMessage).where(
            SysAiMessage.conversation_id == conv_id,
            SysAiMessage.deleted == 0,
        )
        if order == "desc":
            stmt = stmt.order_by(SysAiMessage.create_time.desc(), SysAiMessage.id.desc())
        else:
            stmt = stmt.order_by(SysAiMessage.create_time.asc(), SysAiMessage.id.asc())
        return await self.paginate(db, stmt, page, size)

    async def get_by_id(
        self,
        db: AsyncSession,
        msg_id: int,
    ) -> SysAiMessage | None:
        """按主键查消息（不限归属用户，供管理端审计 view=admin 使用）"""
        stmt = select(SysAiMessage).where(
            SysAiMessage.id == msg_id,
            SysAiMessage.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_and_user(
        self,
        db: AsyncSession,
        msg_id: int,
        user_id: int,
    ) -> SysAiMessage | None:
        stmt = (
            select(SysAiMessage)
            .join(
                SysAiConversation,
                SysAiMessage.conversation_id == SysAiConversation.id,
            )
            .where(
                SysAiMessage.id == msg_id,
                SysAiMessage.deleted == 0,
                SysAiConversation.user_id == user_id,
                SysAiConversation.deleted == 0,
            )
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_by_conversations(
        self,
        db: AsyncSession,
        conv_ids: list[int],
    ) -> int:
        """按会话 ID 列表物理删除消息（会话物理清理时级联调用）"""
        if not conv_ids:
            return 0
        stmt = delete(SysAiMessage).where(SysAiMessage.conversation_id.in_(conv_ids))
        result = await db.execute(stmt)
        return result.rowcount

    async def count_messages_after(
        self,
        db: AsyncSession,
        conv_id: int,
        after_id: int,
    ) -> int:
        """统计会话中 ID 大于指定已读 ID 的未删除消息数（未读数）"""
        stmt = (
            select(func.count())
            .select_from(SysAiMessage)
            .where(
                SysAiMessage.conversation_id == conv_id,
                SysAiMessage.deleted == 0,
                SysAiMessage.id > after_id,
            )
        )
        return (await db.execute(stmt)).scalar() or 0

    async def list_for_summary(
        self,
        db: AsyncSession,
        conv_id: int,
        watermark: int,
    ) -> list[SysAiMessage]:
        """摘要候选消息：水位之后的未删除消息，按时间倒序（最近的在前）。

        摘要服务取"水位之后、最近 N 轮之前"的消息，倒序取用后由调用方
        切片去除最近 N 条并反转回正序。
        """
        stmt = (
            select(SysAiMessage)
            .where(
                SysAiMessage.conversation_id == conv_id,
                SysAiMessage.deleted == 0,
                SysAiMessage.id > watermark,
            )
            .order_by(SysAiMessage.create_time.desc(), SysAiMessage.id.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_last_message_id(
        self,
        db: AsyncSession,
        conv_id: int,
    ) -> int | None:
        """查询会话最后一条未删除消息 ID（用于已读/未读计算）"""
        stmt = (
            select(SysAiMessage.id)
            .where(
                SysAiMessage.conversation_id == conv_id,
                SysAiMessage.deleted == 0,
            )
            .order_by(SysAiMessage.create_time.desc(), SysAiMessage.id.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_children(
        self,
        db: AsyncSession,
        conv_id: int,
        parent_msg_id: int,
    ) -> list[SysAiMessage]:
        """查询某消息的所有子消息（分支列表），按时间倒序"""
        stmt = select(SysAiMessage).where(
            SysAiMessage.conversation_id == conv_id,
            SysAiMessage.parent_message_id == parent_msg_id,
            SysAiMessage.deleted == 0,
        )
        stmt = stmt.order_by(SysAiMessage.create_time.desc(), SysAiMessage.id.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_chain_by_id(
        self,
        db: AsyncSession,
        conv_id: int,
        start_id: int | None,
        limit: int | None = None,
        max_hops: int = 200,
    ) -> list[SysAiMessage]:
        """沿 parent_message_id 链回溯当前激活分支的消息（过滤 deleted=1，按时间正序）。

        why: 分支对话下上下文须严格取自 current_branch_message_id 所在链，避免其他分支
        消息污染；用 get_by_ids 批量取并在内存组链，链异常（环/超长）自动截断防死循环。
        limit=None 时返回全量链（如导出场景）；全量模式 visited 集合已天然防环，
        max_hops 仅作兜底，放宽为 1000 以避免正常长会话被截断；带 limit 的上下文模式维持 200。
        """
        if limit is None:
            max_hops = 1000
        chain: list[SysAiMessage] = []
        current_id = start_id
        visited: set[int] = set()
        hops = 0
        while current_id and hops < max_hops:
            if current_id in visited:
                break
            visited.add(current_id)
            batch = await self.get_by_ids(db, [current_id])
            hops += 1
            if not batch:
                break
            msg = batch[0]
            if msg.conversation_id != conv_id:
                break
            if msg.deleted == 0:
                chain.append(msg)
                if limit is not None and len(chain) >= limit:
                    break
            current_id = msg.parent_message_id
        chain.reverse()
        return chain

    async def update_status(
        self,
        db: AsyncSession,
        msg_id: int,
        status: int,
        error: str | None = None,
    ) -> None:
        values: dict = {"status": status}
        if error is not None:
            values["error"] = error
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(**values)
        await db.execute(stmt)

    async def list_anomaly_status_by_conversations(
        self,
        db: AsyncSession,
        conv_ids: list[int],
    ) -> dict[int, set[int]]:
        """按会话汇总异常消息状态：{conv_id: {status, ...}}（3:失败;4:已取消）"""
        if not conv_ids:
            return {}
        stmt = (
            select(SysAiMessage.conversation_id, SysAiMessage.status)
            .where(
                SysAiMessage.conversation_id.in_(conv_ids),
                SysAiMessage.status.in_((3, 4)),
            )
            .distinct()
        )
        result: dict[int, set[int]] = {}
        for conv_id, status in (await db.execute(stmt)).all():
            result.setdefault(conv_id, set()).add(status)
        return result

    async def find_latest_ids_by_keyword(
        self,
        db: AsyncSession,
        conv_ids: list[int],
        keyword: str,
    ) -> dict[int, int]:
        """按会话定位命中关键词的最新消息：{conv_id: msg_id}（搜索命中消息内容时供前端定位）"""
        if not conv_ids or not keyword:
            return {}
        like_pattern = f"%{escape_like(keyword)}%"
        stmt = (
            select(SysAiMessage.conversation_id, func.max(SysAiMessage.id))
            .where(
                SysAiMessage.conversation_id.in_(conv_ids),
                SysAiMessage.deleted == 0,
                SysAiMessage.content.like(like_pattern, escape="\\"),
            )
            .group_by(SysAiMessage.conversation_id)
        )
        rows = (await db.execute(stmt)).all()
        return {row[0]: row[1] for row in rows if row[0]}

    async def update_task_id(self, db: AsyncSession, msg_id: int, task_id: str) -> None:
        """更新 assistant 消息关联的异步任务 ID（async_wait 中断时写入）。"""
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(task_id=task_id)
        await db.execute(stmt)


ai_message_repository = AiMessageRepository()
