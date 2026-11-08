from typing import Any

from sqlalchemy import String, cast, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.repository.base import BaseRepository, escape_like

# 异常消息状态：失败与取消（会话审计标注用）
_ANOMALY_STATUSES = ("failed", "canceled")


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

    async def list_by_conversation_cursor(
        self,
        db: AsyncSession,
        conv_id: int,
        before: int | None,
        limit: int,
    ) -> tuple[list[SysAiMessage], int, bool]:
        """会话消息游标分页：按 id 倒序取一页（id 单调自增等价时间倒序）。

        before 非空时仅返回 id < before 的消息（缺省取最新一页）；多取一条判定
        hasMore（是否还存在更早消息）；total 为会话消息总数（展示用）。
        """
        base = select(SysAiMessage).where(
            SysAiMessage.conversation_id == conv_id,
            SysAiMessage.deleted == 0,
        )
        # count 聚合恒返回单行，scalar_one() 取该行整数（契约保证恒有值，不以 or 0 掩盖）
        total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
        stmt = base
        if before is not None:
            stmt = stmt.where(SysAiMessage.id < before)
        stmt = stmt.order_by(SysAiMessage.id.desc()).limit(limit + 1)
        rows = list((await db.execute(stmt)).scalars().all())
        has_more = len(rows) > limit
        return rows[:limit], total, has_more

    async def get_by_id(
        self,
        db: AsyncSession,
        id: int,
        *,
        with_deleted: bool = False,
    ) -> SysAiMessage | None:
        """按主键查消息（不限归属用户，供管理端审计 view=admin 使用）"""
        stmt = select(SysAiMessage).where(SysAiMessage.id == id)
        if not with_deleted:
            stmt = stmt.where(SysAiMessage.deleted == 0)
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

    async def get_max_message_id(
        self,
        db: AsyncSession,
        conv_id: int,
    ) -> int | None:
        """会话最大未删除消息 ID（分支指针缺失/悬空时回退的链尾）。"""
        stmt = select(func.max(SysAiMessage.id)).where(
            SysAiMessage.conversation_id == conv_id,
            SysAiMessage.deleted == 0,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    async def list_assistant_sibling_ids(
        self,
        db: AsyncSession,
        conv_id: int,
        parent_ids: list[int],
    ) -> dict[int, list[int]]:
        """批量取指定父消息下未删 assistant 兄弟 ID：{parent_message_id: [id 升序]}。

        一次 IN 查询按父消息分组，避免逐条消息各查一次兄弟（N+1）。siblingIndex 即
        列表下标 + 1，siblingCount 即列表长度；root 级（parent_message_id IS NULL）
        助手消息不参与本方法，调用方相应字段保持 null。
        """
        if not parent_ids:
            return {}
        stmt = (
            select(SysAiMessage.parent_message_id, SysAiMessage.id)
            .where(
                SysAiMessage.conversation_id == conv_id,
                SysAiMessage.parent_message_id.in_(parent_ids),
                SysAiMessage.role == "assistant",
                SysAiMessage.deleted == 0,
            )
            .order_by(SysAiMessage.parent_message_id.asc(), SysAiMessage.id.asc())
        )
        grouped: dict[int, list[int]] = {}
        for parent_id, msg_id in (await db.execute(stmt)).all():
            grouped.setdefault(parent_id, []).append(msg_id)
        return grouped

    async def get_children(
        self,
        db: AsyncSession,
        conv_id: int,
        parent_msg_id: int,
        role: str | None = None,
    ) -> list[SysAiMessage]:
        """查询某消息的子消息（分支列表），按时间倒序。

        role 非空时只取该角色的子消息：分支切换器只列 assistant 回复（user 子消息
        不是可选分支）；删除校验取全部子消息——任一角色的未删子消息都意味着非叶子。
        """
        stmt = select(SysAiMessage).where(
            SysAiMessage.conversation_id == conv_id,
            SysAiMessage.parent_message_id == parent_msg_id,
            SysAiMessage.deleted == 0,
        )
        if role is not None:
            stmt = stmt.where(SysAiMessage.role == role)
        stmt = stmt.order_by(SysAiMessage.create_time.desc(), SysAiMessage.id.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_chain_by_id(
        self,
        db: AsyncSession,
        conv_id: int,
        start_id: int | None,
        limit: int | None = None,
        max_hops: int | None = 200,
    ) -> list[SysAiMessage]:
        """沿 parent_message_id 链回溯当前激活分支的消息（过滤已软删行，按时间正序）。

        why: 分支对话下上下文须严格取自 current_branch_message_id 所在链，避免其他分支
        消息污染；一次查询取本会话消息后在内存组链——链回溯处于推理热路径，逐条单行
        SELECT 会放大成上百次往返。链异常（环/超长）由 visited 与 max_hops 截断防死循环。
        limit=None 时返回全量链（列表/导出场景）：visited 已天然防环，故不设 hop 上限
        ——历史不能因内部兜底被静默截断；带 limit 的上下文模式维持默认 200 兜底。

        deleted=0 显式下推：全局逻辑删除过滤器只在 ORM 事件注册后生效（应用生命周期 /
        测试夹具），本方法的结果正确性不能依赖注册时机；中间节点被软删时链在该处断开，
        只返回断点之后的活跃后缀。
        """
        if start_id is None:
            return []
        if limit is None:
            max_hops = None
        rows = await db.execute(
            select(SysAiMessage).where(
                SysAiMessage.conversation_id == conv_id,
                SysAiMessage.deleted == 0,
            )
        )
        by_id = {msg.id: msg for msg in rows.scalars().all()}
        chain: list[SysAiMessage] = []
        current_id = start_id
        visited: set[int] = set()
        hops = 0
        while current_id and (max_hops is None or hops < max_hops):
            if current_id in visited:
                break
            visited.add(current_id)
            hops += 1
            msg = by_id.get(current_id)
            if msg is None:
                break
            chain.append(msg)
            if limit is not None and len(chain) >= limit:
                break
            current_id = msg.parent_message_id
        chain.reverse()
        return chain

    async def list_anomaly_status_by_conversations(
        self,
        db: AsyncSession,
        conv_ids: list[int],
    ) -> dict[int, set[str]]:
        """按会话汇总异常消息状态：{conv_id: {status, ...}}（failed;canceled）"""
        if not conv_ids:
            return {}
        stmt = (
            select(SysAiMessage.conversation_id, SysAiMessage.status)
            .where(
                SysAiMessage.conversation_id.in_(conv_ids),
                SysAiMessage.status.in_(_ANOMALY_STATUSES),
            )
            .distinct()
        )
        result: dict[int, set[str]] = {}
        for conv_id, status in (await db.execute(stmt)).all():
            result.setdefault(conv_id, set()).add(status)
        return result

    async def find_latest_ids_by_keyword(
        self,
        db: AsyncSession,
        conv_ids: list[int],
        keyword: str,
    ) -> dict[int, int]:
        """按会话定位命中关键词的最新消息：{conv_id: msg_id}（搜索命中消息内容时供前端定位）

        正文已收敛进 blocks(JSON)，按 JSON 文本匹配：MySQL 对 JSON 列 CAST AS CHAR
        后 LIKE，命中范围与旧 content 列一致（含工具入参等块内文本）。
        """
        if not conv_ids or not keyword:
            return {}
        like_pattern = f"%{escape_like(keyword)}%"
        stmt = (
            select(SysAiMessage.conversation_id, func.max(SysAiMessage.id))
            .where(
                SysAiMessage.conversation_id.in_(conv_ids),
                SysAiMessage.deleted == 0,
                cast(SysAiMessage.blocks, String).like(like_pattern, escape="\\"),
            )
            .group_by(SysAiMessage.conversation_id)
        )
        rows = (await db.execute(stmt)).all()
        return {row[0]: row[1] for row in rows if row[0]}

    async def update_task_id(self, db: AsyncSession, msg_id: int, task_id: str | None) -> None:
        """更新/清空 assistant 消息关联的异步任务 ID（async_wait 中断时写入，恢复时清空）。"""
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(task_id=task_id)
        await db.execute(stmt)

    async def update_status(
        self,
        db: AsyncSession,
        msg_id: int,
        status: str,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status}
        if error is not None:
            values["error"] = error
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(**values)
        await db.execute(stmt)

    async def update_blocks(
        self,
        db: AsyncSession,
        msg_id: int,
        blocks: list[dict[str, Any]],
    ) -> None:
        """整篇覆盖 blocks 快照（终态落库；流式期间不逐帧写库）。"""
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(blocks=blocks)
        await db.execute(stmt)

    async def update_plan(
        self,
        db: AsyncSession,
        msg_id: int,
        plan: dict[str, Any] | None,
    ) -> None:
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(plan=plan)
        await db.execute(stmt)

    async def update_pending_interrupt(
        self,
        db: AsyncSession,
        msg_id: int,
        interrupt: dict[str, Any] | None,
    ) -> None:
        stmt = (
            update(SysAiMessage)
            .where(SysAiMessage.id == msg_id)
            .values(pending_interrupt=interrupt)
        )
        await db.execute(stmt)

    async def update_suggestions(
        self,
        db: AsyncSession,
        msg_id: int,
        suggestions: list[str] | None,
    ) -> None:
        stmt = update(SysAiMessage).where(SysAiMessage.id == msg_id).values(suggestions=suggestions)
        await db.execute(stmt)

    async def update_usage(
        self,
        db: AsyncSession,
        msg_id: int,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        credits: int,
        sub_agent_usage: list[dict] | None = None,
    ) -> None:
        stmt = (
            update(SysAiMessage)
            .where(SysAiMessage.id == msg_id)
            .values(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                credits=credits,
                sub_agent_usage=sub_agent_usage,
            )
        )
        await db.execute(stmt)

    async def update_memory_ids(
        self,
        db: AsyncSession,
        msg_id: int,
        memory_ids: list[int],
    ) -> None:
        stmt = (
            update(SysAiMessage).where(SysAiMessage.id == msg_id).values(used_memory_ids=memory_ids)
        )
        await db.execute(stmt)


ai_message_repository = AiMessageRepository()
