"""AI 会话与消息管理服务"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.schema.ai_conversation import (
    AgentThoughtResult,
    ConversationResult,
    MessageResult,
    MessageResume,
)
from app.models.schema.common import PageResult
from app.repository.ai_agent_thought_repository import ai_agent_thought_repository
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.service.ai.conversation_search_service import (
    search_conversations,
    sync_conversation_to_es,
)
from app.service.ai.interrupt_handler import interrupt_handler
from app.service.ai.llm_client import llm_client
from app.service.ai.reasoning_service import reasoning_service
from app.service.ai.scene_templates import SCENE_VALUES, get_scene_prompt

logger = logging.getLogger(__name__)

# 置顶会话上限
PINNED_CONVERSATION_LIMIT = 10
# 软删除恢复窗口（天）
RECYCLE_WINDOW_DAYS = 30


def _iter_payload(payload: str):
    """按行产出导出内容（配合 StreamingResponse 流式写出）"""
    yield payload


async def _run_resume(
    conv_id: int,
    user_id: int,
    msg_id: int,
    resume_data: dict,
    stream_session_id: str,
) -> None:
    """后台任务：恢复中断推理，结束后关闭 SSE 流"""
    try:
        await reasoning_service.resume(conv_id, user_id, msg_id, resume_data)
    except Exception as e:
        logger.error("AI 推理恢复失败: %s", e, exc_info=True)
    finally:
        await sse_emitter_manager.stop_stream(stream_session_id)


async def _resume_stream(
    conv_id: int,
    stream_session_id: str,
    task: asyncio.Task,
):
    try:
        async for chunk in sse_emitter_manager.create_stream(conv_id, stream_session_id):
            yield chunk
    finally:
        # 客户端断连时也等待后台任务完成，确保 assistant 消息正常落库
        if not task.done():
            await asyncio.shield(task)


class AiConversationService:
    async def _resolve_agent_anchor(
        self, db: AsyncSession, agent_code: str | None
    ) -> tuple[str, int | None]:
        """解析会话锚定的 (agent_code, agent_version)。

        agent_code 为空使用默认 Agent；版本锚定取该 Agent 当前已发布版本号
        （无已发布版本时为 None，reasoning 时按需取用）。
        """
        from app.repository.ai_agent_repository import ai_agent_repository
        from app.repository.ai_agent_version_repository import ai_agent_version_repository

        code = (agent_code or "default").strip() or "default"
        agent = await ai_agent_repository.get_by_code(db, code)
        if not agent or agent.deleted:
            return code, None
        published = await ai_agent_version_repository.get_latest_published(db, agent.id)
        return code, published.version_no if published else None

    async def create_conversation(self, db: AsyncSession, user_id: int, form) -> ConversationResult:
        agent_code, agent_version = await self._resolve_agent_anchor(
            db, form.agentCode
        )
        # 场景提示词：显式传入 systemPrompt 优先；否则按 scene 写默认模板。
        # 场景语义已固化进 prompt 文本，无需给表加 scene 列。
        scene = (form.scene or "general") if form.scene in SCENE_VALUES else "general"
        system_prompt = form.systemPrompt or get_scene_prompt(scene)
        conv = SysAiConversation(
            user_id=user_id,
            title=form.title or "新对话",
            model=form.model or settings.AI_DEFAULT_MODEL,
            agent_code=agent_code,
            agent_version=agent_version,
            system_prompt=system_prompt,
            model_config=form.modelConfig,
            api_key_id=form.apiKeyId,
            status=1,
        )
        conv = await ai_conversation_repository.create(db, conv)
        return ConversationResult.model_validate(conv)

    async def list_conversations(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
    ) -> PageResult[ConversationResult]:
        # 三态范围过滤：默认(缺省/None)仅活跃(1)，0=全部(不过滤)，1=活跃，2=归档。
        # 将 0 归一为 None（repo/ES 均以 None 表示不按状态过滤）。
        if status is None:
            status = 1
        status_filter = None if status == 0 else status
        # keyword 搜索走 ES 全文检索（分页 + 计数），无命中即空结果；无 keyword 走 DB 分页
        if keyword:
            conv_ids, total = await search_conversations(
                user_id, keyword, status=status_filter, page=page, size=size
            )
            convs = await ai_conversation_repository.get_by_ids(db, user_id, conv_ids)
            convs = self._sort_conversations(convs)
            return PageResult(
                list=[await self._to_result(db, c) for c in convs], total=total
            )
        convs, total = await ai_conversation_repository.paginate_user_conversations(
            db, user_id, page, size, status=status_filter
        )
        return PageResult(
            list=[await self._to_result(db, c) for c in convs], total=total
        )

    def _sort_conversations(self, convs: list[SysAiConversation]) -> list[SysAiConversation]:
        """置顶（pinned_at 倒序）优先，其余按最后消息时间倒序"""
        return sorted(
            convs,
            key=lambda c: (
                not c.pinned,
                -(c.pinned_at.timestamp() if c.pinned_at else 0),
                -(c.last_message_at.timestamp() if c.last_message_at else 0),
                -c.id,
            ),
        )

    async def _to_result(self, db: AsyncSession, conv: SysAiConversation) -> ConversationResult:
        """构建会话结果，补充未读数"""
        result = ConversationResult.model_validate(conv)
        if conv.last_read_message_id:
            result.unread_count = await ai_message_repository.count_messages_after(
                db, conv.id, conv.last_read_message_id
            )
        else:
            result.unread_count = conv.message_count or 0
        return result

    async def get_conversation(self, db: AsyncSession, conv_id: int, user_id: int) -> ConversationResult:
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        return ConversationResult.model_validate(conv)

    async def update_conversation(
        self, db: AsyncSession, conv_id: int, user_id: int, form
    ) -> ConversationResult:
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        data = form.model_dump(exclude_unset=True)
        field_map = {
            "title": "title",
            "model": "model",
            "systemPrompt": "system_prompt",
            "modelConfig": "model_config",
            "pinned": "pinned",
            "status": "status",
        }
        for camel, snake in field_map.items():
            if camel in data:
                setattr(conv, snake, data[camel])
        # 通过 PATCH 置顶：同步维护 pinned_at（置顶写时间，取消置顶清空）
        if "pinned" in data:
            if data["pinned"]:
                if conv.pinned != 1:
                    await self._ensure_pin_limit(db, user_id)
                conv.pinned_at = datetime.now()
            else:
                conv.pinned_at = None
        if "agentCode" in data:
            # 切换 Agent：重新锚定版本（下一条消息生效）
            agent_code, agent_version = await self._resolve_agent_anchor(
                db, data.get("agentCode")
            )
            conv.agent_code = agent_code
            conv.agent_version = agent_version
        if "title" in data:
            conv.title_source = "manual"
        await db.flush()
        await db.refresh(conv)
        # 会话状态/标题变化触发 ES 文档幂等更新
        if "status" in data or "title" in data or "pinned" in data:
            await sync_conversation_to_es(conv.id)
        return await self._to_result(db, conv)

    async def _ensure_pin_limit(self, db: AsyncSession, user_id: int) -> None:
        """校验当前用户置顶数量上限"""
        count = await ai_conversation_repository.count_active_pinned(db, user_id)
        if count >= PINNED_CONVERSATION_LIMIT:
            raise BusinessException(ResultCode.DATA_EXISTS, "置顶会话已达上限")

    async def _auto_generate_title(self, conversation_id: int, first_user_content: str) -> None:
        """异步用 LLM 生成标题，失败降级截取前 20 字"""
        title = ""
        try:
            async with get_db_session() as db:
                conv = await ai_conversation_repository.get_by_id(db, conversation_id)
                if not conv:
                    return
                model_id = conv.model or settings.AI_DEFAULT_MODEL
                prompt = f"请为以下对话生成一个简洁的标题（不超过20字）：{first_user_content}"
                redis = await get_redis_client()
                chunks = []
                async for chunk in llm_client.stream_chat(
                    db, redis, model_id, [{"role": "user", "content": prompt}], max_tokens=50
                ):
                    if chunk.type == "text_delta":
                        chunks.append(chunk.content)
                title = "".join(chunks).strip()
        except Exception:
            logger.warning("LLM 生成标题失败，降级截取前 20 字", exc_info=True)
            title = ""
        if not title:
            title = (first_user_content or "新对话")[:20]
        async with get_db_session() as db:
            await ai_conversation_repository.update_title(
                db, conversation_id, title, title_source="auto"
            )

    async def delete_conversation(self, db: AsyncSession, conv_id: int, user_id: int) -> None:
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        await ai_conversation_repository.soft_delete_by_ids(db, [conv.id])
        # 软删同步 ES（deleted=1），全文检索默认过滤已删会话
        await sync_conversation_to_es(conv.id)

    async def list_messages(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        page: int,
        size: int,
    ) -> PageResult[MessageResult]:
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        msgs, total = await ai_message_repository.list_by_conversation(db, conv_id, page, size)
        return PageResult(list=[MessageResult.model_validate(m) for m in msgs], total=total)

    async def get_message(self, db: AsyncSession, msg_id: int, user_id: int) -> dict:
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        result = MessageResult.model_validate(msg).model_dump(by_alias=True)
        thoughts = await ai_agent_thought_repository.list_by_message(db, msg_id)
        result["thoughts"] = [
            AgentThoughtResult.model_validate(t).model_dump(by_alias=True) for t in thoughts
        ]
        return result

    async def get_branches(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        msg_id: int,
    ) -> list[MessageResult]:
        """查询某消息的所有子消息（分支列表），按时间倒序"""
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg or msg.conversation_id != conv_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        children = await ai_message_repository.get_children(db, conv_id, msg_id)
        return [MessageResult.model_validate(m) for m in children]

    async def switch_branch(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        msg_id: int,
    ) -> ConversationResult:
        """切换当前激活分支（更新 current_branch_message_id）"""
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg or msg.conversation_id != conv_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        await ai_conversation_repository.update_current_branch(db, conv_id, msg_id)
        conv.current_branch_message_id = msg_id
        return ConversationResult.model_validate(conv)

    async def regenerate_message(self, db: AsyncSession, msg_id: int, user_id: int) -> StreamingResponse:
        """重新生成助手回复：基于原 assistant 的父 user 消息新建兄弟分支并触发推理（SSE 流式）"""
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if msg.role != "assistant":
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可重新生成")
        if msg.deleted:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "该消息已删除，无法重新生成")
        if not msg.parent_message_id:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "该消息无父消息，无法重新生成")
        conv = await ai_conversation_repository.get_by_id_and_user(db, msg.conversation_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        if conv.status != 1:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "会话已归档，无法重新生成")
        # 中断挂起（待确认）时拒绝重新生成，避免闯入挂起流
        if conv.current_branch_message_id and await interrupt_handler.get_interrupt(
            f"{conv.id}:{conv.current_branch_message_id}"
        ):
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "会话有未完成的中断确认，请先确认或停止"
            )
        if not await sse_emitter_manager.acquire_lock(conv.id):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该会话正在生成回复，请稍后再试")

        model = msg.model or conv.model or settings.AI_DEFAULT_MODEL
        stream_session_id = str(uuid4())
        # 新 assistant 消息：parent 沿用原 assistant 的父 user 消息，与原回复形成兄弟分支
        new_msg = SysAiMessage(
            conversation_id=msg.conversation_id,
            parent_message_id=msg.parent_message_id,
            role="assistant",
            content="",
            model=model,
            status=1,
            task_id=stream_session_id,
        )
        new_msg = await ai_message_repository.create(db, new_msg)
        await ai_conversation_repository.update_last_message(
            db, msg.conversation_id, new_msg.id, datetime.now()
        )

        # regenerate 为显式操作，无需幂等；复用 send 的 SSE 触发链路（上下文由 reasoning 重建）
        from app.service.ai_message_service import _stream_generator

        idem_key = f"ai:msg:idempotent:{user_id}:{uuid4()}"
        return StreamingResponse(
            _stream_generator(
                conv_id=msg.conversation_id,
                user_id=user_id,
                model=model,
                assistant_msg_id=new_msg.id,
                stream_session_id=stream_session_id,
                idem_key=idem_key,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def stop_message(self, db: AsyncSession, msg_id: int, user_id: int) -> MessageResult:
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if msg.role != "assistant" or msg.status != 1:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "当前消息不可停止")
        if msg.task_id:
            await reasoning_service.stop(msg.conversation_id, msg_id, msg.task_id)
        else:
            await ai_message_repository.update_status(db, msg_id, 4)
        msg.status = 4
        return MessageResult.model_validate(msg)

    async def resume_message(
        self,
        db: AsyncSession,
        msg_id: int,
        user_id: int,
        form: MessageResume,
    ) -> StreamingResponse:
        """恢复中断的推理（算法推荐确认/拒绝），SSE 续流"""
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        conv_id = msg.conversation_id
        thread_id = f"{conv_id}:{msg_id}"
        interrupt = await interrupt_handler.get_interrupt(thread_id)
        if not interrupt:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到中断点，无法恢复")
        stream_session_id = (interrupt.get("data") or {}).get("stream_session_id", "")
        if not stream_session_id:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "中断点缺少流会话，无法恢复")
        if not await sse_emitter_manager.acquire_lock(conv_id):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该会话正在生成回复，请稍后再试")

        # 按中断类型组装恢复载荷：
        # - confirm：确认/拒绝算法推荐，需 confirmed + 可选 algorithmId
        # - plan_approve：计划确认/干预（Plan-and-Execute），透传 plan_edit
        # - quota：用户升级 VIP 后直接恢复，无需业务载荷（reasoning 层转 Command(resume=True)）
        # - async_wait：由任务完成回调自动恢复，不走本端点（此处兜底按空载荷继续）
        if interrupt.get("type") == "confirm":
            if form.confirm is None:
                raise BusinessException(ResultCode.PARAM_ERROR, "confirm 中断必须携带确认结果")
            resume_data = {"confirmed": form.confirm, **(form.params or {})}
        elif interrupt.get("type") == "plan_approve":
            resume_data = {"plan_edit": form.plan_edit, **(form.params or {})}
        else:
            resume_data = form.params or {}
        task = asyncio.create_task(
            _run_resume(conv_id, user_id, msg_id, resume_data, stream_session_id)
        )
        return StreamingResponse(
            _resume_stream(conv_id, stream_session_id, task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def delete_message(self, db: AsyncSession, msg_id: int, user_id: int) -> None:
        """删除助手消息（软删除）；仅 assistant 消息可删，删除后不参与上下文"""
        msg = await ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if msg.role != "assistant":
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可删除")
        await ai_message_repository.soft_delete_by_ids(db, [msg.id])

    # ==================== 会话生命周期扩展 ====================

    async def _get_owned_active(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> SysAiConversation:
        """查询归属于当前用户且未删除的会话，不存在则抛错"""
        conv = await ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        return conv

    async def batch_operate(
        self,
        db: AsyncSession,
        user_id: int,
        action: str,
        ids: list[int],
        confirm: bool = False,
    ) -> int:
        """批量操作会话：逐条复用单操作校验，任一失败整体回滚并返回首个错误。

        archive: 归档（status=2）；restore: 撤销归档（status=1）；delete: 软删除（需 confirm）。
        """
        try:
            count = 0
            for conv_id in ids:
                conv = await self._get_owned_active(db, conv_id, user_id)
                if action == "archive":
                    if conv.status != 1:
                        raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅活跃会话可归档")
                    await ai_conversation_repository.update_status(db, [conv.id], 2)
                    await sync_conversation_to_es(conv.id)
                elif action == "restore":
                    if conv.status != 2:
                        raise BusinessException(
                            ResultCode.DATA_STATE_NOT_ALLOW, "仅已归档会话可恢复"
                        )
                    await ai_conversation_repository.update_status(db, [conv.id], 1)
                    await sync_conversation_to_es(conv.id)
                elif action == "delete":
                    if not confirm:
                        raise BusinessException(ResultCode.PARAM_ERROR, "批量删除需二次确认")
                    await ai_conversation_repository.soft_delete_by_ids(db, [conv.id])
                    await sync_conversation_to_es(conv.id)
                count += 1
            return count
        except Exception:
            await db.rollback()
            raise

    async def restore_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> ConversationResult:
        """软删恢复：仅 30 天恢复窗口内可恢复"""
        window_start = datetime.now() - timedelta(days=RECYCLE_WINDOW_DAYS)
        conv = await ai_conversation_repository.get_in_trash(db, conv_id, user_id, window_start)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在或已超出恢复窗口")
        await ai_conversation_repository.restore_by_ids(db, [conv.id])
        conv.deleted = 0
        conv.delete_time = None
        await sync_conversation_to_es(conv.id)
        return await self._to_result(db, conv)

    async def list_trash(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
    ) -> PageResult[ConversationResult]:
        """回收站列表：已软删且未超 30 天，按 delete_time 倒序"""
        window_start = datetime.now() - timedelta(days=RECYCLE_WINDOW_DAYS)
        convs, total = await ai_conversation_repository.paginate_trash(
            db, user_id, page, size, window_start
        )
        return PageResult(
            list=[await self._to_result(db, c) for c in convs], total=total
        )

    async def pin_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> ConversationResult:
        """置顶会话（上限校验）"""
        conv = await self._get_owned_active(db, conv_id, user_id)
        if conv.pinned != 1:
            await self._ensure_pin_limit(db, user_id)
        await ai_conversation_repository.set_pinned(db, conv_id, 1, datetime.now())
        conv.pinned = 1
        conv.pinned_at = datetime.now()
        return await self._to_result(db, conv)

    async def unpin_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> ConversationResult:
        """取消置顶"""
        conv = await self._get_owned_active(db, conv_id, user_id)
        await ai_conversation_repository.set_pinned(db, conv_id, 0, None)
        conv.pinned = 0
        conv.pinned_at = None
        return await self._to_result(db, conv)

    async def mark_read(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> ConversationResult:
        """标记已读：last_read_message_id 置为会话最后一条消息 ID"""
        conv = await self._get_owned_active(db, conv_id, user_id)
        last_msg_id = await ai_message_repository.get_last_message_id(db, conv_id)
        if last_msg_id is not None:
            await ai_conversation_repository.mark_read(db, conv_id, last_msg_id)
            conv.last_read_message_id = last_msg_id
        return await self._to_result(db, conv)

    async def export_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        fmt: str = "markdown",
    ) -> StreamingResponse:
        """导出会话：沿当前激活分支回溯全部消息，过滤推理/工具调用，仅导 user/assistant content。"""
        conv = await self._get_owned_active(db, conv_id, user_id)
        tail_msg_id = conv.current_branch_message_id
        messages: list[SysAiMessage] = []
        if tail_msg_id:
            # 仅取 user/assistant 消息（推理过程/工具调用不导出）
            messages = [
                m
                for m in await ai_message_repository.get_chain_by_id(
                    db, conv_id, tail_msg_id, limit=None
                )
                if m.role in ("user", "assistant")
            ]

        ext = "md" if fmt == "markdown" else fmt
        filename = f"conversation_{conv_id}.{ext}"
        if fmt == "json":
            payload = json.dumps(
                {
                    "conversation": {
                        "id": conv.id,
                        "title": conv.title,
                        "model": conv.model,
                        "agent_code": conv.agent_code,
                        "create_time": conv.create_time.isoformat() if conv.create_time else None,
                    },
                    "messages": [
                        {
                            "role": m.role,
                            "content": m.content or "",
                            "create_time": m.create_time.isoformat() if m.create_time else None,
                        }
                        for m in messages
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            media_type = "application/json"
        else:
            lines = [f"# {conv.title}", ""]
            role_label = {"user": "用户", "assistant": "助手"}
            for m in messages:
                lines.append(f"## {role_label.get(m.role, m.role)}")
                lines.append("")
                lines.append(m.content or "")
                lines.append("")
            payload = "\n".join(lines)
            media_type = "text/markdown"

        return StreamingResponse(
            _iter_payload(payload),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


ai_conversation_service = AiConversationService()
