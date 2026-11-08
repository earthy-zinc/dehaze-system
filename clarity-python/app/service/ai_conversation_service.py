"""AI 会话与消息管理服务"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_lock import acquire_lock, release_lock
from app.infrastructure.llm.call.llm_client import llm_client
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.models.entity.sys_ai_conversation import SysAiConversation
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.schema.ai_contract_gen import Interrupt, Message
from app.models.schema.ai_conversation import (
    ConversationResult,
    MessageListResult,
    MessageResume,
)
from app.models.schema.common import PageResult
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_agent_thought_repository import ai_agent_thought_repository
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_trace_repository import ai_trace_repository
from app.repository.user_repository import user_repository
from app.service.ai import message_streaming
from app.service.ai.middleware.interrupt_handler import interrupt_handler
from app.service.ai.service.conversation_search_service import (
    defer_conversation_sync,
    search_conversations,
)
from app.service.ai.service.message_view import (
    build_message_detail,
    build_messages,
    interrupt_view,
    load_sibling_positions,
)
from app.service.ai.service.model_defaults import resolve_default_chat_model
from app.service.ai.service.reasoning_service import reasoning_service
from app.service.ai.strategies.scene_templates import SCENE_VALUES, get_scene_prompt

logger = logging.getLogger(__name__)

# 置顶会话上限
PINNED_CONVERSATION_LIMIT = 10
# 置顶串行化锁 TTL（秒）：仅覆盖"校验上限 + 置顶写入 + 提交"临界区
_PIN_LOCK_TTL = 10
# 软删除恢复窗口（天）
RECYCLE_WINDOW_DAYS = 30
# 会话标题最大字数（对齐 T-CV-010：≤20 字）
TITLE_MAX_LEN = 20

# 思考链残留：部分模型（Qwen3 未关思考时）把推理过程与正文一起回传
_THINK_CLOSED_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# 推理式自述开头：模型把「我该怎么起标题」当标题输出，这类文本不是标题
_REASONING_TITLE_PREFIXES = (
    "嗯",
    "哦",
    "好的",
    "让我",
    "我需要",
    "我们需要",
    "首先",
    "这个对话",
    "对话内容",
    "根据对话",
    "用户让我",
    "用户要求",
)


def _iter_payload(payload: str):
    """按行产出导出内容（配合 StreamingResponse 流式写出）"""
    yield payload


def _clean_generated_title(raw: str) -> str:
    """清洗 LLM 标题输出：剥离思考链、取首行、去引号与前缀。

    为空、超 TITLE_MAX_LEN 字，或形如模型内心独白（「嗯，用户让我为…」）时返回空串，
    由调用方降级为首条用户消息截断。会话标题会被用户直接看到，不能出现推理过程。
    """
    title = _THINK_CLOSED_RE.sub("", raw or "")
    # 未闭合的思考块：其后的内容都是推理，直接丢弃
    open_at = title.lower().find("<think")
    if open_at != -1:
        title = title[:open_at]
    title = title.strip()
    if not title:
        return ""
    title = title.splitlines()[0].strip()
    for prefix in ("标题：", "标题:", "Title:", "title:"):
        if title.startswith(prefix):
            title = title[len(prefix) :].strip()
            break
    title = title.strip("“”\"'`#* ").strip()
    if not 0 < len(title) <= TITLE_MAX_LEN:
        return ""
    # 推理式自述：模型把「我该怎么起标题」当标题输出（本地 Qwen3 未关思考时高发）
    if title.startswith(_REASONING_TITLE_PREFIXES):
        return ""
    return title


class AiConversationService:
    def __init__(
        self,
        ai_conversation_repository=ai_conversation_repository,
        ai_message_repository=ai_message_repository,
        ai_agent_thought_repository=ai_agent_thought_repository,
        reasoning_service=reasoning_service,
        llm_client=llm_client,
        sse_emitter_manager=sse_emitter_manager,
        interrupt_handler=interrupt_handler,
        ai_agent_repository=ai_agent_repository,
        ai_agent_version_repository=ai_agent_version_repository,
        user_repository=user_repository,
        get_redis_client=get_redis_client,
    ):
        self.ai_conversation_repository = ai_conversation_repository
        self.ai_message_repository = ai_message_repository
        self.ai_agent_thought_repository = ai_agent_thought_repository
        self.reasoning_service = reasoning_service
        self.llm_client = llm_client
        self.sse_emitter_manager = sse_emitter_manager
        self.interrupt_handler = interrupt_handler
        self.ai_agent_repository = ai_agent_repository
        self.ai_agent_version_repository = ai_agent_version_repository
        self.user_repository = user_repository
        self.get_redis_client = get_redis_client

    async def _run_resume(
        self,
        conv_id: int,
        user_id: int,
        msg_id: int,
        resume_data: dict,
        stream_session_id: str,
    ) -> None:
        """后台任务：恢复中断推理，结束后关闭 SSE 流"""
        try:
            await self.reasoning_service.resume(
                conv_id, user_id, msg_id, resume_data, stream_session_id
            )
        except Exception as e:
            logger.error("AI 推理恢复失败: %s", e, exc_info=True)
        finally:
            await self.sse_emitter_manager.stop_stream(stream_session_id)

    async def _resume_stream(
        self,
        conv_id: int,
        stream_session_id: str,
        task: asyncio.Task,
    ):
        try:
            async for chunk in self.sse_emitter_manager.create_stream(conv_id, stream_session_id):
                yield chunk
        finally:
            # 客户端断连时也等待后台任务完成，确保 assistant 消息正常落库
            if not task.done():
                await asyncio.shield(task)

    async def _resolve_agent_anchor(
        self, db: AsyncSession, agent_code: str | None
    ) -> tuple[str, int | None]:
        """解析会话锚定的 (agent_code, agent_version)。

        agent_code 为空使用默认 Agent；版本锚定取该 Agent 当前已发布版本号
        （无已发布版本时为 None，reasoning 时按需取用）。
        """
        code = (agent_code or "default").strip() or "default"
        agent = await self.ai_agent_repository.get_by_code(db, code)
        if not agent or agent.deleted:
            return code, None
        published = await self.ai_agent_version_repository.get_latest_published(db, agent.id)
        return code, published.version_no if published else None

    async def create_conversation(self, db: AsyncSession, user_id: int, form) -> ConversationResult:
        agent_code, agent_version = await self._resolve_agent_anchor(db, form.agentCode)
        # 场景提示词：显式传入 systemPrompt 优先；否则按 scene 写默认模板。
        # 场景语义已固化进 prompt 文本，无需给表加 scene 列。
        scene = (form.scene or "general") if form.scene in SCENE_VALUES else "general"
        system_prompt = form.systemPrompt or get_scene_prompt(scene)
        conv = SysAiConversation(
            user_id=user_id,
            title=form.title or "新对话",
            # 未指定模型：用户偏好 > 该 Agent 配置的模型 > 平台默认模型（见 model_defaults）
            model=form.model or await resolve_default_chat_model(db, user_id, agent_code),
            agent_code=agent_code,
            agent_version=agent_version,
            system_prompt=system_prompt,
            model_config=form.modelConfig,
            suggestions_enabled=int(form.suggestionsEnabled),
            api_key_id=form.apiKeyId,
            status=1,
        )
        conv = await self.ai_conversation_repository.create(db, conv)
        defer_conversation_sync(db, conv.id)
        return ConversationResult.model_validate(conv)

    async def list_conversations(
        self,
        db: AsyncSession,
        user_id: int,
        page: int,
        size: int,
        keyword: str | None = None,
        status: int | None = None,
        view: str | None = None,
    ) -> PageResult[ConversationResult]:
        if view == "admin":
            # 管理端审计视角（后端实现-会话与消息.md §7）：全量用户会话，不过滤 user_id，
            # 响应含 userId；审计范围关键词用 DB like，不引 ES 全量检索
            audit_status = None if status == 0 else status
            if keyword:
                convs, total = await self.ai_conversation_repository.paginate_all_with_keyword(
                    db, page, size, keyword, status=audit_status
                )
            else:
                convs, total = await self.ai_conversation_repository.paginate_all_conversations(
                    db, page, size, status=audit_status
                )
            results = [await self._to_result(db, c) for c in convs]
            await self._attach_audit_fields(db, results)
            return PageResult(list=results, total=total)
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
            convs = await self.ai_conversation_repository.get_by_ids_and_user(db, conv_ids, user_id)
            convs = self._sort_conversations(convs)
            results = [await self._to_result(db, c) for c in convs]
            await self._attach_matched_messages(db, results, keyword)
            return PageResult(list=results, total=total)
        convs, total = await self.ai_conversation_repository.paginate_user_conversations(
            db, user_id, page, size, status=status_filter
        )
        return PageResult(list=[await self._to_result(db, c) for c in convs], total=total)

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
            result.unread_count = await self.ai_message_repository.count_messages_after(
                db, conv.id, conv.last_read_message_id
            )
        else:
            result.unread_count = conv.message_count or 0
        return result

    async def _attach_audit_fields(
        self, db: AsyncSession, results: list[ConversationResult]
    ) -> None:
        """审计视角补充用户名、消耗汇总与异常标注（批量查询，避免逐会话 N+1）"""
        if not results:
            return
        conv_ids = [r.id for r in results]
        names = await self.user_repository.get_display_names(db, {r.user_id for r in results})
        consumption = await self.ai_conversation_repository.sum_consumption_by_conversation(
            db, conv_ids
        )
        anomaly_status = await self.ai_message_repository.list_anomaly_status_by_conversations(
            db, conv_ids
        )
        quota_conv_ids = await self.ai_conversation_repository.list_quota_anomaly_conversation_ids(
            db, conv_ids
        )
        risky_tool_conv_ids = await ai_trace_repository.list_risky_tool_conversation_ids(
            db, conv_ids
        )
        for result in results:
            result.user_name = names.get(result.user_id)
            stat = consumption.get(result.id) or {}
            result.token_consumed = stat.get("token", 0)
            result.credits_consumed = stat.get("credits", 0)
            statuses = anomaly_status.get(result.id, set())
            if "failed" in statuses:
                result.anomaly_type, result.anomaly_label = "failed", "存在失败消息"
            elif result.id in quota_conv_ids:
                result.anomaly_type, result.anomaly_label = "quota", "配额不足中断"
            elif result.id in risky_tool_conv_ids:
                result.anomaly_type, result.anomaly_label = "risky_tool", "存在高风险工具调用"
            elif "canceled" in statuses:
                result.anomaly_type, result.anomaly_label = "canceled", "存在已取消消息"

    async def _attach_matched_messages(
        self, db: AsyncSession, results: list[ConversationResult], keyword: str
    ) -> None:
        """回填搜索命中消息内容的消息ID（标题已命中无需定位，跳过）"""
        if not results or not keyword:
            return
        conv_ids = [r.id for r in results if keyword not in (r.title or "")]
        if not conv_ids:
            return
        matched = await self.ai_message_repository.find_latest_ids_by_keyword(db, conv_ids, keyword)
        for result in results:
            result.matched_message_id = matched.get(result.id)

    async def get_conversation(
        self, db: AsyncSession, conv_id: int, user_id: int, admin: bool = False
    ) -> ConversationResult:
        """会话详情。admin=True 为管理端审计视角：不限归属用户并补充审计字段"""
        if admin:
            conv = await self.ai_conversation_repository.get_by_id(db, conv_id)
            if not conv:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
            result = await self._to_result(db, conv)
            await self._attach_audit_fields(db, [result])
            return result
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        return ConversationResult.model_validate(conv)

    async def update_conversation(
        self, db: AsyncSession, conv_id: int, user_id: int, form
    ) -> ConversationResult:
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        # 置顶前状态须在字段回写前取值：下面 field_map 会先把 conv.pinned 改成目标值
        was_pinned = conv.pinned == 1
        data = form.model_dump(exclude_unset=True)
        field_map = {
            "title": "title",
            "model": "model",
            "systemPrompt": "system_prompt",
            "modelConfig": "model_config",
            "suggestionsEnabled": "suggestions_enabled",
            "pinned": "pinned",
            "status": "status",
        }
        for camel, snake in field_map.items():
            if camel in data:
                setattr(conv, snake, data[camel])
        # 通过 PATCH 置顶：同步维护 pinned_at（置顶写时间，取消置顶清空）；
        # 已置顶会话不再刷新 pinned_at，避免无谓改动打乱置顶排序
        if "pinned" in data:
            if data["pinned"]:
                if not was_pinned:
                    conv.pinned_at = await self._pin_with_limit(db, conv_id, user_id)
            else:
                conv.pinned_at = None
        if "agentCode" in data:
            # 切换 Agent：重新锚定版本（下一条消息生效）；未同时显式指定模型时，
            # 会话模型随之按新 Agent 重新解析（否则"换了 Agent 模型不变"）
            agent_code, agent_version = await self._resolve_agent_anchor(db, data.get("agentCode"))
            conv.agent_code = agent_code
            conv.agent_version = agent_version
            if "model" not in data:
                conv.model = await resolve_default_chat_model(db, user_id, agent_code)
        if "title" in data:
            conv.title_source = "manual"
        await db.flush()
        await db.refresh(conv)
        # 会话状态/标题变化触发 ES 文档幂等更新（事务提交后执行）
        if "status" in data or "title" in data or "pinned" in data:
            defer_conversation_sync(db, conv.id)
        return await self._to_result(db, conv)

    async def _ensure_pin_limit(self, db: AsyncSession, user_id: int) -> None:
        """校验当前用户置顶数量上限"""
        count = await self.ai_conversation_repository.count_active_pinned(db, user_id)
        if count >= PINNED_CONVERSATION_LIMIT:
            raise BusinessException(ResultCode.DATA_EXISTS, "置顶会话已达上限")

    async def _pin_with_limit(self, db: AsyncSession, conv_id: int, user_id: int) -> datetime:
        """置顶会话并占用名额，返回置顶时间。

        why: 上限校验与置顶写入分离（check-then-act）时，并发置顶同一用户的不同会话会
        双双通过校验导致超限；故在用户级锁内完成校验、写入并提交，后到请求读到的是已
        生效的置顶数。
        """
        lock_key = f"ai:conv:pin:{user_id}"
        token = await acquire_lock(lock_key, _PIN_LOCK_TTL)
        if token is None:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "置顶操作并发冲突，请稍后再试")
        try:
            await self._ensure_pin_limit(db, user_id)
            now = datetime.now()
            await self.ai_conversation_repository.set_pinned(db, conv_id, 1, now)
            await db.commit()
        finally:
            await release_lock(lock_key, token)
        return now

    async def _auto_generate_title(self, conversation_id: int, first_user_content: str) -> None:
        """异步用 LLM 生成标题（清洗后限 20 字），不可用时降级截取首条消息前 20 字"""
        fallback = (first_user_content or "新对话").replace("\n", " ").strip()[:TITLE_MAX_LEN]
        title = ""
        try:
            async with get_db_session() as db:
                conv = await self.ai_conversation_repository.get_by_id(db, conversation_id)
                if not conv:
                    return
                model_id = conv.model or await resolve_default_chat_model(db)
                prompt = (
                    f"请为以下对话生成一个不超过{TITLE_MAX_LEN}个字的标题，"
                    "直接输出标题本身，不要引号、不要解释、不要任何前缀。\n"
                    f"对话内容：{first_user_content}"
                )
                chunks = [
                    chunk.content
                    async for chunk in self.llm_client.stream_chat(
                        db,
                        model_id,
                        [{"role": "user", "content": prompt}],
                        # 带 system 提示：本地 provider 按 Qwen3 约定在 system 尾部注入 /no_think
                        # 关闭思考，从源头避免推理过程混进标题（远程 provider 视为普通指令）
                        system_prompt="你是会话标题生成器：只输出标题本身，不要思考过程、不要解释、不要引号、不要前缀。",
                        max_tokens=50,
                        user_id=conv.user_id,
                    )
                    if chunk.type == "text_delta"
                ]
                title = _clean_generated_title("".join(chunks))
        except Exception:
            logger.warning("LLM 生成标题失败，降级截取首条消息前 20 字", exc_info=True)
            title = ""
        if not title:
            title = fallback or "新对话"
        async with get_db_session() as db:
            await self.ai_conversation_repository.update_title(
                db, conversation_id, title, title_source="auto"
            )

    async def delete_conversation(self, db: AsyncSession, conv_id: int, user_id: int) -> None:
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        await self.ai_conversation_repository.soft_delete_by_ids(db, [conv.id])
        # 软删同步 ES，全文检索默认过滤已删会话
        defer_conversation_sync(db, conv.id)

    async def list_messages(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        before: int | None,
        limit: int,
        admin: bool = False,
    ) -> MessageListResult:
        """会话消息列表（游标分页，admin=True 为管理端审计视角，
        路由层已校验 ai:conversation:audit）。

        self 视角（默认）只返回当前激活链：以会话 current_branch_message_id 为链尾沿
        parent_message_id 回溯（指针为空/悬空/指向已删消息时回退到会话最大 id 的未删
        消息），链即用户实际看到的对话。分页在链上切：before 之前的最后 limit 条按 id
        倒序返回，hasMore 表示链上是否还有更早消息，total 为链长——分支消息不再混入
        列表，前端无需再做"每个分叉取最新"的选择算法。

        admin 视角保持全量分支分页（审计需要看到所有分叉），total 为会话消息总数。
        siblingIndex/siblingCount 仅在 self 视角的 assistant 消息上填充。

        出参为契约 Message：内容在 blocks，计划与中断挂在消息上，刷新后与流式终态同形。
        """
        if admin:
            conv = await self.ai_conversation_repository.get_by_id(db, conv_id)
        else:
            conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        if admin:
            msgs, total, has_more = await self.ai_message_repository.list_by_conversation_cursor(
                db, conv_id, before, limit
            )
            # 契约 Message 一次批量装配（产物 ID 单次 IN 查询，避免逐条 N+1）
            return MessageListResult(
                list=await build_messages(db, msgs), total=total, hasMore=has_more
            )
        chain = await self._load_active_chain(db, conv)
        eligible = chain if before is None else [m for m in chain if m.id < before]
        has_more = len(eligible) > limit
        # 链回溯按 id 升序，接口契约仍按 id 倒序返回（最新一页在前）
        page = list(reversed(eligible[-limit:]))
        sibling_positions = await load_sibling_positions(
            self.ai_message_repository, db, conv_id, page
        )
        return MessageListResult(
            list=await build_messages(db, page, sibling_positions=sibling_positions),
            total=len(chain),
            hasMore=has_more,
        )

    async def _load_active_chain(
        self,
        db: AsyncSession,
        conv: SysAiConversation,
    ) -> list[SysAiMessage]:
        """当前激活链（id 升序）：指针为空/悬空/指向已删消息时回退到最新未删消息。"""
        tail_id = conv.current_branch_message_id
        if tail_id is not None:
            tail = await self.ai_message_repository.get_by_id(db, tail_id)
            if tail is None or tail.conversation_id != conv.id:
                tail_id = None
        if tail_id is None:
            tail_id = await self.ai_message_repository.get_max_message_id(db, conv.id)
        if tail_id is None:
            return []
        return await self.ai_message_repository.get_chain_by_id(db, conv.id, tail_id)

    async def get_message(
        self, db: AsyncSession, msg_id: int, user_id: int, admin: bool = False
    ) -> dict:
        """消息详情。admin=True 为管理端审计视角：不限归属用户"""
        msg = await self._load_message(db, msg_id, user_id, admin=admin)
        positions = await load_sibling_positions(
            self.ai_message_repository, db, msg.conversation_id, [msg]
        )
        detail = await build_message_detail(db, msg, sibling_position=positions.get(msg.id))
        return detail.model_dump(by_alias=True, mode="json")

    async def _load_message(
        self, db: AsyncSession, msg_id: int, user_id: int, *, admin: bool = False
    ) -> SysAiMessage:
        if admin:
            msg = await self.ai_message_repository.get_by_id(db, msg_id)
        else:
            msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        return msg

    async def get_interrupt(
        self, db: AsyncSession, msg_id: int, user_id: int, admin: bool = False
    ) -> Interrupt | None:
        """查询消息的待恢复中断：无中断返回 null，已过期返回 actionable=false。

        DB 的 pending_interrupt 是事实源：刷新、重启、跨实例后仍能查到同一条中断。
        """
        msg = await self._load_message(db, msg_id, user_id, admin=admin)
        return interrupt_view(msg.pending_interrupt)

    async def get_branches(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        msg_id: int,
    ) -> list[Message]:
        """查询某消息的分支回复（assistant 子消息），按时间倒序。

        只列 assistant：user 子消息不是可切换的分支（分支切换器选择的是助手回复）；
        删除校验仍取全部子消息，两者口径不同见 delete_message。
        """
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg or msg.conversation_id != conv_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        children = await self.ai_message_repository.get_children(
            db, conv_id, msg_id, role="assistant"
        )
        return await build_messages(db, children)

    async def _assert_not_suspended(self, db: AsyncSession, conv: SysAiConversation) -> None:
        """当前指针存在待恢复中断时拒绝改变会话状态。

        why: 推理中断时图暂停、SSE 流挂起等待 resume，此时切换分支或重新生成会改变
        会话上下文，与 send/edit 的挂起校验同口径，避免闯入挂起流。

        Redis 中断句柄只是同一次运行内的快速通道，DB 的 pending_interrupt 才是事实源：
        句柄丢失（重启/换实例）时继续按当前指针消息判定——status=awaiting_input 且
        interrupt_view 判定 actionable（未过期）仍视为挂起，否则待确认中断会因切换分支
        失去入口，用户再也无法恢复或停止。
        """
        tail_id = conv.current_branch_message_id
        if tail_id is None:
            return
        if await self.interrupt_handler.get_interrupt(f"{conv.id}:{tail_id}"):
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "会话有未完成的中断确认，请先确认或停止"
            )
        msg = await self.ai_message_repository.get_by_id(db, tail_id)
        if msg is None or msg.conversation_id != conv.id or msg.status != "awaiting_input":
            return
        pending = interrupt_view(msg.pending_interrupt)
        if pending is not None and pending.actionable:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "会话有未完成的中断确认，请先确认或停止"
            )

    async def switch_branch(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        msg_id: int,
    ) -> ConversationResult:
        """切换当前激活分支（更新 current_branch_message_id）。

        挂起期间拒绝切换：send/regenerate 的挂起校验依赖当前指针，指针移走后检查
        对象随之改变，会让待确认的中断失去入口。
        """
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg or msg.conversation_id != conv_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        await self._assert_not_suspended(db, conv)
        await self.ai_conversation_repository.update_current_branch(db, conv_id, msg_id)
        conv.current_branch_message_id = msg_id
        return ConversationResult.model_validate(conv)

    async def regenerate_message(
        self, db: AsyncSession, msg_id: int, user_id: int
    ) -> StreamingResponse:
        """重新生成助手回复：基于原 assistant 的父 user 消息新建兄弟分支并触发推理（SSE 流式）"""
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if msg.role != "assistant":
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可重新生成")
        if msg.deleted:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "该消息已删除，无法重新生成")
        if not msg.parent_message_id:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "该消息无父消息，无法重新生成")
        conv = await self.ai_conversation_repository.get_by_id_and_user(
            db, msg.conversation_id, user_id
        )
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        if conv.status != 1:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "会话已归档，无法重新生成")
        # 中断挂起（待确认）时拒绝重新生成，避免闯入挂起流
        await self._assert_not_suspended(db, conv)
        if not await self.sse_emitter_manager.acquire_lock(conv.id):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该会话正在生成回复，请稍后再试")

        model = (
            msg.model
            or conv.model
            or await resolve_default_chat_model(db, conv.user_id, conv.agent_code)
        )
        stream_session_id = str(uuid4())
        try:
            # 新 assistant 消息：parent 沿用原 assistant 的父 user 消息，与原回复形成兄弟分支
            new_msg = SysAiMessage(
                conversation_id=msg.conversation_id,
                parent_message_id=msg.parent_message_id,
                role="assistant",
                model=model,
                status="running",
                blocks=[],
                task_id=stream_session_id,
            )
            new_msg = await self.ai_message_repository.create(db, new_msg)
            await self.ai_conversation_repository.update_last_message(
                db, msg.conversation_id, new_msg.id, datetime.now()
            )
            # 显式提交：推理后台任务用独立 session 读取上下文，须先提交本事务
            await db.commit()
        except Exception:
            # 锁获流式锁后建消息失败：释放锁，避免会话被 120s 锁 TTL 阻塞
            await self.sse_emitter_manager.release_lock(msg.conversation_id)
            raise

        # regenerate 为显式操作，无需幂等；复用 send 的 SSE 触发链路（上下文由 reasoning 重建）
        idem_key = f"ai:msg:idempotent:{user_id}:{uuid4()}"
        return StreamingResponse(
            message_streaming.stream_generator(
                sse_emitter_manager=self.sse_emitter_manager,
                reasoning_service=self.reasoning_service,
                get_redis_client=self.get_redis_client,
                conv_id=msg.conversation_id,
                user_id=user_id,
                model=model,
                assistant_msg_id=new_msg.id,
                # 契约 message.start 必须携带本轮 user 消息：重生成沿用原 assistant 的父 user
                user_msg_id=msg.parent_message_id,
                stream_session_id=stream_session_id,
                idem_key=idem_key,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def stop_message(self, db: AsyncSession, msg_id: int, user_id: int) -> Message:
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        # 中断挂起（awaiting_input）同样允许停止：推理已暂停等待确认，用户放弃本轮
        # 时必须能终结消息，否则会话被"待确认"永久卡住（新消息与重新生成均被拒绝）
        if msg.role != "assistant" or msg.status not in ("running", "awaiting_input"):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "当前消息不可停止")
        # 任务标识优先级：async_wait 批量任务 ID（task_id 列）> 消息→流会话反查键
        # （send/resume 路径 task_id 不存流会话 ID，见 message_streaming 登记键）
        target_id = msg.task_id
        if not target_id:
            redis = await self.get_redis_client()
            target_id = await redis.get(f"ai:msg:stream:{msg_id}")
        blocks = None
        if target_id:
            # 推理侧先把当前块快照刷库并返回：流式期间 blocks 只在内存累积，
            # 直接读库会拿到 []，响应会把前端已渲染内容清空
            blocks = await self.reasoning_service.stop(msg.conversation_id, msg_id, target_id)
        else:
            # 无活跃流（中断挂起且流会话键已过期）：仍须清中断点并终结，
            # 否则 pending_interrupt 留在库里，刷新后仍显示"待确认"
            await self.interrupt_handler.clear_interrupt(f"{msg.conversation_id}:{msg_id}")
            await self.ai_message_repository.update_status(db, msg_id, "canceled")
            await self.ai_message_repository.update_pending_interrupt(db, msg_id, None)
        if blocks is not None:
            msg.blocks = blocks
        msg.status = "canceled"
        msg.pending_interrupt = None
        positions = await load_sibling_positions(
            self.ai_message_repository, db, msg.conversation_id, [msg]
        )
        return await build_message_detail(db, msg, sibling_position=positions.get(msg.id))

    async def resume_message(
        self,
        db: AsyncSession,
        msg_id: int,
        user_id: int,
        form: MessageResume,
    ) -> StreamingResponse:
        """恢复中断的推理（算法推荐确认/拒绝），SSE 续流"""
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        conv_id = msg.conversation_id
        # 中断以 DB 的 pending_interrupt 为准：Redis 句柄丢失（重启/换实例）不影响恢复
        pending = interrupt_view(msg.pending_interrupt)
        if pending is None:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到中断，无法恢复")
        if not pending.actionable:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "中断已过期，无法恢复")
        if not await self.sse_emitter_manager.acquire_lock(conv_id):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该会话正在生成回复，请稍后再试")

        # 按中断类型组装恢复载荷：
        # - confirm：确认/拒绝算法推荐，需 confirmed + 可选 modelId
        # - plan_approve：计划确认/干预（Plan-and-Execute），透传 planEdit
        # - quota：用户升级 VIP 后直接恢复，无需业务载荷（reasoning 层转 Command(resume=True)）
        # - async_wait：由任务完成回调自动恢复，不走本端点（此处兜底按空载荷继续）
        if pending.type == "confirm":
            if form.confirm is None:
                await self.sse_emitter_manager.release_lock(conv_id)
                raise BusinessException(ResultCode.PARAM_ERROR, "confirm 中断必须携带确认结果")
            resume_data = {"confirmed": form.confirm, **(form.params or {})}
        elif pending.type == "plan_approve":
            resume_data = {"plan_edit": form.plan_edit, **(form.params or {})}
        else:
            resume_data = form.params or {}
        # 续流用新的流会话：旧流已在上一次终结时关闭，续写下发必须落在新流上
        stream_session_id = str(uuid4())
        # 重绑消息→流反查键：停止端点按该键定位当前续流，否则仍停在旧流上
        # （旧流已关闭，停止标志虽生效但客户端收不到终帧）；同时清掉 async_wait
        # 遗留的批量任务 ID（任务已完成，停止应作用于当前流而非已结束的后台任务）
        msg.task_id = None
        await self.ai_message_repository.update_task_id(db, msg_id, None)
        redis = await self.get_redis_client()
        await message_streaming.bind_message_stream(redis, msg_id, stream_session_id)
        # 显式提交：停止端点用独立 session 读 task_id，不提交会读到已失效的旧任务标识
        await db.commit()
        await self.sse_emitter_manager.register_stream(stream_session_id)
        task = asyncio.create_task(
            self._run_resume(conv_id, user_id, msg_id, resume_data, stream_session_id)
        )
        return StreamingResponse(
            self._resume_stream(conv_id, stream_session_id, task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def delete_message(self, db: AsyncSession, msg_id: int, user_id: int) -> None:
        """删除助手消息（软删除）。

        删除策略（user/assistant 交替，不做链桥接）：
        - 仍有未删子消息（非叶子）拒绝删除：中间节点软删会让后续消息失去父链、历史不可读，
          须先删该回复之后的对话；
        - 被删消息恰为当前分支指针时，指针回退到其父消息（可为空），避免指针悬空导致
          链回溯在已删节点断开；
        - 非激活分支上的旧消息允许删除，指针不动。

        并发：进入临界区先对会话行 SELECT ... FOR UPDATE——children 校验、软删、指针回退
        全部在持锁期间完成，与并发 send 的 update_last_message 天然串行（同一会话行互斥），
        指针回退再用 CAS 兜底，避免本请求读到的旧指针覆盖 send 已推进的新指针。
        """
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if msg.role != "assistant":
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可删除")
        conv = await self.ai_conversation_repository.get_by_id_and_user(
            db, msg.conversation_id, user_id, for_update=True
        )
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        children = await self.ai_message_repository.get_children(db, msg.conversation_id, msg.id)
        if children:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "请先删除该回复之后的对话")
        await self.ai_message_repository.soft_delete_by_ids(db, [msg.id])
        # 指针回退与软删同事务提交（事务边界由 DBSessionMiddleware 统一管理）
        if conv.current_branch_message_id == msg.id:
            moved = await self.ai_conversation_repository.update_current_branch_if(
                db,
                conv.id,
                expected_message_id=msg.id,
                message_id=msg.parent_message_id,
            )
            if moved:
                conv.current_branch_message_id = msg.parent_message_id

    # ==================== 会话生命周期扩展 ====================

    async def _get_owned_active(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> SysAiConversation:
        """查询归属于当前用户且未删除的会话，不存在则抛错"""
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
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
                    await self.ai_conversation_repository.update_status(db, [conv.id], 2)
                    defer_conversation_sync(db, conv.id)
                elif action == "restore":
                    if conv.status != 2:
                        raise BusinessException(
                            ResultCode.DATA_STATE_NOT_ALLOW, "仅已归档会话可恢复"
                        )
                    await self.ai_conversation_repository.update_status(db, [conv.id], 1)
                    defer_conversation_sync(db, conv.id)
                elif action == "delete":
                    if not confirm:
                        raise BusinessException(ResultCode.PARAM_ERROR, "批量删除需二次确认")
                    await self.ai_conversation_repository.soft_delete_by_ids(db, [conv.id])
                    defer_conversation_sync(db, conv.id)
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
        conv = await self.ai_conversation_repository.get_in_trash(
            db, conv_id, user_id, window_start
        )
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在或已超出恢复窗口")
        await self.ai_conversation_repository.restore_by_ids(db, [conv.id])
        conv.deleted = 0
        conv.delete_time = None
        defer_conversation_sync(db, conv.id)
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
        convs, total = await self.ai_conversation_repository.paginate_trash(
            db, user_id, page, size, window_start
        )
        return PageResult(list=[await self._to_result(db, c) for c in convs], total=total)

    async def pin_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> ConversationResult:
        """置顶会话（上限校验）"""
        conv = await self._get_owned_active(db, conv_id, user_id)
        if conv.pinned != 1:
            conv.pinned = 1
            conv.pinned_at = await self._pin_with_limit(db, conv_id, user_id)
            # _pin_with_limit 在用户级锁内 commit（并发置顶的上限校验依赖提交后可见），
            # commit 会 expire 本 session 的全部 ORM 实例；不刷新则序列化时访问 pinned
            # 触发同步懒加载 → MissingGreenlet（ConversationResult 校验失败 500）
            await db.refresh(conv)
        return await self._to_result(db, conv)

    async def unpin_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
    ) -> ConversationResult:
        """取消置顶"""
        conv = await self._get_owned_active(db, conv_id, user_id)
        await self.ai_conversation_repository.set_pinned(db, conv_id, 0, None)
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
        last_msg_id = await self.ai_message_repository.get_last_message_id(db, conv_id)
        if last_msg_id is not None:
            await self.ai_conversation_repository.mark_read(db, conv_id, last_msg_id)
            conv.last_read_message_id = last_msg_id
        return await self._to_result(db, conv)

    async def export_conversation(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        fmt: str = "markdown",
    ) -> StreamingResponse:
        """导出会话：沿当前激活分支回溯全部消息，仅导 user/assistant 的正文块。

        正文由 blocks 中的 text 块按 seq 拼接（思考与工具块不导出）。
        """
        conv = await self._get_owned_active(db, conv_id, user_id)
        tail_msg_id = conv.current_branch_message_id
        messages: list[SysAiMessage] = []
        if tail_msg_id:
            # 仅取 user/assistant 消息（推理过程/工具调用不导出）
            messages = [
                m
                for m in await self.ai_message_repository.get_chain_by_id(
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
                            "content": m.text,
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
                lines.append(m.text)
                lines.append("")
            payload = "\n".join(lines)
            media_type = "text/markdown"

        return StreamingResponse(
            _iter_payload(payload),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


ai_conversation_service = AiConversationService()
