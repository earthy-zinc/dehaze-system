"""AI 消息发送服务（SSE 流式输出）"""

import asyncio
import json
import logging
from datetime import datetime
from uuid import uuid4

from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import success
from app.dependencies.redis import get_redis_client
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.models.entity.sys_ai_message import SysAiMessage
from app.models.schema.ai_conversation import MessageResult
from app.repository.ai_agent_repository import ai_agent_repository
from app.repository.ai_conversation_repository import ai_conversation_repository
from app.repository.ai_message_repository import ai_message_repository
from app.repository.ai_model_repository import ai_model_repository
from app.service.ai.message_streaming import run_reasoning, stream_generator
from app.service.ai.middleware.interrupt_handler import interrupt_handler
from app.service.ai.service.reasoning_service import reasoning_service
from app.service.ai_conversation_service import ai_conversation_service
from app.service.ai_model_service import ai_model_service

logger = logging.getLogger(__name__)

_IDEMPOTENT_PREFIX = "ai:msg:idempotent:"


def _has_attachments(content: str) -> bool:
    """判断消息是否含图片/文件附件：识别 markdown 图片链接或文件下载引用"""
    if not content:
        return False
    return (
        ("![" in content and "]" in content and "(" in content)
        or ("[文件](" in content)
        or ("[图片](" in content)
    )


class AiMessageService:
    def __init__(
        self,
        ai_conversation_repository=ai_conversation_repository,
        ai_message_repository=ai_message_repository,
        ai_agent_repository=ai_agent_repository,
        ai_model_repository=ai_model_repository,
        reasoning_service=reasoning_service,
        sse_emitter_manager=sse_emitter_manager,
        interrupt_handler=interrupt_handler,
        ai_conversation_service=ai_conversation_service,
        ai_model_service=ai_model_service,
        get_redis_client=get_redis_client,
    ):
        self.ai_conversation_repository = ai_conversation_repository
        self.ai_message_repository = ai_message_repository
        self.ai_agent_repository = ai_agent_repository
        self.ai_model_repository = ai_model_repository
        self.reasoning_service = reasoning_service
        self.sse_emitter_manager = sse_emitter_manager
        self.interrupt_handler = interrupt_handler
        self.ai_conversation_service = ai_conversation_service
        self.ai_model_service = ai_model_service
        self.get_redis_client = get_redis_client

    async def _assert_conversation_not_suspended(self, conv) -> None:
        """会话处于中断挂起（待确认）时拒绝发起新流式操作。

        why: 推理中断时图暂停、SSE 流挂起等待 resume，会话并发锁已让渡；
        挂起期间新发送应得到明确的"待确认"错误，而非并发冲突或闯入挂起流。
        """
        if conv.current_branch_message_id and await self.interrupt_handler.get_interrupt(
            f"{conv.id}:{conv.current_branch_message_id}"
        ):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "会话有未完成的中断确认，请先确认或停止")

    async def _needs_tool_call(self, db: AsyncSession, conv) -> bool:
        """判断本次会话推理是否需要工具调用（用于 supports_tool_call 校验）。

        why: 推理引擎已重构为 deepagents，主 Agent 恒定装载业务工具
        （algorithm_recommend 等），除 reasoning_mode=direct 直连路径外几乎所有
        推理都要求模型 supports_tool_call。故默认按 True 校验（宁可对 direct 误拦，
        也不放行后让推理到 tool_call 阶段才中途失败）。仅当会话锚定的 Agent 显式
        固定为 direct（无工具直连回复）且未在模型参数里显式声明 needTools 时才跳过。
        """
        if (conv.model_config or {}).get("needTools") is not None:
            return bool(conv.model_config["needTools"])
        # 与 resolve_reasoning_mode 共用会话锚定 Agent 的 reasoning_mode 判定：
        # 固定为 direct 才跳过工具校验；auto 在此保守按需工具校验（不重复跑复杂度评估，
        # 其运行时解析由 reasoning 层统一负责），避免对 direct 之外的范式放行后中途失败。
        agent = await self.ai_agent_repository.get_by_code(db, conv.agent_code or "default")
        if agent and agent.reasoning_mode == "direct":
            return False
        return True

    async def _run_reasoning(
        self,
        conv_id: int,
        user_id: int,
        model: str,
        assistant_msg_id: int,
        stream_session_id: str,
        idem_key: str,
    ) -> None:
        """后台任务：调用 ReasoningService 推理，成功后写入幂等键（下沉至共享模块）"""
        await run_reasoning(
            reasoning_service=self.reasoning_service,
            get_redis_client=self.get_redis_client,
            sse_emitter_manager=self.sse_emitter_manager,
            conv_id=conv_id,
            user_id=user_id,
            model=model,
            assistant_msg_id=assistant_msg_id,
            stream_session_id=stream_session_id,
            idem_key=idem_key,
        )

    def _stream_generator(
        self,
        conv_id: int,
        user_id: int,
        model: str,
        assistant_msg_id: int,
        stream_session_id: str,
        idem_key: str,
    ):
        """SSE 流式消息生成器（下沉至共享模块，供 send/edit/regenerate 复用）"""
        return stream_generator(
            sse_emitter_manager=self.sse_emitter_manager,
            reasoning_service=self.reasoning_service,
            get_redis_client=self.get_redis_client,
            conv_id=conv_id,
            user_id=user_id,
            model=model,
            assistant_msg_id=assistant_msg_id,
            stream_session_id=stream_session_id,
            idem_key=idem_key,
        )

    async def _idempotent_response(self, db: AsyncSession, existing: str) -> JSONResponse:
        """幂等命中（已完成）：返回已有消息结果"""
        try:
            data = json.loads(existing)
            msg = await self.ai_message_repository.get_by_id(db, data.get("messageId"))
            if msg:
                return JSONResponse(
                    content=success(
                        MessageResult.model_validate(msg).model_dump(by_alias=True)
                    ).model_dump()
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return JSONResponse(content=success().model_dump())

    async def send_message(
        self,
        db: AsyncSession,
        conv_id: int,
        user_id: int,
        form,
        idempotency_key: str,
    ) -> StreamingResponse:
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        if conv.status != 1:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "会话已归档，无法发送消息")
        await self._assert_conversation_not_suspended(conv)
        if len(form.content) > settings.AI_MESSAGE_MAX_LENGTH:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"消息长度不能超过 {settings.AI_MESSAGE_MAX_LENGTH} 字符"
            )

        redis = await self.get_redis_client()
        idem_key = f"{_IDEMPOTENT_PREFIX}{user_id}:{idempotency_key}"
        existing = await redis.get(idem_key)
        if existing:
            if existing == "pending":
                # pending 命中：409 冲突语义（对齐设计 §4.2），用 REPEAT_SUBMIT_ERROR 编码表达
                raise BusinessException(ResultCode.REPEAT_SUBMIT_ERROR)
            return await self._idempotent_response(db, existing)
        # pending TTL 对齐流式超时（120s）+ 60s 余量，避免长推理 pending 过期后同 key 重复落库
        await redis.set(idem_key, "pending", ex=settings.AI_MESSAGE_STREAM_TIMEOUT + 60)

        if not await self.sse_emitter_manager.acquire_lock(conv_id):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该会话正在生成回复，请稍后再试")

        model = form.model or conv.model or settings.AI_DEFAULT_MODEL

        # 模型能力校验（§2.8）：消息含附件需多模态、本会话推理需工具调用需 supports_tool_call
        model_entity = await self.ai_model_repository.get_by_model_id(db, model)
        if not model_entity or model_entity.status != 1:
            raise BusinessException(ResultCode.AI_MODEL_NOT_AVAILABLE, "模型不可用或已禁用")
        await self.ai_model_service.validate_model_caps(
            model_entity,
            has_attachments=_has_attachments(form.content),
            need_tools=await self._needs_tool_call(db, conv),
        )

        stream_session_id = str(uuid4())
        # 捕获"首条消息"判定所需的本条消息前状态：update_last_message 的批量 UPDATE
        # 会经 SQLAlchemy synchronize_session 使 conv 属性失效，之后访问 conv.message_count
        # 会重读数据库（已 +2），导致 <=1 判定恒假、首条消息永不触发标题生成。故提前取值。
        prev_message_count = conv.message_count
        prev_title = conv.title
        prev_title_source = conv.title_source

        user_msg = SysAiMessage(
            conversation_id=conv_id,
            parent_message_id=conv.current_branch_message_id,
            role="user",
            content=form.content,
            model=model,
            status=2,
        )
        user_msg = await self.ai_message_repository.create(db, user_msg)

        assistant_msg = SysAiMessage(
            conversation_id=conv_id,
            parent_message_id=user_msg.id,
            role="assistant",
            content="",
            model=model,
            status=1,
            # task_id 仅承载异步任务 ID（async_wait 中断时写入），不存流会话标识
            task_id=None,
        )
        assistant_msg = await self.ai_message_repository.create(db, assistant_msg)

        # 消息计数收敛于 update_last_message 单点（本次追加 user+assistant 两条）
        await self.ai_conversation_repository.update_last_message(
            db, conv_id, assistant_msg.id, datetime.now(), message_delta=2
        )

        # 首条消息发送后异步用 LLM 生成标题（不阻塞消息发送）
        if (
            prev_message_count <= 1
            and prev_title == "新对话"
            and prev_title_source != "manual"
        ):
            asyncio.create_task(
                self.ai_conversation_service._auto_generate_title(conv_id, form.content)
            )

        # 上下文由 reasoning_service.run 内部组装，此处不再预热（避免 build_context 二次执行）
        return StreamingResponse(
            self._stream_generator(
                conv_id=conv_id,
                user_id=user_id,
                model=model,
                assistant_msg_id=assistant_msg.id,
                stream_session_id=stream_session_id,
                idem_key=idem_key,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def edit_message(
        self,
        db: AsyncSession,
        user_id: int,
        msg_id: int,
        form,
    ) -> StreamingResponse:
        """编辑已发送的用户消息并重新触发回复（SSE 流式）"""
        msg = await self.ai_message_repository.get_by_id_and_user(db, msg_id, user_id)
        if not msg:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")
        if msg.role != "user":
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅用户消息可编辑")
        if len(form.content) > settings.AI_MESSAGE_MAX_LENGTH:
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, f"消息长度不能超过 {settings.AI_MESSAGE_MAX_LENGTH} 字符"
            )

        conv_id = msg.conversation_id
        conv = await self.ai_conversation_repository.get_by_id_and_user(db, conv_id, user_id)
        if not conv:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")
        if conv.status != 1:
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "会话已归档，无法发送消息")
        await self._assert_conversation_not_suspended(conv)
        if not await self.sse_emitter_manager.acquire_lock(conv_id):
            raise BusinessException(ResultCode.BUSINESS_ERROR, "该会话正在生成回复，请稍后再试")

        # 原消息标记为已编辑，保留原文（content 不变，前端用 original_content 展示编辑前）
        msg.edited = 1
        msg.original_content = msg.content
        await db.flush()

        model = conv.model or settings.AI_DEFAULT_MODEL
        stream_session_id = str(uuid4())

        # 新 user 消息：parent 沿用原 user 消息的 parent，保持上下文链
        user_msg = SysAiMessage(
            conversation_id=conv_id,
            parent_message_id=msg.parent_message_id,
            role="user",
            content=form.content,
            model=model,
            status=2,
        )
        user_msg = await self.ai_message_repository.create(db, user_msg)

        assistant_msg = SysAiMessage(
            conversation_id=conv_id,
            parent_message_id=user_msg.id,
            role="assistant",
            content="",
            model=model,
            status=1,
            # task_id 仅承载异步任务 ID（async_wait 中断时写入），不存流会话标识
            task_id=None,
        )
        assistant_msg = await self.ai_message_repository.create(db, assistant_msg)

        # 消息计数收敛于 update_last_message 单点（本次追加 user+assistant 两条）
        await self.ai_conversation_repository.update_last_message(
            db, conv_id, assistant_msg.id, datetime.now(), message_delta=2
        )

        # 编辑重发不参与幂等，使用一次性 key 复用推理后台任务；
        # 上下文由 reasoning_service.run 内部组装，此处不再预热（避免 build_context 二次执行）
        idem_key = f"{_IDEMPOTENT_PREFIX}{user_id}:{uuid4()}"

        return StreamingResponse(
            self._stream_generator(
                conv_id=conv_id,
                user_id=user_id,
                model=model,
                assistant_msg_id=assistant_msg.id,
                stream_session_id=stream_session_id,
                idem_key=idem_key,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )


ai_message_service = AiMessageService()
