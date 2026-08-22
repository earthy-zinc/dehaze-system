from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.models.schema.ai_conversation import (
    ConversationBatchAction,
    ConversationCreate,
    ConversationExportQuery,
    ConversationPageQuery,
    ConversationResult,
    ConversationUpdate,
    MessageEdit,
    MessagePageQuery,
    MessageResult,
    MessageResume,
    MessageSend,
)
from app.models.schema.common import PageResult
from app.service.ai_conversation_service import ai_conversation_service
from app.service.ai_message_service import ai_message_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI对话"])


@router.post("/conversations", response_model=Result[ConversationResult], summary="创建会话")
async def create_conversation(
    form: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.create_conversation(db, user.id, form)
    return success(result)


@router.get(
    "/conversations", response_model=Result[PageResult[ConversationResult]], summary="会话列表"
)
async def list_conversations(
    query: ConversationPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.list_conversations(
        db, user.id, query.pageNum, query.pageSize, query.keyword, query.status
    )
    return success(result)


@router.get(
    "/conversations/trash",
    response_model=Result[PageResult[ConversationResult]],
    summary="回收站列表",
)
async def list_trash_conversations(
    query: ConversationPageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.list_trash(db, user.id, query.pageNum, query.pageSize)
    return success(result)


@router.post("/conversations/batch", response_model=Result[int], summary="批量操作会话")
async def batch_operate_conversations(
    form: ConversationBatchAction,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    count = await ai_conversation_service.batch_operate(
        db, user.id, form.action, form.ids, form.confirm
    )
    return success(count)


@router.get(
    "/conversations/{conv_id}", response_model=Result[ConversationResult], summary="会话详情"
)
async def get_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.get_conversation(db, conv_id, user.id)
    return success(result)


@router.patch(
    "/conversations/{conv_id}", response_model=Result[ConversationResult], summary="更新会话"
)
async def update_conversation(
    conv_id: int,
    form: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.update_conversation(db, conv_id, user.id, form)
    return success(result)


@router.delete("/conversations/{conv_id}", response_model=Result[None], summary="删除会话")
async def delete_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_conversation_service.delete_conversation(db, conv_id, user.id)
    return success(msg="一切ok")


@router.post(
    "/conversations/{conv_id}/restore",
    response_model=Result[ConversationResult],
    summary="恢复软删会话",
)
async def restore_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.restore_conversation(db, conv_id, user.id)
    return success(result)


@router.put(
    "/conversations/{conv_id}/pin", response_model=Result[ConversationResult], summary="置顶会话"
)
async def pin_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.pin_conversation(db, conv_id, user.id)
    return success(result)


@router.put(
    "/conversations/{conv_id}/unpin", response_model=Result[ConversationResult], summary="取消置顶"
)
async def unpin_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.unpin_conversation(db, conv_id, user.id)
    return success(result)


@router.put(
    "/conversations/{conv_id}/read",
    response_model=Result[ConversationResult],
    summary="标记会话已读",
)
async def read_conversation(
    conv_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.mark_read(db, conv_id, user.id)
    return success(result)


@router.get("/conversations/{conv_id}/export", summary="导出会话")
async def export_conversation(
    conv_id: int,
    query: ConversationExportQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return await ai_conversation_service.export_conversation(db, conv_id, user.id, query.format)


@router.post("/conversations/{conv_id}/messages", summary="发送消息（SSE流式）")
async def send_message(
    conv_id: int,
    form: MessageSend,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
    # 发送触发 LLM 调用与计费，幂等键必传（缺失返回 422），保证同 key 只执行一次
    idempotency_key: str = Header(
        alias="Idempotency-Key", description="幂等键（UUID，客户端生成）"
    ),
):
    return await ai_message_service.send_message(db, conv_id, user.id, form, idempotency_key)


@router.get(
    "/conversations/{conv_id}/messages",
    response_model=Result[PageResult[MessageResult]],
    summary="会话消息列表",
)
async def list_messages(
    conv_id: int,
    query: MessagePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.list_messages(
        db, conv_id, user.id, query.pageNum, query.pageSize
    )
    return success(result)


@router.get("/conversations/{conv_id}/messages/stream/{stream_session_id}", summary="SSE断线重连")
async def reconnect(
    conv_id: int,
    stream_session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    # 校验会话归属，防止越权访问他人流
    await ai_conversation_service.get_conversation(db, conv_id, user.id)
    last_event_id = int(request.headers.get("Last-Event-ID", "0"))
    return StreamingResponse(
        sse_emitter_manager.reconnect(stream_session_id, last_event_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/messages/{msg_id}", summary="消息详情（含推理步骤）")
async def get_message(
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.get_message(db, msg_id, user.id)
    return success(result)


@router.post("/messages/{msg_id}/regenerate", summary="重新生成回复（SSE流式）")
async def regenerate_message(
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return await ai_conversation_service.regenerate_message(db, msg_id, user.id)


@router.post("/messages/{msg_id}/resume", summary="恢复中断的推理（SSE续流）")
async def resume_message(
    msg_id: int,
    form: MessageResume,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return await ai_conversation_service.resume_message(db, msg_id, user.id, form)


@router.delete(
    "/messages/{msg_id}", response_model=Result[None], summary="删除助手回复消息（软删除）"
)
async def delete_message(
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await ai_conversation_service.delete_message(db, msg_id, user.id)
    return success(msg="一切ok")


@router.put("/messages/{msg_id}", summary="编辑用户消息并重新触发回复（SSE流式）")
async def edit_message(
    msg_id: int,
    form: MessageEdit,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return await ai_message_service.edit_message(db, user.id, msg_id, form)


@router.get(
    "/conversations/{conv_id}/messages/{msg_id}/branches",
    response_model=Result[list[MessageResult]],
    summary="查询消息的分支列表",
)
async def get_branches(
    conv_id: int,
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.get_branches(db, conv_id, user.id, msg_id)
    return success(result)


@router.put(
    "/conversations/{conv_id}/branches/{msg_id}",
    response_model=Result[ConversationResult],
    summary="切换当前分支",
)
async def switch_branch(
    conv_id: int,
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.switch_branch(db, conv_id, user.id, msg_id)
    return success(result)


@router.post(
    "/messages/{msg_id}/stop", response_model=Result[MessageResult], summary="停止流式输出"
)
async def stop_message(
    msg_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_conversation_service.stop_message(db, msg_id, user.id)
    return success(result)
