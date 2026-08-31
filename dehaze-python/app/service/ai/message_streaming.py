"""SSE 流式消息生成：message/regenerate 共用，消除两 service 间循环依赖。

send_message / edit_message / regenerate_message 复用同一套事件流触发链路，
依赖经参数注入，使 ai_message_service 与 ai_conversation_service 各自调用而不互相 import。
"""

import asyncio
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_message_repository import ai_message_repository

logger = logging.getLogger(__name__)


def _to_error_payload(exc: Exception) -> dict:
    """将推理异常映射为 SSE error 事件载荷（{code, message}）。

    BusinessException 透出业务码与消息（如配额拒绝 A0503、LLM 调用失败 A0600、
    模型不可用 A0601）；未知异常统一按 LLM 调用失败（A0600）呈现，避免泄露内部细节。
    """
    if isinstance(exc, BusinessException):
        code = getattr(exc.code, "code", None) or ResultCode.AI_LLM_CALL_FAILED.code
        message = exc.message or ResultCode.AI_LLM_CALL_FAILED.msg
        return {"code": code, "message": message}
    return {
        "code": ResultCode.AI_LLM_CALL_FAILED.code,
        "message": ResultCode.AI_LLM_CALL_FAILED.msg,
    }


async def run_reasoning(
    *,
    reasoning_service,
    get_redis_client,
    sse_emitter_manager,
    db: AsyncSession,
    conv_id,
    user_id,
    model,
    assistant_msg_id,
    stream_session_id,
    idem_key,
) -> None:
    """后台任务：调用 ReasoningService 推理，成功后写入幂等键。

    上下文由 reasoning_service.run 内部经 build_context 一次性组装，此处不再预热。
    db 用于推理失败时把助手消息落库为失败态（避免前端重进会话误显示"生成中"）。
    """
    redis = await get_redis_client()
    try:
        await reasoning_service.run(
            conv_id=conv_id,
            user_id=user_id,
            msg_id=assistant_msg_id,
            model_id=model,
            stream_session_id=stream_session_id,
        )
        await redis.set(idem_key, json.dumps({"messageId": assistant_msg_id, "status": 2}), ex=300)
    except Exception as e:
        logger.error("AI 推理失败: %s", e, exc_info=True)
        await redis.delete(idem_key)
        # 落库失败态：推理失败后消息 status 置 3 并记错误，防止前端重进会话显示"生成中"
        try:
            await ai_message_repository.update_status(
                db, assistant_msg_id, 3, error=str(e)[:500]
            )
        except Exception:
            logger.warning(
                "标记消息失败态失败: msg_id=%s", assistant_msg_id, exc_info=True
            )
        # 向客户端推送 SSE error 事件：让前端区分"网络断开"与"后端推理失败"并展示真实原因
        # （前端 onError 收到 {code, message} 后将该消息置为失败态）
        try:
            await sse_emitter_manager.send_event(
                stream_session_id, "error", _to_error_payload(e)
            )
        except Exception:
            logger.warning(
                "推送 SSE error 事件失败: stream=%s", stream_session_id, exc_info=True
            )
    finally:
        await sse_emitter_manager.stop_stream(stream_session_id)


async def stream_generator(
    *,
    sse_emitter_manager,
    reasoning_service,
    get_redis_client,
    db: AsyncSession,
    conv_id,
    user_id,
    model,
    assistant_msg_id,
    stream_session_id,
    idem_key,
):
    # 先预注册事件队列，再推 message.start：否则 send_event 时队列未建立，
    # 事件仅写入 Redis 缓存而无法经活跃连接送达客户端（message.start 丢失）。
    await sse_emitter_manager.register_stream(stream_session_id)
    await sse_emitter_manager.send_event(
        stream_session_id,
        "message.start",
        {
            "messageId": assistant_msg_id,
            "conversationId": conv_id,
            "model": model,
        },
    )
    task = asyncio.create_task(
        run_reasoning(
            reasoning_service=reasoning_service,
            get_redis_client=get_redis_client,
            sse_emitter_manager=sse_emitter_manager,
            db=db,
            conv_id=conv_id,
            user_id=user_id,
            model=model,
            assistant_msg_id=assistant_msg_id,
            stream_session_id=stream_session_id,
            idem_key=idem_key,
        )
    )
    try:
        async for chunk in sse_emitter_manager.create_stream(conv_id, stream_session_id):
            yield chunk
    finally:
        # 客户端断连时也等待后台任务完成，确保 assistant 消息正常落库
        if not task.done():
            await asyncio.shield(task)
