"""SSE 流式消息生成：message/regenerate 共用，消除两 service 间循环依赖。

send_message / edit_message / regenerate_message 复用同一套事件流触发链路，
依赖经参数注入，使 ai_message_service 与 ai_conversation_service 各自调用而不互相 import。
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


async def run_reasoning(
    *,
    reasoning_service,
    get_redis_client,
    sse_emitter_manager,
    conv_id,
    user_id,
    model,
    assistant_msg_id,
    stream_session_id,
    idem_key,
) -> None:
    """后台任务：调用 ReasoningService 推理，成功后写入幂等键。

    上下文由 reasoning_service.run 内部经 build_context 一次性组装，此处不再预热。
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
    finally:
        await sse_emitter_manager.stop_stream(stream_session_id)


async def stream_generator(
    *,
    sse_emitter_manager,
    reasoning_service,
    get_redis_client,
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
