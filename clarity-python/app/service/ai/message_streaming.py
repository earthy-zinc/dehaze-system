"""SSE 流式消息生成：message/regenerate 共用，消除两 service 间循环依赖。

send_message / edit_message / regenerate_message 复用同一套事件流触发链路，
依赖经参数注入，使 ai_message_service 与 ai_conversation_service 各自调用而不互相 import。

执行与连接解耦：本模块只负责流侧生命周期（起流、起后台推理、结束流）。
连接断开或空闲超时都不取消推理——推理跑完照常落库计费，用户从消息历史
或重连重放拿到完整回复。失败态由推理侧落库，error 事件由本模块推一次。
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# 消息 → 流会话反查键（TTL 对齐流缓存窗口）：停止端点据此定位 send 路径消息的流
_MSG_STREAM_KEY = "ai:msg:stream:{msg_id}"
_MSG_STREAM_TTL = 300

# 幂等完成态保留时长（供重试查询结果）
_IDEM_RESULT_TTL = 300

# pending 续期间隔 = 初始 TTL / 3：推理（含空闲超时后继续跑完的部分）可能长于
# 初始 TTL，不续期会让 pending 中途过期、同 key 重试被判定为首次发送而重复落库
_IDEM_RENEW_DIVISOR = 3

# 完成态/释放写入必须与"本轮抢占标识"比对后原子执行：pending 若已过期并被新一轮
# 请求重新抢占，上一轮的推理结果不得覆盖新一轮的 pending（否则第二次发送被误判完成）
_COMPLETE_IDEM_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
    return 1
end
return 0
"""

_RELEASE_IDEM_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('DEL', KEYS[1])
    return 1
end
return 0
"""

# 后台推理任务引用：流关闭后任务仍须跑完落库，与推理侧一致显式持有
_pending_tasks: set[asyncio.Task] = set()


async def _renew_pending(redis, idem_key: str, token: str, ttl: int) -> None:
    """周期性续期本轮 pending 键；键已被新一轮抢占（值不是本轮标识）即停止。"""
    while True:
        await asyncio.sleep(max(ttl // _IDEM_RENEW_DIVISOR, 1))
        if await redis.get(idem_key) != token:
            return
        await redis.expire(idem_key, ttl)


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
    本任务生命周期超出请求作用域，故不接请求 session：失败态落库由推理侧用自己的
    session 完成，此处负责幂等键生命周期（续期 + 条件写入）、推一次 error 事件、结束流。
    """
    redis = await get_redis_client()
    # 用本轮唯一标识接管 pending（保留原 TTL）：后续完成/释放都按该标识做 CAS
    token = f"pending:{stream_session_id}"
    pending_ttl = await redis.ttl(idem_key)
    await redis.set(idem_key, token, ex=pending_ttl if pending_ttl > 0 else None)
    renewer = (
        asyncio.create_task(_renew_pending(redis, idem_key, token, pending_ttl))
        if pending_ttl > 0
        else None
    )
    try:
        result = await reasoning_service.run(
            conv_id=conv_id,
            user_id=user_id,
            msg_id=assistant_msg_id,
            model_id=model,
            stream_session_id=stream_session_id,
        )
        if (result or {}).get("stop_reason") == "canceled":
            # 用户停止：清除幂等键允许重发；否则按完成态写入幂等结果
            await redis.eval(_RELEASE_IDEM_LUA, 1, idem_key, token)
        else:
            await redis.eval(
                _COMPLETE_IDEM_LUA,
                1,
                idem_key,
                token,
                json.dumps({"messageId": assistant_msg_id, "status": 2}),
                _IDEM_RESULT_TTL,
            )
    except Exception as e:
        logger.error("AI 推理失败: %s", e, exc_info=True)
        await redis.eval(_RELEASE_IDEM_LUA, 1, idem_key, token)
        # 向客户端推送 SSE error 事件：让前端区分"网络断开"与"后端推理失败"并展示原因
        # （前端 onError 收到 {code, message} 后将该消息置为失败态）
        try:
            await sse_emitter_manager.send_error(stream_session_id, e)
        except Exception:
            logger.warning("推送 SSE error 事件失败: stream=%s", stream_session_id, exc_info=True)
    finally:
        if renewer is not None:
            renewer.cancel()
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
    # 登记消息→流会话反查键：send 路径 assistant 消息 task_id 不存流会话 ID，
    # 停止端点经该键定位流（TTL 对齐流缓存重连窗口，过期后流已终结无需停止）
    redis = await get_redis_client()
    await redis.set(
        _MSG_STREAM_KEY.format(msg_id=assistant_msg_id), stream_session_id, ex=_MSG_STREAM_TTL
    )
    await sse_emitter_manager.send_event(
        stream_session_id,
        "message.start",
        {
            "messageId": assistant_msg_id,
            "conversationId": conv_id,
            "model": model,
            # 断线重连（Last-Event-ID + stream/{stream_session_id} 端点）必需
            "streamSessionId": stream_session_id,
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
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)
    # 生成器结束即关闭连接，不 await 后台任务：等待会让空闲超时后的连接继续挂起
    # （期间无心跳无事件），代理超时保护失效。推理与连接解耦，任务跑完自行落库计费。
    async for chunk in sse_emitter_manager.create_stream(conv_id, stream_session_id):
        yield chunk
