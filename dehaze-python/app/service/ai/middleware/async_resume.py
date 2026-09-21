"""异步任务等待与自动恢复（async_wait 中断链路）

设计文档 §8.4 异步任务等待与回调：批量处理等异步工具提交后台任务后进入
interrupt(type=async_wait)，任务完成后由回调自动恢复推理，而非用户手动 resume。

职责：
- submit_batch_task：把批量处理封装为 asyncio 后台任务，登记 task_id→恢复信息
  反查键（ai:async_task:{task_id}），任务完成（成功/失败）时触发自动 resume。
- notify_task_completed：prediction 回调统一入口，按 task_id 反查后调用
  reasoning_service.resume，把任务结果作为 resume 载荷注入工具中断点。
"""

import asyncio
import json
import logging

from app.dependencies.redis import get_redis_client
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager

logger = logging.getLogger(__name__)

# 后台任务引用，防止被垃圾回收
_pending_tasks: set[asyncio.Task] = set()

# task_id → asyncio.Task 注册表：用户停止（reasoning_service.stop）时据此取消批量任务
_running_tasks: dict[str, asyncio.Task] = {}

# 任务反查键 TTL：与中断点一致（24h），确保断线后任务完成仍可恢复
_ASYNC_TASK_TTL = 86400

# 会话锁被占用时的重试策略：锁持有方（用户侧 resume/新消息 run）通常很快释放，
# 放弃恢复会让中断点挂起到 TTL 过期，故回滚反查键后延迟重试
_RESUME_RETRY_MAX = 3
_RESUME_RETRY_DELAY = 2.0


def _schedule_retry(task_id: str, task_result: dict | None) -> None:
    """延迟重试自动恢复（锁竞争场景）。"""

    async def _retry() -> None:
        await asyncio.sleep(_RESUME_RETRY_DELAY)
        await notify_task_completed(task_id, task_result)

    task = asyncio.get_running_loop().create_task(_retry())
    _pending_tasks.add(task)
    task.add_done_callback(_pending_tasks.discard)


def _task_key(task_id: str) -> str:
    return f"ai:async_task:{task_id}"


async def _save_task_mapping(task_id: str, mapping: dict) -> None:
    """登记 task_id → 恢复信息反查键，供任务完成回调恢复推理。"""
    try:
        redis = await get_redis_client()
        await redis.set(_task_key(task_id), json.dumps(mapping), ex=_ASYNC_TASK_TTL)
    except Exception:
        logger.warning("异步任务反查键写入失败: task_id=%s", task_id, exc_info=True)


async def _load_task_mapping(task_id: str) -> dict | None:
    """按 task_id 反查恢复信息。"""
    try:
        redis = await get_redis_client()
        raw = await redis.get(_task_key(task_id))
        return json.loads(raw) if raw else None
    except Exception:
        logger.warning("异步任务反查键读取失败: task_id=%s", task_id, exc_info=True)
        return None


async def _clear_task_mapping(task_id: str) -> None:
    try:
        redis = await get_redis_client()
        await redis.delete(_task_key(task_id))
    except Exception:
        # 与 _save/_load_task_mapping 同模式：反查键清理为尽力而为，但失败必须可见。
        # 清理失败会残留反查键（最长 TTL 24h），可能令已取消任务被回调重复恢复。
        logger.warning("异步任务反查键清理失败: task_id=%s", task_id, exc_info=True)


def submit_batch_task(
    *,
    conv_id: int,
    msg_id: int,
    user_id: int,
    image_urls: list[str],
    algorithm_id: int,
    stream_session_id: str,
    thread_id: str,
) -> str:
    """提交批量处理为后台任务，返回可关联的 task_id（供中断数据与消息落库）。

    任务完成后回调 _on_batch_done：成功/失败统一调用 reasoning_service.resume，
    把处理摘要作为 resume 载荷注入工具中断点，图自动从 async_wait 处继续。
    """
    task_id = f"batch:{thread_id}:{int(asyncio.get_running_loop().time() * 1000)}"

    # 登记反查键：回调据此恢复中断推理
    mapping = {
        "thread_id": thread_id,
        "conv_id": conv_id,
        "msg_id": msg_id,
        "user_id": user_id,
        "stream_session_id": stream_session_id,
    }
    # 任务登记需在独立事件循环中执行（当前在工具节点内，get_redis_client 可用）
    task = asyncio.get_running_loop().create_task(
        _schedule_run(task_id, mapping, image_urls, algorithm_id, stream_session_id)
    )
    _pending_tasks.add(task)
    _running_tasks[task_id] = task
    task.add_done_callback(_pending_tasks.discard)
    task.add_done_callback(lambda _t: _running_tasks.pop(task_id, None))
    return task_id


async def cancel_task(task_id: str) -> None:
    """取消异步批量任务（用户停止推理时调用）。

    先清反查键再取消：任务 finally 中的完成回调因反查键缺失而静默跳过，
    不会对已取消的消息发起自动恢复。
    """
    task = _running_tasks.pop(task_id, None)
    await _clear_task_mapping(task_id)
    if task and not task.done():
        task.cancel()


async def _schedule_run(task_id, mapping, image_urls, algorithm_id, stream_session_id) -> None:
    from app.service.ai.service.batch_process_service import process_batch

    # 先登记反查键，再执行任务（任务极快完成时确保回调能反查到）
    await _save_task_mapping(task_id, mapping)
    result = None
    try:
        result = await process_batch(
            mapping["conv_id"],
            mapping["msg_id"],
            mapping["user_id"],
            image_urls,
            algorithm_id,
            stream_session_id,
        )
    except Exception as e:
        logger.error("批量处理后台任务失败: task_id=%s, error=%s", task_id, e, exc_info=True)
    finally:
        await notify_task_completed(task_id, result)


async def notify_task_completed(task_id: str, task_result: dict | None) -> None:
    """prediction 回调入口：按 task_id 反查并自动恢复推理。

    任务结果以 resume 载荷传入，工具中断点 interrupt() 的返回值即 task_result，
    图从中断处继续并把批量摘要注入上下文。

    恢复前获取会话并发锁，避免与用户手动 resume（async_wait 兜底路径）或恢复期间
    新消息 run 并发导致同 thread 双跑 checkpoint 冲突；锁被占用时回滚反查键并延迟
    重试（重试耗尽才交回用户侧），避免清键后无人恢复导致中断点挂起到 TTL 过期。
    """
    mapping = await _load_task_mapping(task_id)
    if not mapping:
        # 常规路径：任务已被取消（用户停止）或已恢复过
        logger.info("异步任务无可恢复中断点，忽略回调: task_id=%s", task_id)
        return
    conv_id = mapping.get("conv_id", 0)
    # 恢复前先清除反查键，避免重复恢复；中断点由 resume 流程清理
    await _clear_task_mapping(task_id)
    if not await sse_emitter_manager.acquire_lock(conv_id):
        attempt = int(mapping.get("resume_retry") or 0) + 1
        if attempt > _RESUME_RETRY_MAX:
            logger.warning("异步任务回调重试耗尽，交由用户侧恢复: task_id=%s", task_id)
            return
        logger.info("异步任务回调时会话锁被占用，延迟重试(%s): task_id=%s", attempt, task_id)
        await _save_task_mapping(task_id, {**mapping, "resume_retry": attempt})
        _schedule_retry(task_id, task_result)
        return
    try:
        from app.service.ai.middleware.interrupt_handler import interrupt_handler
        from app.service.ai.service.reasoning_service import reasoning_service

        # 中断点已清理（用户已停止/手动恢复过）：静默跳过，不对取消态消息续流
        if not await interrupt_handler.get_interrupt(mapping.get("thread_id", "")):
            logger.info("中断点已清理，跳过异步任务自动恢复: task_id=%s", task_id)
            return

        resume_data = {"async_task": task_result or {"failed": 1}}
        await reasoning_service.resume(
            conv_id=conv_id,
            user_id=mapping.get("user_id", 0),
            msg_id=mapping.get("msg_id", 0),
            resume_data=resume_data,
        )
    except Exception as e:
        logger.error("异步任务自动恢复失败: task_id=%s, error=%s", task_id, e, exc_info=True)
        stream_session_id = mapping.get("stream_session_id", "")
        if stream_session_id:
            await sse_emitter_manager.send_error(stream_session_id, e)
    finally:
        await sse_emitter_manager.release_lock(conv_id)
