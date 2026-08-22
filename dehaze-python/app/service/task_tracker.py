"""
任务追踪管理器

追踪运行中的后台任务，支持优雅关闭时等待任务完成。
支持多 Worker：本地追踪用于优雅关闭，Redis 追踪用于全局视图。

Redis key 设计：
    {prefix}:{task_id} → hash, 字段: task_type, worker, started_at, metadata
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from redis.asyncio import Redis

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TrackedTask:
    """被追踪的任务"""

    task_id: str
    task: asyncio.Task
    task_type: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskTracker:
    """
    任务追踪管理器

    功能：
    1. 追踪所有运行中的后台任务（本地 + Redis）
    2. 支持优雅关闭时等待本 Worker 的任务完成
    3. 支持广播关闭信号给本 Worker 的所有任务
    4. 支持查询全局任务状态（跨 Worker）
    """

    def __init__(self, shutdown_timeout: float = 30.0):
        """
        初始化任务追踪器

        Args:
            shutdown_timeout: 关闭超时时间（秒）
        """
        self._running_tasks: dict[str, TrackedTask] = {}
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False
        self._shutdown_timeout = shutdown_timeout
        self._redis: Redis | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._worker_id = str(os.getpid())

    @property
    def is_shutting_down(self) -> bool:
        """是否正在关闭"""
        return self._is_shutting_down

    @property
    def running_count(self) -> int:
        """本 Worker 运行中的任务数量"""
        return len(self._running_tasks)

    async def start(self, redis: Redis):
        """启动 Redis 背景状态同步（在 lifespan 中调用）"""
        self._redis = redis
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"TaskTracker Redis 状态同步已启动, worker={self._worker_id}")

    async def stop(self):
        """停止 Redis 背景状态同步（在 lifespan 中调用）"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # 清理本 Worker 在 Redis 中的任务记录
        if self._redis:
            for task_id in list(self._running_tasks.keys()):
                try:
                    await self._redis.delete(f"{settings.TASK_REDIS_KEY_PREFIX}:{task_id}")
                except Exception as e:
                    logger.warning(f"Redis 任务状态清理失败: taskId={task_id}, error={e}")

        logger.info("TaskTracker Redis 状态同步已停止")

    async def register(
        self,
        task_id: str,
        task: asyncio.Task,
        task_type: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        注册任务

        Args:
            task_id: 任务唯一标识
            task: asyncio.Task 实例
            task_type: 任务类型
            metadata: 任务元数据
        """
        if self._is_shutting_down:
            logger.warning(f"服务正在关闭，拒绝新任务注册: taskId={task_id}")
            task.cancel()
            return

        tracked = TrackedTask(
            task_id=task_id,
            task=task,
            task_type=task_type,
            metadata=metadata or {},
        )
        self._running_tasks[task_id] = tracked
        logger.debug(f"任务已注册: taskId={task_id}, type={task_type}")

        # 同步到 Redis
        await self._register_in_redis(task_id, task_type, metadata)

        # 任务完成时同步清理
        task.add_done_callback(lambda _: asyncio.create_task(self._cleanup_task(task_id)))

    async def _register_in_redis(
        self,
        task_id: str,
        task_type: str,
        metadata: dict[str, Any] | None,
    ):
        """将任务状态写入 Redis"""
        if not self._redis:
            return
        try:
            key = f"{settings.TASK_REDIS_KEY_PREFIX}:{task_id}"
            pipe = self._redis.pipeline()
            pipe.hset(
                key,
                mapping={
                    "task_type": task_type,
                    "worker": self._worker_id,
                    "started_at": str(time.time()),
                    "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                },
            )
            pipe.expire(key, settings.TASK_REDIS_TTL)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Redis 任务状态同步失败: taskId={task_id}, error={e}")

    async def _cleanup_task(self, task_id: str):
        """任务完成时清理本地和 Redis 记录"""
        self._running_tasks.pop(task_id, None)
        if self._redis:
            try:
                await self._redis.delete(f"{settings.TASK_REDIS_KEY_PREFIX}:{task_id}")
            except Exception as e:
                logger.warning(f"Redis 任务状态清理失败: taskId={task_id}, error={e}")

    async def unregister(self, task_id: str) -> None:
        """取消注册任务"""
        self._running_tasks.pop(task_id, None)
        if self._redis:
            try:
                await self._redis.delete(f"{settings.TASK_REDIS_KEY_PREFIX}:{task_id}")
            except Exception as e:
                logger.warning(f"Redis 任务状态清理失败: taskId={task_id}, error={e}")
        logger.debug(f"任务已取消注册: taskId={task_id}")

    async def wait_for_completion(
        self,
        timeout: float | None = None,
    ) -> tuple[int, int]:
        """
        等待本 Worker 所有任务完成

        Args:
            timeout: 超时时间（秒），None 使用默认值

        Returns:
            (成功完成的任务数, 超时取消的任务数)
        """
        timeout = timeout or self._shutdown_timeout
        completed = 0
        cancelled = 0

        if not self._running_tasks:
            logger.info("没有运行中的任务，立即返回")
            return completed, cancelled

        tasks_to_wait = list(self._running_tasks.values())
        logger.info(f"等待 {len(tasks_to_wait)} 个任务完成，超时: {timeout}s")

        try:
            done, pending = await asyncio.wait(
                [t.task for t in tasks_to_wait],
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            completed = len(done)

            if pending:
                logger.warning(f"超时，还有 {len(pending)} 个任务未完成，将取消")
                cancelled = len(pending)

                for task in pending:
                    task.cancel()

                # 等待取消完成
                await asyncio.wait(pending, timeout=5.0)

        except Exception as e:
            logger.error(f"等待任务完成时出错: {e}")

        return completed, cancelled

    async def initiate_shutdown(self) -> None:
        """启动关闭流程，拒绝新任务注册"""
        self._is_shutting_down = True
        self._shutdown_event.set()
        logger.info("任务追踪器已进入关闭模式")

    async def cancel_all(self, reason: str = "服务关闭") -> int:
        """取消本 Worker 所有运行中的任务"""
        tasks_to_cancel = list(self._running_tasks.values())

        if not tasks_to_cancel:
            return 0

        logger.warning(f"取消 {len(tasks_to_cancel)} 个任务: {reason}")

        for tracked in tasks_to_cancel:
            tracked.task.cancel()

        await asyncio.gather(
            *[t.task for t in tasks_to_cancel],
            return_exceptions=True,
        )

        return len(tasks_to_cancel)

    async def cancel_task(self, task_id: str) -> bool:
        """取消本 Worker 中指定的运行任务。

        Args:
            task_id: 任务标识（如 pred:{log_id}）

        Returns:
            是否成功找到并取消该任务（不存在或非运行中返回 False）
        """
        tracked = self._running_tasks.get(task_id)
        if not tracked or tracked.task.done():
            return False

        logger.info(f"取消任务: taskId={task_id}, type={tracked.task_type}")
        tracked.task.cancel()
        try:
            await asyncio.wait_for(tracked.task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"任务取消后异常: taskId={task_id}, error={e}")
        await self.unregister(task_id)
        return True

    def get_running_tasks(self) -> list[dict[str, Any]]:
        """获取本 Worker 运行中的任务列表"""
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "created_at": t.created_at.isoformat(),
                "metadata": t.metadata,
                "worker": self._worker_id,
            }
            for t in self._running_tasks.values()
        ]

    async def get_global_running_tasks(self) -> list[dict[str, Any]]:
        """获取全局运行中的任务列表（跨 Worker）"""
        if not self._redis:
            return self.get_running_tasks()

        try:
            tasks = []
            pattern = f"{settings.TASK_REDIS_KEY_PREFIX}:*"
            async for key in self._redis.scan_iter(match=pattern, count=100):
                key_str = key if isinstance(key, str) else key.decode("utf-8")
                task_id = key_str.split(":", 2)[-1]
                data = await self._redis.hgetall(key_str)

                if data:
                    tasks.append(
                        {
                            "task_id": task_id,
                            "task_type": data.get("task_type", "unknown"),
                            "worker": data.get("worker", "unknown"),
                            "started_at": data.get("started_at", ""),
                            "metadata": json.loads(data.get("metadata", "{}")),
                        }
                    )

            return tasks
        except Exception as e:
            logger.warning(f"获取全局任务列表失败: {e}")
            return self.get_running_tasks()

    async def get_global_running_count(self) -> int:
        """获取全局运行中任务数（跨 Worker）"""
        if not self._redis:
            return self.running_count

        try:
            count = 0
            pattern = f"{settings.TASK_REDIS_KEY_PREFIX}:*"
            async for _ in self._redis.scan_iter(match=pattern, count=100):
                count += 1
            return count
        except Exception as e:
            logger.warning(f"获取全局任务数失败: {e}")
            return self.running_count

    async def _heartbeat_loop(self):
        """心跳循环：续期 Redis 中的任务状态"""
        while not self._is_shutting_down:
            try:
                await asyncio.sleep(settings.TASK_HEARTBEAT_INTERVAL)

                if self._redis and self._running_tasks:
                    pipe = self._redis.pipeline()
                    for task_id in self._running_tasks:
                        pipe.expire(
                            f"{settings.TASK_REDIS_KEY_PREFIX}:{task_id}",
                            settings.TASK_REDIS_TTL,
                        )
                    await pipe.execute()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"任务心跳失败: {e}")


# 全局任务追踪器实例
_task_tracker: TaskTracker | None = None


def get_task_tracker() -> TaskTracker:
    """获取全局任务追踪器实例"""
    global _task_tracker
    if _task_tracker is None:
        _task_tracker = TaskTracker()
    return _task_tracker


def init_task_tracker(shutdown_timeout: float = 30.0) -> TaskTracker:
    """
    初始化全局任务追踪器

    Args:
        shutdown_timeout: 关闭超时时间（秒）

    Returns:
        TaskTracker 实例
    """
    global _task_tracker
    _task_tracker = TaskTracker(shutdown_timeout=shutdown_timeout)
    logger.info(f"任务追踪器已初始化，关闭超时: {shutdown_timeout}s")
    return _task_tracker
