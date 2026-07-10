"""
任务追踪管理器

追踪运行中的后台任务，支持优雅关闭时等待任务完成。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

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
    1. 追踪所有运行中的后台任务
    2. 支持优雅关闭时等待任务完成
    3. 支持广播关闭信号给所有任务
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

    @property
    def is_shutting_down(self) -> bool:
        """是否正在关闭"""
        return self._is_shutting_down

    @property
    def running_count(self) -> int:
        """运行中的任务数量"""
        return len(self._running_tasks)

    async def register(
        self,
        task_id: str,
        task: asyncio.Task,
        task_type: str = "default",
        metadata: Optional[dict[str, Any]] = None,
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

        # 任务完成时同步清理，无需 asyncio.Lock（单线程事件循环）
        task.add_done_callback(
            lambda _: self._running_tasks.pop(task_id, None))

    async def unregister(self, task_id: str) -> None:
        """
        取消注册任务

        Args:
            task_id: 任务唯一标识
        """
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]
            logger.debug(f"任务已取消注册: taskId={task_id}")

    async def wait_for_completion(
        self,
        timeout: Optional[float] = None,
    ) -> tuple[int, int]:
        """
        等待所有任务完成

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
        """
        启动关闭流程

        设置关闭标志，拒绝新任务注册
        """
        self._is_shutting_down = True
        self._shutdown_event.set()
        logger.info("任务追踪器已进入关闭模式")

    async def cancel_all(self, reason: str = "服务关闭") -> int:
        """
        取消所有运行中的任务

        Args:
            reason: 取消原因

        Returns:
            取消的任务数量
        """
        tasks_to_cancel = list(self._running_tasks.values())

        if not tasks_to_cancel:
            return 0

        logger.warning(f"取消 {len(tasks_to_cancel)} 个任务: {reason}")

        for tracked in tasks_to_cancel:
            tracked.task.cancel()

        # 等待取消完成
        await asyncio.gather(
            *[t.task for t in tasks_to_cancel],
            return_exceptions=True,
        )

        return len(tasks_to_cancel)

    def get_running_tasks(self) -> list[dict[str, Any]]:
        """
        获取运行中的任务列表

        Returns:
            任务信息列表
        """
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "created_at": t.created_at.isoformat(),
                "metadata": t.metadata,
            }
            for t in self._running_tasks.values()
        ]


# 全局任务追踪器实例
_task_tracker: Optional[TaskTracker] = None


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
