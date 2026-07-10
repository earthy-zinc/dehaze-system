"""
任务策略抽象基类

定义统一的任务执行接口和进度回调协议。
"""

from __future__ import annotations

import abc
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_task import SysTask

# 进度回调签名：(processed_files, total_files) -> None
ProgressCallback = Callable[[int, int], Awaitable[None]]

# 取消检测签名：() -> bool
CancelChecker = Callable[[], Awaitable[bool]]


class TaskStrategy(abc.ABC):
    """
    任务策略抽象基类

    每种任务类型实现一个具体策略，注册到 TaskStrategyFactory 后，
    执行器无需 if/elif 判断类型。
    """

    @abc.abstractmethod
    async def execute(
        self,
        db: AsyncSession,
        sys_task: SysTask,
        target_id: Optional[int],
        target_ids: Optional[List[int]],
        options: Dict[str, Any],
        progress_callback: ProgressCallback,
        cancel_checker: CancelChecker,
    ) -> Optional[str]:
        """
        执行任务

        Args:
            db: 数据库会话
            sys_task: 任务实体
            target_id: 单个目标 ID
            target_ids: 批量目标 ID 列表
            options: 导出选项
            progress_callback: 进度回调（带频率控制）
            cancel_checker: 取消检测回调

        Returns:
            下载 URL（上传 MinIO 后的预签名 URL），失败返回 None

        Raises:
            TaskCancelledException: 任务被取消
            BusinessException: 业务参数错误
        """
        ...
