"""
任务策略抽象基类

定义统一的任务执行接口和进度回调协议。
"""

from __future__ import annotations

import abc
from typing import Any, Awaitable, Callable, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_task import SysTask

ProgressCallback = Callable[[int, int], Awaitable[None]]

CancelChecker = Callable[[], Awaitable[bool]]


class TaskStrategy(abc.ABC):
    """任务策略抽象基类"""

    @abc.abstractmethod
    def get_task_types(self) -> List[str]:
        """返回该策略支持的所有任务类型"""

    @abc.abstractmethod
    async def execute(
        self,
        db: AsyncSession,
        sys_task: SysTask,
        params_json: Optional[str],
        progress_callback: ProgressCallback,
        cancel_checker: CancelChecker,
    ) -> Optional[str]:
        """
        执行任务

        Args:
            db: 数据库会话
            sys_task: 任务实体
            params_json: 任务参数（JSON 字符串）
            progress_callback: 进度回调（带频率控制）
            cancel_checker: 取消检测回调

        Returns:
            下载 URL（上传 MinIO 后的预签名 URL），导入类任务可返回结果 JSON 字符串，失败返回 None

        Raises:
            TaskCancelledException: 任务被取消
            BusinessException: 业务参数错误
        """
        ...
