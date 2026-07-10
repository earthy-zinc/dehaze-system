"""
任务策略工厂

自动注册策略实现类，根据 task_type 返回对应的策略实例。
"""

from __future__ import annotations

import logging
from typing import Dict

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.enum.task_enum import TaskType
from app.service.task.strategy import TaskStrategy

logger = logging.getLogger(__name__)

# 策略注册表
_registry: Dict[str, TaskStrategy] = {}


def register_strategy(task_type: str, strategy: TaskStrategy) -> None:
    """注册任务策略"""
    _registry[task_type] = strategy
    logger.debug(f"注册任务策略: {task_type} -> {type(strategy).__name__}")


class TaskStrategyFactory:
    """任务策略工厂"""

    @staticmethod
    def get_strategy(task_type: str) -> TaskStrategy:
        """
        根据任务类型获取对应的策略实例

        Args:
            task_type: 任务类型

        Returns:
            策略实例

        Raises:
            BusinessException: 不支持的任务类型
        """
        strategy = _registry.get(task_type)
        if strategy is None:
            raise BusinessException(
                ResultCode.TASK_TYPE_UNSUPPORTED,
                f"不支持的任务类型: {task_type}",
            )
        return strategy


def _init_strategies() -> None:
    """初始化并注册所有策略（模块加载时自动执行）"""
    from app.service.task.strategies.dataset_export import DatasetExportStrategy
    from app.service.task.strategies.item_download import ItemDownloadStrategy
    from app.service.task.strategies.batch_download import BatchDownloadStrategy
    from app.service.task.strategies.custom_export import CustomExportStrategy

    register_strategy(TaskType.DATASET_EXPORT.value, DatasetExportStrategy())
    register_strategy(TaskType.ITEM_DOWNLOAD.value, ItemDownloadStrategy())
    register_strategy(TaskType.BATCH_DOWNLOAD.value, BatchDownloadStrategy())
    register_strategy(TaskType.CUSTOM_EXPORT.value, CustomExportStrategy())


_init_strategies()
