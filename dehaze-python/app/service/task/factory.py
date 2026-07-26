"""
任务策略工厂

自动注册策略实现类，根据 task_type 返回对应的策略实例。
支持一个策略类处理多个 taskType（通过 get_task_types() 返回列表）。
"""

from __future__ import annotations

import logging
from typing import Dict

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.task.strategy import TaskStrategy

logger = logging.getLogger(__name__)

_registry: Dict[str, TaskStrategy] = {}


def register_strategy(task_type: str, strategy: TaskStrategy) -> None:
    _registry[task_type] = strategy
    logger.debug(f"注册任务策略: {task_type} -> {type(strategy).__name__}")


def register_strategy_instance(strategy: TaskStrategy) -> None:
    for task_type in strategy.get_task_types():
        register_strategy(task_type, strategy)


class TaskStrategyFactory:
    """任务策略工厂"""

    @staticmethod
    def get_strategy(task_type: str) -> TaskStrategy:
        strategy = _registry.get(task_type)
        if strategy is None:
            raise BusinessException(
                ResultCode.TASK_TYPE_UNSUPPORTED,
                f"不支持的任务类型: {task_type}",
            )
        return strategy


def _init_strategies() -> None:
    from app.service.task.strategies.generic_export import GenericExportStrategy
    from app.service.task.strategies.generic_import import GenericImportStrategy

    register_strategy_instance(GenericExportStrategy())
    register_strategy_instance(GenericImportStrategy())


_init_strategies()
