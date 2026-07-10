"""
文件事件模块

提供进程内事件总线，用于文件服务与其他业务模块的解耦。
文件上传/删除后通过事件通知其他模块（如数据集模块）。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)


# ==================== 事件定义 ====================


@dataclass
class FileEvent:
    """文件事件基类"""
    file_id: int
    filename: str
    object_name: str
    md5: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FileCreatedEvent(FileEvent):
    """文件创建事件"""
    size_bytes: int = 0


@dataclass
class FileDeletedEvent(FileEvent):
    """文件删除事件"""
    pass


# ==================== 事件总线 ====================


class FileEventBus:
    """
    进程内文件事件总线

    支持订阅和发布文件相关事件，用于解耦文件服务与业务模块。
    所有 handler 同步执行，不应包含耗时操作（耗时逻辑应发消息到队列）。
    """

    def __init__(self):
        self._handlers: dict[type, list[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable) -> None:
        """
        订阅事件

        Args:
            event_type: 事件类型（如 FileCreatedEvent）
            handler: 事件处理函数，签名 (event: FileEvent) -> None
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"已注册事件处理器: {event_type.__name__} -> {handler.__name__}")

    def publish(self, event: FileEvent) -> None:
        """
        发布事件

        依次调用所有已注册的 handler，单个 handler 异常不影响其他 handler。

        Args:
            event: 事件实例
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"事件处理器执行失败: {handler.__name__} "
                    f"[{event_type.__name__}, file_id={event.file_id}]: {e}",
                    exc_info=True,
                )

    def clear(self) -> None:
        """清除所有事件订阅（主要用于测试）"""
        self._handlers.clear()


# 全局事件总线单例
file_event_bus = FileEventBus()
