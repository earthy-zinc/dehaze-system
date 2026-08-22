"""
导入导出处理器注册表与抽象接口
"""

from __future__ import annotations

import abc
import io
from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.import_export.models import (
    ExportContext,
    ExportFieldConfig,
    ImportFieldConfig,
    ImportOptions,
    ImportResult,
)


class ExportHandler(abc.ABC):
    @abc.abstractmethod
    def get_module(self) -> str:
        """模块标识"""

    @abc.abstractmethod
    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        """预估导出行数"""

    @abc.abstractmethod
    async def export(
        self,
        db: AsyncSession,
        ctx: ExportContext,
        output: io.BytesIO,
        progress_cb,
        cancel_cb,
    ) -> None:
        """执行导出，将结果写入 output（BytesIO）"""

    @abc.abstractmethod
    def get_field_configs(self) -> list[ExportFieldConfig]:
        """字段配置"""

    def use_direct_export(self) -> bool:
        """是否直接导出（如 ZIP 包），不使用 Excel/CSV 文件生成器"""
        return False

    def filter_fields(self, selected: list[str] | None) -> list[ExportFieldConfig]:
        all_fields = self.get_field_configs()
        if not selected:
            return [f for f in all_fields if not f.hidden]
        sel = set(selected)
        return [f for f in all_fields if f.field in sel and not f.hidden]


class ImportHandler(abc.ABC):
    @abc.abstractmethod
    def get_module(self) -> str:
        """模块标识"""

    @abc.abstractmethod
    def get_field_configs(self) -> list[ImportFieldConfig]:
        """字段配置"""

    @abc.abstractmethod
    async def import_batch(
        self,
        db: AsyncSession,
        rows: list[dict],
        options: ImportOptions,
        progress_cb,
        cancel_cb,
    ) -> ImportResult:
        """执行批量导入"""

    def get_template_sample_data(self) -> list[dict]:
        return []


T = TypeVar("T")


class _HandlerRegistry(Generic[T]):
    """按模块名注册/查找处理器，未注册模块抛出业务异常"""

    def __init__(self, kind_label: str) -> None:
        self._kind_label = kind_label
        self._handlers: dict[str, T] = {}

    def register(self, handler: T) -> None:
        self._handlers[handler.get_module()] = handler

    def get_handler(self, module: str) -> T:
        handler = self._handlers.get(module)
        if handler is None:
            raise BusinessException(
                ResultCode.MODULE_IMPORT_NOT_SUPPORTED,
                f"模块 {module} 不支持{self._kind_label}",
            )
        return handler

    def has_module(self, module: str) -> bool:
        return module in self._handlers


class ExportHandlerRegistry(_HandlerRegistry[ExportHandler]):
    def __init__(self) -> None:
        super().__init__("导出")


class ImportHandlerRegistry(_HandlerRegistry[ImportHandler]):
    def __init__(self) -> None:
        super().__init__("导入")


export_handler_registry = ExportHandlerRegistry()
import_handler_registry = ImportHandlerRegistry()
