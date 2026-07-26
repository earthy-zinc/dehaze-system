"""
导入导出处理器注册表与抽象接口
"""
from __future__ import annotations

import abc
import io
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.import_export.models import (ExportContext, ExportFieldConfig,
                                              ImportFieldConfig, ImportOptions,
                                              ImportResult)


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

    def filter_fields(self, selected: Optional[list[str]]) -> list[ExportFieldConfig]:
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


class ExportHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ExportHandler] = {}

    def register(self, handler: ExportHandler) -> None:
        self._handlers[handler.get_module()] = handler

    def get_handler(self, module: str) -> ExportHandler:
        handler = self._handlers.get(module)
        if handler is None:
            raise BusinessException(
                ResultCode.MODULE_IMPORT_NOT_SUPPORTED,
                f"模块 {module} 不支持导出",
            )
        return handler

    def has_module(self, module: str) -> bool:
        return module in self._handlers


class ImportHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ImportHandler] = {}

    def register(self, handler: ImportHandler) -> None:
        self._handlers[handler.get_module()] = handler

    def get_handler(self, module: str) -> ImportHandler:
        handler = self._handlers.get(module)
        if handler is None:
            raise BusinessException(
                ResultCode.MODULE_IMPORT_NOT_SUPPORTED,
                f"模块 {module} 不支持导入",
            )
        return handler

    def has_module(self, module: str) -> bool:
        return module in self._handlers


export_handler_registry = ExportHandlerRegistry()
import_handler_registry = ImportHandlerRegistry()


def register_export_handler(handler: ExportHandler) -> None:
    export_handler_registry.register(handler)


def register_import_handler(handler: ImportHandler) -> None:
    import_handler_registry.register(handler)
