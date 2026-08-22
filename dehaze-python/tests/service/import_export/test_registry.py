from __future__ import annotations

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.import_export.models import ExportFieldConfig, ImportFieldConfig
from app.service.import_export.registry import (
    ExportHandler,
    ExportHandlerRegistry,
    ImportHandler,
    ImportHandlerRegistry,
)


class _StubExportHandler(ExportHandler):
    def __init__(self, module: str):
        self._module = module

    def get_module(self) -> str:
        return self._module

    async def estimate_count(self, db, query_params: dict) -> int:
        return 0

    async def export(self, db, ctx, output, progress_cb, cancel_cb) -> None:
        return None

    def get_field_configs(self) -> list[ExportFieldConfig]:
        return [ExportFieldConfig(field="f", label="F", order=1)]


class _StubImportHandler(ImportHandler):
    def __init__(self, module: str):
        self._module = module

    def get_module(self) -> str:
        return self._module

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [ImportFieldConfig(field="f", label="F")]

    async def import_batch(self, db, rows, options, progress_cb, cancel_cb):
        return None


class TestExportHandlerRegistry:
    def test_register_and_get_handler(self):
        registry = ExportHandlerRegistry()
        handler = _StubExportHandler("user")
        registry.register(handler)

        assert registry.get_handler("user") is handler
        assert registry.has_module("user") is True

    def test_get_handler_not_registered_raises(self):
        registry = ExportHandlerRegistry()
        with pytest.raises(BusinessException) as exc_info:
            registry.get_handler("unknown")
        assert exc_info.value.code == ResultCode.MODULE_IMPORT_NOT_SUPPORTED

    def test_has_module_returns_false_for_unregistered(self):
        registry = ExportHandlerRegistry()
        assert registry.has_module("role") is False

    def test_register_multiple_handlers(self):
        registry = ExportHandlerRegistry()
        user_h = _StubExportHandler("user")
        role_h = _StubExportHandler("role")
        registry.register(user_h)
        registry.register(role_h)

        assert registry.get_handler("user") is user_h
        assert registry.get_handler("role") is role_h
        assert registry.has_module("user")
        assert registry.has_module("role")


class TestImportHandlerRegistry:
    def test_register_and_get_handler(self):
        registry = ImportHandlerRegistry()
        handler = _StubImportHandler("user")
        registry.register(handler)

        assert registry.get_handler("user") is handler
        assert registry.has_module("user") is True

    def test_get_handler_not_registered_raises(self):
        registry = ImportHandlerRegistry()
        with pytest.raises(BusinessException) as exc_info:
            registry.get_handler("unknown")
        assert exc_info.value.code == ResultCode.MODULE_IMPORT_NOT_SUPPORTED

    def test_has_module_returns_false_for_unregistered(self):
        registry = ImportHandlerRegistry()
        assert registry.has_module("role") is False


class TestExportHandlerFilterFields:
    def test_filter_fields_returns_all_visible_when_no_selected(self):
        handler = _StubExportHandler("user")
        handler.get_field_configs = lambda: [
            ExportFieldConfig(field="a", label="A", order=1),
            ExportFieldConfig(field="b", label="B", order=2, hidden=True),
            ExportFieldConfig(field="c", label="C", order=3),
        ]
        result = handler.filter_fields(None)
        assert [f.field for f in result] == ["a", "c"]

    def test_filter_fields_filters_by_selected(self):
        handler = _StubExportHandler("user")
        handler.get_field_configs = lambda: [
            ExportFieldConfig(field="a", label="A", order=1),
            ExportFieldConfig(field="b", label="B", order=2),
            ExportFieldConfig(field="c", label="C", order=3),
        ]
        result = handler.filter_fields(["a", "c"])
        assert [f.field for f in result] == ["a", "c"]

    def test_filter_fields_excludes_hidden_even_if_selected(self):
        handler = _StubExportHandler("user")
        handler.get_field_configs = lambda: [
            ExportFieldConfig(field="a", label="A", order=1),
            ExportFieldConfig(field="b", label="B", order=2, hidden=True),
        ]
        result = handler.filter_fields(["a", "b"])
        assert [f.field for f in result] == ["a"]
