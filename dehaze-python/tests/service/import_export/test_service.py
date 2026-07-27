"""
通用导入导出服务单元测试
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from app.core.code import ResultCode
from app.core.constants import MAX_ROWS, MAX_IMPORT_FILE_SIZE, SYNC_THRESHOLD
from app.core.exceptions import BusinessException
from app.models.enum.task_enum import TaskStatus
from app.service.import_export.models import (ImportError, ImportOptions,
                                              ImportResult)


def _make_upload_file(filename: str, content: bytes, content_type: str = "text/csv") -> UploadFile:
    file = UploadFile(filename=filename, file=io.BytesIO(content))
    file.headers = {"content-type": content_type}
    return file


def _valid_csv_bytes(rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    buf.write("\ufeff")
    for row in rows:
        buf.write(",".join(row) + "\n")
    return buf.getvalue().encode("utf-8")


class _AsyncExportHandler:
    def __init__(self, module: str, count: int):
        self._module = module
        self._count = count

    def get_module(self) -> str:
        return self._module

    async def estimate_count(self, db, query_params: dict) -> int:
        return self._count

    async def export(self, db, ctx, output, progress_cb, cancel_cb) -> None:
        output.write(b"fake-excel-content")

    def get_field_configs(self):
        from app.service.import_export.models import ExportFieldConfig
        return [ExportFieldConfig(field="username", label="用户名", order=1)]

    def use_direct_export(self) -> bool:
        return False

    def filter_fields(self, selected):
        return self.get_field_configs()


class _AsyncImportHandler:
    def __init__(self, module: str, result: ImportResult = None):
        self._module = module
        self._result = result or ImportResult(total_rows=1, success_count=1)

    def get_module(self) -> str:
        return self._module

    def get_field_configs(self):
        from app.service.import_export.models import ImportFieldConfig
        return [ImportFieldConfig(field="username", label="用户名", required=True)]

    async def import_batch(self, db, rows, options, progress_cb, cancel_cb) -> ImportResult:
        return self._result

    def get_template_sample_data(self):
        return [{"username": "zhangsan"}]


@pytest.fixture
def export_handler_registry():
    registry = MagicMock()
    registry.get_handler = MagicMock(side_effect=lambda m: (_ for _ in ()).throw(
        BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED, f"模块 {m} 不支持导出")
    ))
    return registry


@pytest.fixture
def import_handler_registry():
    registry = MagicMock()
    registry.get_handler = MagicMock(side_effect=lambda m: (_ for _ in ()).throw(
        BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED, f"模块 {m} 不支持导入")
    ))
    return registry


class TestExport:
    @pytest.mark.asyncio
    async def test_export_count_exceeds_max_rows_raises(self, export_handler_registry):
        handler = _AsyncExportHandler("user", MAX_ROWS + 1)
        export_handler_registry.get_handler = MagicMock(return_value=handler)

        with patch("app.service.import_export_service.export_handler_registry", export_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.export(
                    db=None, redis=None, module="user", params={}
                )
            assert exc_info.value.code == ResultCode.EXPORT_ROWS_EXCEED_LIMIT

    @pytest.mark.asyncio
    async def test_export_module_not_supported_raises(self, export_handler_registry):
        with patch("app.service.import_export_service.export_handler_registry", export_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.export(
                    db=None, redis=None, module="unknown", params={}
                )
            assert exc_info.value.code == ResultCode.MODULE_IMPORT_NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_export_sync_returns_streaming_response(self, export_handler_registry):
        handler = _AsyncExportHandler("user", SYNC_THRESHOLD)
        export_handler_registry.get_handler = MagicMock(return_value=handler)

        with patch("app.service.import_export_service.export_handler_registry", export_handler_registry):
            from app.service.import_export_service import ImportExportService

            result = await ImportExportService.export(
                db=None, redis=None, module="user", params={}, format="excel"
            )
            assert isinstance(result, StreamingResponse)
            assert "attachment" in result.headers.get("content-disposition", "")

    @pytest.mark.asyncio
    async def test_export_csv_format_sets_csv_content_type(self, export_handler_registry):
        handler = _AsyncExportHandler("user", 10)
        export_handler_registry.get_handler = MagicMock(return_value=handler)

        with patch("app.service.import_export_service.export_handler_registry", export_handler_registry):
            from app.service.import_export_service import ImportExportService

            result = await ImportExportService.export(
                db=None, redis=None, module="user", params={}, format="csv"
            )
            assert isinstance(result, StreamingResponse)
            assert result.media_type == "text/csv"

    @pytest.mark.asyncio
    async def test_export_async_returns_task_vo(self, export_handler_registry):
        handler = _AsyncExportHandler("user", SYNC_THRESHOLD + 1)
        export_handler_registry.get_handler = MagicMock(return_value=handler)

        task_data = {"task_id": "task-001", "status": TaskStatus.PENDING.value}
        with patch("app.service.import_export_service.export_handler_registry", export_handler_registry), \
             patch("app.service.import_export_service.TaskServiceAsync") as TaskSvc:
            TaskSvc.create_task = AsyncMock(return_value=task_data)
            from app.service.import_export_service import ImportExportService

            result = await ImportExportService.export(
                db=None, redis=None, module="user", params={}, user_id=1
            )
            assert isinstance(result, dict)
            assert result["taskId"] == "task-001"
            assert result["status"] == TaskStatus.PENDING.value
            assert result["estimatedCount"] == SYNC_THRESHOLD + 1
            TaskSvc.create_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_export_force_sync_overrides_threshold(self, export_handler_registry):
        handler = _AsyncExportHandler("user", SYNC_THRESHOLD + 100)
        export_handler_registry.get_handler = MagicMock(return_value=handler)

        with patch("app.service.import_export_service.export_handler_registry", export_handler_registry):
            from app.service.import_export_service import ImportExportService

            result = await ImportExportService.export(
                db=None, redis=None, module="user", params={}, async_flag=False
            )
            assert isinstance(result, StreamingResponse)


class TestImportData:
    @pytest.mark.asyncio
    async def test_import_unsupported_file_type_raises(self, import_handler_registry):
        file = _make_upload_file("test.txt", b"hello", "text/plain")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.import_data(
                    db=None, redis=None, module="user", file=file
                )
            assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH

    @pytest.mark.asyncio
    async def test_import_file_too_large_raises(self, import_handler_registry):
        large_content = b"0" * (MAX_IMPORT_FILE_SIZE + 1)
        file = _make_upload_file("test.xlsx", large_content, "application/octet-stream")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.import_data(
                    db=None, redis=None, module="user", file=file
                )
            assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_SIZE_EXCEEDS

    @pytest.mark.asyncio
    async def test_import_csv_magic_mismatch_raises(self, import_handler_registry):
        content = b"\xff\xfe\x00\x01invalid"
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.import_data(
                    db=None, redis=None, module="user", file=file
                )
            assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH

    @pytest.mark.asyncio
    async def test_import_module_not_supported_raises(self, import_handler_registry):
        content = _valid_csv_bytes([["用户名"], ["u1"]])
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.import_data(
                    db=None, redis=None, module="unknown", file=file
                )
            assert exc_info.value.code == ResultCode.MODULE_IMPORT_NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_import_empty_rows_raises(self, import_handler_registry):
        handler = _AsyncImportHandler("user")
        import_handler_registry.get_handler = MagicMock(return_value=handler)
        content = _valid_csv_bytes([["用户名"]])
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.import_data(
                    db=None, redis=None, module="user", file=file
                )
            assert exc_info.value.code == ResultCode.IMPORT_FILE_EMPTY

    @pytest.mark.asyncio
    async def test_import_sync_returns_result_vo(self, import_handler_registry):
        result = ImportResult(
            total_rows=2, success_count=2, failure_count=0, errors=[]
        )
        handler = _AsyncImportHandler("user", result=result)
        import_handler_registry.get_handler = MagicMock(return_value=handler)
        content = _valid_csv_bytes([["用户名"], ["u1"], ["u2"]])
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            ret = await ImportExportService.import_data(
                db=None, redis=None, module="user", file=file, mode="all"
            )
            assert isinstance(ret, dict)
            assert ret["totalRows"] == 2
            assert ret["successCount"] == 2
            assert ret["failureCount"] == 0

    @pytest.mark.asyncio
    async def test_import_partial_mode_returns_errors(self, import_handler_registry):
        result = ImportResult(
            total_rows=2,
            success_count=1,
            failure_count=1,
            errors=[ImportError(row=2, message="用户名已存在")],
        )
        handler = _AsyncImportHandler("user", result=result)
        import_handler_registry.get_handler = MagicMock(return_value=handler)
        content = _valid_csv_bytes([["用户名"], ["u1"], ["u2"]])
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            ret = await ImportExportService.import_data(
                db=None, redis=None, module="user", file=file, mode="partial"
            )
            assert ret["failureCount"] == 1
            assert len(ret["errors"]) == 1
            assert ret["errors"][0]["row"] == 2

    @pytest.mark.asyncio
    async def test_import_async_returns_task_vo(self, import_handler_registry):
        handler = _AsyncImportHandler("user")
        import_handler_registry.get_handler = MagicMock(return_value=handler)
        rows_data = [["用户名"]] + [[f"u{i}"] for i in range(SYNC_THRESHOLD + 1)]
        content = _valid_csv_bytes(rows_data)
        file = _make_upload_file("test.csv", content, "text/csv")

        task_data = {"task_id": "task-import-001", "status": TaskStatus.PENDING.value}
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry), \
             patch("app.service.import_export_service.TaskServiceAsync") as TaskSvc, \
             patch("app.service.import_export_service._upload_import_file", new=AsyncMock(return_value="temp/imports/abc.csv")):
            TaskSvc.create_task = AsyncMock(return_value=task_data)
            from app.service.import_export_service import ImportExportService

            ret = await ImportExportService.import_data(
                db=None, redis=None, module="user", file=file, user_id=1
            )
            assert ret["taskId"] == "task-import-001"
            assert ret["status"] == TaskStatus.PENDING.value
            TaskSvc.create_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_import_rows_exceed_limit_raises(self, import_handler_registry):
        handler = _AsyncImportHandler("user")
        import_handler_registry.get_handler = MagicMock(return_value=handler)
        rows_data = [["用户名"]] + [[f"u{i}"] for i in range(MAX_ROWS + 1)]
        content = _valid_csv_bytes(rows_data)
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch("app.service.import_export_service.import_handler_registry", import_handler_registry):
            from app.service.import_export_service import ImportExportService

            with pytest.raises(BusinessException) as exc_info:
                await ImportExportService.import_data(
                    db=None, redis=None, module="user", file=file
                )
            assert exc_info.value.code == ResultCode.IMPORT_ROWS_EXCEED_LIMIT
