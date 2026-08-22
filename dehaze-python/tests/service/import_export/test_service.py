from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from app.core.code import ResultCode
from app.core.constants import MAX_IMPORT_FILE_SIZE, MAX_ROWS, SYNC_THRESHOLD
from app.core.exceptions import BusinessException
from app.models.enum.task_enum import TaskStatus
from app.service.import_export.models import ImportError, ImportFieldConfig, ImportResult
from app.service.import_export.registry import (
    ExportHandlerRegistry,
    ImportHandlerRegistry,
)
from app.service.import_export_service import import_export_service


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


class _FakeExportHandler:
    def __init__(self, module: str, count: int):
        self._module = module
        self._count = count

    def get_module(self) -> str:
        return self._module

    async def estimate_count(self, db, query_params: dict) -> int:
        return self._count

    async def export(self, db, ctx, output, progress_cb, cancel_cb) -> None:
        output.write(b"fake-excel-content")

    def use_direct_export(self) -> bool:
        return False


class _FakeImportHandler:
    def __init__(self, module: str, result: ImportResult | None = None):
        self._module = module
        self._result = result or ImportResult(total_rows=1, success_count=1)

    def get_module(self) -> str:
        return self._module

    def get_field_configs(self):
        return [ImportFieldConfig(field="username", label="用户名", required=True)]

    async def import_batch(self, db, rows, options, progress_cb, cancel_cb) -> ImportResult:
        return self._result


def _export_registry_with(handler) -> ExportHandlerRegistry:
    registry = ExportHandlerRegistry()
    registry.register(handler)
    return registry


def _import_registry_with(handler) -> ImportHandlerRegistry:
    registry = ImportHandlerRegistry()
    registry.register(handler)
    return registry


class TestExport:
    async def test_export_count_exceeds_max_rows_raises(self):
        handler = _FakeExportHandler("user", MAX_ROWS + 1)
        with patch(
            "app.service.import_export_service.export_handler_registry",
            _export_registry_with(handler),
        ):
            with pytest.raises(BusinessException) as exc_info:
                await import_export_service.export(db=None, redis=None, module="user", params={})
            assert exc_info.value.code == ResultCode.EXPORT_ROWS_EXCEED_LIMIT

    async def test_export_sync_returns_streaming_response(self):
        handler = _FakeExportHandler("user", SYNC_THRESHOLD)
        with patch(
            "app.service.import_export_service.export_handler_registry",
            _export_registry_with(handler),
        ):
            result = await import_export_service.export(
                db=None, redis=None, module="user", params={}, format="excel"
            )
            assert isinstance(result, StreamingResponse)
            assert result.headers["content-disposition"].startswith("attachment")
            assert "user_export.xlsx" in result.headers["content-disposition"]

    async def test_export_csv_format_sets_csv_content_type(self):
        handler = _FakeExportHandler("user", 10)
        with patch(
            "app.service.import_export_service.export_handler_registry",
            _export_registry_with(handler),
        ):
            result = await import_export_service.export(
                db=None, redis=None, module="user", params={}, format="csv"
            )
            assert isinstance(result, StreamingResponse)
            assert result.media_type == "text/csv"

    async def test_export_async_returns_task_vo(self):
        handler = _FakeExportHandler("user", SYNC_THRESHOLD + 1)
        task_data = {"task_id": "task-001", "status": TaskStatus.PENDING.value}
        with (
            patch(
                "app.service.import_export_service.export_handler_registry",
                _export_registry_with(handler),
            ),
            patch("app.service.import_export_service.create_task", autospec=True) as create_task,
        ):
            create_task.return_value = task_data
            result = await import_export_service.export(
                db=None, redis=None, module="user", params={"keywords": "张"}, user_id=1
            )
            assert result == {
                "taskId": "task-001",
                "status": TaskStatus.PENDING.value,
                "estimatedCount": SYNC_THRESHOLD + 1,
            }
            create_task.assert_awaited_once()
            call_kwargs = create_task.call_args.kwargs
            assert call_kwargs["task_type"] == "user_export"
            assert call_kwargs["user_id"] == 1
            assert json.loads(call_kwargs["params_json"]) == {
                "module": "user",
                "queryParams": {"keywords": "张"},
                "format": "excel",
                "selectedFields": None,
            }

    async def test_export_force_sync_overrides_threshold(self):
        handler = _FakeExportHandler("user", SYNC_THRESHOLD + 100)
        with (
            patch(
                "app.service.import_export_service.export_handler_registry",
                _export_registry_with(handler),
            ),
            patch("app.service.import_export_service.create_task", autospec=True) as create_task,
        ):
            result = await import_export_service.export(
                db=None, redis=None, module="user", params={}, async_flag=False
            )
            assert isinstance(result, StreamingResponse)
            create_task.assert_not_called()


class TestImportData:
    async def test_import_unsupported_file_type_raises(self):
        file = _make_upload_file("test.txt", b"hello", "text/plain")
        with pytest.raises(BusinessException) as exc_info:
            await import_export_service.import_data(db=None, redis=None, module="user", file=file)
        assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH

    async def test_import_file_too_large_raises(self):
        large_content = b"0" * (MAX_IMPORT_FILE_SIZE + 1)
        file = _make_upload_file("test.xlsx", large_content, "application/octet-stream")
        with pytest.raises(BusinessException) as exc_info:
            await import_export_service.import_data(db=None, redis=None, module="user", file=file)
        assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_SIZE_EXCEEDS

    async def test_import_csv_magic_mismatch_raises(self):
        content = b"\xff\xfe\x00\x01invalid"
        file = _make_upload_file("test.csv", content, "text/csv")
        with pytest.raises(BusinessException) as exc_info:
            await import_export_service.import_data(db=None, redis=None, module="user", file=file)
        assert exc_info.value.code == ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH

    async def test_import_empty_rows_raises(self):
        handler = _FakeImportHandler("user")
        with patch(
            "app.service.import_export_service.import_handler_registry",
            _import_registry_with(handler),
        ):
            content = _valid_csv_bytes([["用户名"]])
            file = _make_upload_file("test.csv", content, "text/csv")
            with pytest.raises(BusinessException) as exc_info:
                await import_export_service.import_data(
                    db=None, redis=None, module="user", file=file
                )
            assert exc_info.value.code == ResultCode.IMPORT_FILE_EMPTY

    async def test_import_sync_returns_result_vo(self):
        result = ImportResult(total_rows=2, success_count=2, failure_count=0, errors=[])
        handler = _FakeImportHandler("user", result=result)
        with patch(
            "app.service.import_export_service.import_handler_registry",
            _import_registry_with(handler),
        ):
            content = _valid_csv_bytes([["用户名"], ["u1"], ["u2"]])
            file = _make_upload_file("test.csv", content, "text/csv")
            ret = await import_export_service.import_data(
                db=None, redis=None, module="user", file=file, mode="all"
            )
            assert ret == {
                "totalRows": 2,
                "successCount": 2,
                "failureCount": 0,
                "skippedCount": 0,
                "errors": [],
                "errorReportUrl": None,
            }

    async def test_import_partial_mode_returns_errors(self):
        result = ImportResult(
            total_rows=2,
            success_count=1,
            failure_count=1,
            errors=[ImportError(row=2, message="用户名已存在")],
        )
        handler = _FakeImportHandler("user", result=result)
        with patch(
            "app.service.import_export_service.import_handler_registry",
            _import_registry_with(handler),
        ):
            content = _valid_csv_bytes([["用户名"], ["u1"], ["u2"]])
            file = _make_upload_file("test.csv", content, "text/csv")
            ret = await import_export_service.import_data(
                db=None, redis=None, module="user", file=file, mode="partial"
            )
            assert ret["failureCount"] == 1
            assert ret["errors"] == [{"row": 2, "field": None, "message": "用户名已存在"}]

    async def test_import_async_returns_task_vo(self):
        handler = _FakeImportHandler("user")
        rows_data = [["用户名"]] + [[f"u{i}"] for i in range(SYNC_THRESHOLD + 1)]
        content = _valid_csv_bytes(rows_data)
        file = _make_upload_file("test.csv", content, "text/csv")
        task_data = {"task_id": "task-import-001", "status": TaskStatus.PENDING.value}
        with (
            patch(
                "app.service.import_export_service.import_handler_registry",
                _import_registry_with(handler),
            ),
            patch("app.service.import_export_service.create_task", autospec=True) as create_task,
            patch(
                "app.service.import_export_service._upload_import_file",
                new=AsyncMock(return_value="temp/imports/abc.csv"),
            ),
        ):
            create_task.return_value = task_data
            ret = await import_export_service.import_data(
                db=None, redis=None, module="user", file=file, user_id=1
            )
            assert ret == {"taskId": "task-import-001", "status": TaskStatus.PENDING.value}
            create_task.assert_awaited_once()
            call_kwargs = create_task.call_args.kwargs
            assert call_kwargs["task_type"] == "user_import"
            assert call_kwargs["user_id"] == 1
            assert json.loads(call_kwargs["params_json"]) == {
                "module": "user",
                "fileObjectName": "temp/imports/abc.csv",
                "mode": "all",
                "extra": {},
            }

    async def test_import_rows_exceed_limit_raises(self):
        handler = _FakeImportHandler("user")
        rows_data = [["用户名"]] + [[f"u{i}"] for i in range(MAX_ROWS + 1)]
        content = _valid_csv_bytes(rows_data)
        file = _make_upload_file("test.csv", content, "text/csv")
        with patch(
            "app.service.import_export_service.import_handler_registry",
            _import_registry_with(handler),
        ):
            with pytest.raises(BusinessException) as exc_info:
                await import_export_service.import_data(db=None, redis=None, module="user", file=file)
            assert exc_info.value.code == ResultCode.IMPORT_ROWS_EXCEED_LIMIT
