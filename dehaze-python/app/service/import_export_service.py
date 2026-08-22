"""
通用导入导出服务

对齐 Java ImportExportService：
- 同步/异步自适应
- 文件验证（类型、大小、魔数）
- 病毒扫描接口
- MinIO 上传/下载
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from typing import Any

from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.constants import (
    MAX_IMPORT_FILE_SIZE,
    MAX_ROWS,
    SYNC_THRESHOLD,
)
from app.core.exceptions import BusinessException
from app.models.schema.task import ExportTaskVO, ImportErrorVO, ImportResultVO, ImportTaskVO
from app.service.import_export.file_parser import parse_csv, parse_excel
from app.service.import_export.registry import (
    ExportHandler,
    export_handler_registry,
    import_handler_registry,
)
from app.service.import_export.template_manager import (
    generate_template_csv,
    generate_template_excel,
)
from app.service.import_export.virus_scanner import get_virus_scanner
from app.service.storage.factory import get_storage_service
from app.service.task_service import create_task

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
_ALLOWED_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "application/octet-stream",
    "text/plain",
}
_EXCEL_MAGIC = (b"\x50\x4b\x03\x04", b"\xd0\xcf\x11\xe0")


class ImportExportService:
    @staticmethod
    def get_supported_export_modules() -> list[str]:
        return list(export_handler_registry._handlers.keys())

    @staticmethod
    def get_supported_import_modules() -> list[str]:
        return list(import_handler_registry._handlers.keys())

    @staticmethod
    async def export(
        db: AsyncSession,
        redis: Redis,
        module: str,
        params: dict[str, Any],
        format: str = "excel",
        async_flag: bool | None = None,
        fields: list[str] | None = None,
        user_id: int = 0,
    ) -> dict | StreamingResponse:
        handler = export_handler_registry.get_handler(module)
        count = await handler.estimate_count(db, params)
        if count > MAX_ROWS:
            raise BusinessException(
                ResultCode.EXPORT_ROWS_EXCEED_LIMIT,
                f"导出行数 {count} 超出限制 {MAX_ROWS}",
            )
        should_async = (
            async_flag
            if async_flag is not None
            else (count > SYNC_THRESHOLD or handler.use_direct_export())
        )
        if should_async:
            task_params = _build_export_task_params(module, params, format, fields)
            task_data = await create_task(
                db=db,
                redis=redis,
                task_type=_build_export_task_type(module),
                params_json=json.dumps(task_params, ensure_ascii=False),
                user_id=user_id,
            )
            return ExportTaskVO(
                taskId=task_data["task_id"],
                status=task_data["status"],
                estimatedCount=count,
            ).model_dump()
        return await _sync_export(db, handler, params, format, fields, module)

    @staticmethod
    async def import_data(
        db: AsyncSession,
        redis: Redis,
        module: str,
        file: UploadFile,
        mode: str = "all",
        async_flag: bool | None = None,
        extra_params: dict | None = None,
        user_id: int = 0,
    ) -> dict:
        content = await _validate_upload_file(file)
        scanner = get_virus_scanner()
        if scanner.is_enabled() and not scanner.scan(content):
            raise BusinessException(ResultCode.USER_UPLOAD_FILE_ERROR, "文件未通过病毒扫描")
        handler = import_handler_registry.get_handler(module)
        fields_cfg = handler.get_field_configs()
        ext = _get_extension(file.filename or "")
        if ext == ".csv":
            rows = parse_csv(content, fields_cfg)
        else:
            rows = parse_excel(content, fields_cfg)
        if not rows:
            raise BusinessException(ResultCode.IMPORT_FILE_EMPTY)
        if len(rows) > MAX_ROWS:
            raise BusinessException(
                ResultCode.IMPORT_ROWS_EXCEED_LIMIT,
                f"导入行数 {len(rows)} 超出限制 {MAX_ROWS}",
            )
        should_async = async_flag if async_flag is not None else len(rows) > SYNC_THRESHOLD
        if should_async:
            object_name = await _upload_import_file(file.filename or "import.bin", content)
            task_params = _build_import_task_params(module, object_name, mode, extra_params)
            task_data = await create_task(
                db=db,
                redis=redis,
                task_type=_build_import_task_type(module),
                params_json=json.dumps(task_params, ensure_ascii=False),
                user_id=user_id,
            )
            return ImportTaskVO(
                taskId=task_data["task_id"],
                status=task_data["status"],
            ).model_dump()
        from app.service.import_export.models import ImportOptions

        options = ImportOptions(mode=mode, extra=extra_params or {})
        result = await handler.import_batch(db, rows, options, _noop_progress, _noop_cancel)
        return _to_result_vo(result).model_dump()

    @staticmethod
    def download_template(module: str, format: str = "excel") -> StreamingResponse:
        handler = import_handler_registry.get_handler(module)
        if format == "csv":
            content = generate_template_csv(handler)
            media_type = "text/csv"
            filename = f"{module}_template.csv"
        else:
            content = generate_template_excel(handler)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"{module}_template.xlsx"
        return StreamingResponse(
            io.BytesIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


def _build_export_task_type(module: str) -> str:
    return f"{module}_export"


def _build_import_task_type(module: str) -> str:
    return f"{module}_import"


def _build_export_task_params(
    module: str,
    params: dict,
    format: str,
    fields: list[str] | None,
) -> dict:
    return {
        "module": module,
        "queryParams": params,
        "format": format,
        "selectedFields": fields,
    }


def _build_import_task_params(
    module: str,
    object_name: str,
    mode: str,
    extra_params: dict | None,
) -> dict:
    return {
        "module": module,
        "fileObjectName": object_name,
        "mode": mode,
        "extra": extra_params or {},
    }


async def _sync_export(
    db: AsyncSession,
    handler: ExportHandler,
    params: dict,
    format: str,
    fields: list[str] | None,
    module: str,
) -> StreamingResponse:
    from app.service.import_export.models import ExportContext

    ctx = ExportContext(
        task_id="sync",
        module=module,
        format=format,
        selected_fields=fields,
        query_params=params,
    )
    output = io.BytesIO()
    await handler.export(db, ctx, output, _noop_progress, _noop_cancel)
    output.seek(0)
    if format == "csv":
        media_type = "text/csv"
        filename = f"{module}_export.csv"
    else:
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{module}_export.xlsx"
    return StreamingResponse(
        output,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _validate_upload_file(file: UploadFile) -> bytes:
    filename = file.filename or ""
    ext = _get_extension(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        raise BusinessException(
            ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "仅支持 .xlsx/.xls/.csv 格式"
        )
    content = await file.read()
    if len(content) > MAX_IMPORT_FILE_SIZE:
        raise BusinessException(
            ResultCode.USER_UPLOAD_FILE_SIZE_EXCEEDS,
            f"文件大小超限（最大 {MAX_IMPORT_FILE_SIZE // (1024 * 1024)}MB）",
        )
    if not _check_magic(ext, content):
        raise BusinessException(
            ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "文件内容与扩展名不匹配"
        )
    return content


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _check_magic(ext: str, content: bytes) -> bool:
    if ext == ".csv":
        try:
            content.decode("utf-8")
            return True
        except UnicodeDecodeError:
            try:
                content.decode("gbk")
                return True
            except UnicodeDecodeError:
                return False
    if ext in (".xlsx", ".xls"):
        return any(content.startswith(magic) for magic in _EXCEL_MAGIC)
    return False


async def _upload_import_file(filename: str, content: bytes) -> str:
    ext = _get_extension(filename)
    object_name = f"temp/imports/{uuid.uuid4().hex}{ext}"
    storage = get_storage_service()
    bucket = settings.MINIO_BUCKET_NAME
    content_type = (
        "text/csv"
        if ext == ".csv"
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: storage.upload(bucket, object_name, content, content_type),
    )
    return object_name


async def _noop_progress(processed: int, total: int) -> None:
    return None


async def _noop_cancel() -> bool:
    return False


def _to_result_vo(result) -> ImportResultVO:
    return ImportResultVO(
        totalRows=result.total_rows,
        successCount=result.success_count,
        failureCount=result.failure_count,
        skippedCount=result.skipped_count,
        errors=[ImportErrorVO(row=e.row, field=e.field, message=e.message) for e in result.errors],
        errorReportUrl=None,
    )


import_export_service = ImportExportService()
