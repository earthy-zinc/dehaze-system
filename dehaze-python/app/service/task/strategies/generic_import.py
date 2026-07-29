"""
通用导入策略

处理所有 *_import 类型的任务（user_import/role_import/dept_import/menu_import/
dict_import/algorithm_import）。策略本身只做调度，具体导入逻辑由对应的
ImportHandler 实现。
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.constants import RESULT_FILE_EXPIRE_DAYS
from app.core.exceptions import BusinessException, TaskCancelledException
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import IMPORT_TASK_TYPES
from app.service.file_service import _minio_executor, get_minio_client
from app.service.import_export.file_parser import parse_csv, parse_excel
from app.service.import_export.models import ImportOptions
from app.service.import_export.registry import import_handler_registry
from app.service.task.strategy import CancelChecker, ProgressCallback, TaskStrategy

logger = logging.getLogger(__name__)


class GenericImportStrategy(TaskStrategy):

    def get_task_types(self) -> list[str]:
        return list(IMPORT_TASK_TYPES)

    async def execute(
        self,
        db: AsyncSession,
        sys_task: SysTask,
        params_json: Optional[str],
        progress_callback: ProgressCallback,
        cancel_checker: CancelChecker,
    ) -> Optional[str]:
        params = json.loads(params_json or "{}")
        # 通用端点创建任务时 params 可能不含 module，从任务类型推导（如 user_import -> user）
        module = params.get("module") or sys_task.task_type.rsplit("_", 1)[0]
        if not module:
            raise BusinessException(ResultCode.TASK_PARAM_ERROR, "缺少模块参数 module")

        object_name = params.get("fileObjectName")
        if not object_name:
            raise BusinessException(ResultCode.TASK_PARAM_ERROR, "缺少导入文件 objectName")

        mode = params.get("mode") or "all"
        extra = params.get("extra") or {}

        handler = import_handler_registry.get_handler(module)
        fields_cfg = handler.get_field_configs()

        content = await _download_from_minio(object_name)
        if object_name.lower().endswith(".csv"):
            rows = parse_csv(content, fields_cfg)
        else:
            rows = parse_excel(content, fields_cfg)
        if not rows:
            raise BusinessException(ResultCode.IMPORT_FILE_EMPTY)

        options = ImportOptions(mode=mode, extra=extra)

        async def progress_cb(processed: int, total: int) -> None:
            await progress_callback(processed, total)

        async def cancel_cb() -> bool:
            if await cancel_checker():
                raise TaskCancelledException()
            return False

        result = await handler.import_batch(db, rows, options, progress_cb, cancel_cb)

        if result.errors:
            report_url = await _upload_error_report(result, sys_task.task_id, module)
            result.error_report_object_name = report_url

        return result.model_dump_json()


async def _download_from_minio(object_name: str) -> bytes:
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET_NAME

    def _sync() -> bytes:
        response = client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_minio_executor, _sync)


async def _upload_error_report(result, task_id: str, module: str) -> Optional[str]:
    if not result.errors:
        return None
    import csv

    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(["行号", "字段", "错误信息"])
    for err in result.errors:
        writer.writerow([err.row, err.field or "", err.message])
    data = output.getvalue().encode("utf-8")

    object_name = f"exports/{task_id}/{module}_import_errors.csv"
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET_NAME

    def _sync() -> str:
        from datetime import timedelta
        client.put_object(
            bucket, object_name,
            data=io.BytesIO(data), length=len(data),
            content_type="text/csv",
        )
        return client.presigned_get_object(
            bucket, object_name, expires=timedelta(days=RESULT_FILE_EXPIRE_DAYS),
        )

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_minio_executor, _sync)
    except Exception as e:
        logger.warning("错误报告上传失败: %s", e)
        return None
