"""
通用导出策略

处理所有 *_export 类型的任务（user_export/role_export/dept_export/menu_export/
dict_export/algorithm_export/dataset_export）。策略本身只做调度，具体导出逻辑
由对应的 ExportHandler 实现。
"""

from __future__ import annotations

import asyncio
import io
import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import EXPORT_TASK_TYPES
from app.infrastructure.storage.minio_client import get_minio_client, minio_executor
from app.service.import_export.models import ExportContext
from app.service.import_export.registry import export_handler_registry
from app.service.task.strategies._common import make_cancel_cb, resolve_module
from app.service.task.strategy import CancelChecker, ProgressCallback, TaskStrategy


class GenericExportStrategy(TaskStrategy):
    def get_task_types(self) -> list[str]:
        return list(EXPORT_TASK_TYPES)

    async def execute(
        self,
        db: AsyncSession,
        sys_task: SysTask,
        params_json: str | None,
        progress_callback: ProgressCallback,
        cancel_checker: CancelChecker,
    ) -> str | None:
        params = json.loads(params_json or "{}")
        module = resolve_module(params, sys_task.task_type)

        handler = export_handler_registry.get_handler(module)
        query_params = params.get("queryParams") or {}
        fmt = params.get("format") or "excel"
        selected_fields = params.get("selectedFields")

        total_count = await handler.estimate_count(db, query_params)
        ctx = ExportContext(
            task_id=sys_task.task_id,
            module=module,
            format=fmt,
            selected_fields=selected_fields,
            query_params=query_params,
            total_count=total_count,
        )

        output = io.BytesIO()
        await handler.export(db, ctx, output, progress_callback, make_cancel_cb(cancel_checker))
        output.seek(0)

        ext = ".csv" if fmt == "csv" else ".xlsx"
        object_name = f"exports/{sys_task.task_id}/{module}_export{ext}"
        content_type = (
            "text/csv"
            if fmt == "csv"
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        if handler.use_direct_export():
            object_name = f"exports/{sys_task.task_id}/{module}_export.zip"
            content_type = "application/zip"

        await _upload_to_minio(output.getvalue(), object_name, content_type)
        return object_name


async def _upload_to_minio(data: bytes, object_name: str, content_type: str) -> None:
    client = get_minio_client()
    bucket = settings.MINIO_BUCKET_NAME

    def _sync() -> None:
        client.put_object(
            bucket,
            object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(minio_executor, _sync)
