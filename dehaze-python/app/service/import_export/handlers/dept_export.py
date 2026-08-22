"""
部门导出处理器
"""

from __future__ import annotations

import io

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import BATCH_SIZE
from app.repository.dept_repository import dept_repository
from app.service.import_export.file_generator import write_csv, write_excel
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler


class DeptExportHandler(ExportHandler):
    def get_module(self) -> str:
        return "dept"

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        keywords = query_params.get("keywords")
        status = _parse_int(query_params.get("status"))
        depts = await dept_repository.get_dept_list(db, keywords=keywords, status=status)
        return len(depts)

    async def export(
        self,
        db: AsyncSession,
        ctx: ExportContext,
        output: io.BytesIO,
        progress_cb,
        cancel_cb,
    ) -> None:
        params = ctx.query_params
        keywords = params.get("keywords")
        status = _parse_int(params.get("status"))
        depts = await dept_repository.get_dept_list(db, keywords=keywords, status=status)
        total = ctx.total_count or len(depts)

        all_rows: list[dict] = []
        for i, d in enumerate(depts, 1):
            all_rows.append(_dept_to_row(d))
            if i % BATCH_SIZE == 0:
                await progress_cb(min(i, total), total)
                if await cancel_cb():
                    break
        await progress_cb(len(all_rows), total)

        fields = self.filter_fields(ctx.selected_fields)
        if ctx.format == "csv":
            write_csv(fields, all_rows, output)
        else:
            write_excel(fields, all_rows, output)

    def get_field_configs(self) -> list[ExportFieldConfig]:
        return [
            ExportFieldConfig(field="id", label="ID", order=1),
            ExportFieldConfig(field="name", label="部门名称", order=2),
            ExportFieldConfig(field="parent_id", label="父部门ID", order=3),
            ExportFieldConfig(field="sort", label="排序", order=4),
            ExportFieldConfig(field="status_label", label="状态", order=5),
            ExportFieldConfig(
                field="create_time", label="创建时间", order=6, date_format="%Y-%m-%d %H:%M:%S"
            ),
        ]


def _dept_to_row(d) -> dict:
    status = int(d.status if d.status is not None else 1)
    return {
        "id": d.id,
        "name": d.name or "",
        "parent_id": d.parent_id if d.parent_id is not None else "",
        "sort": d.sort if d.sort is not None else "",
        "status_label": "启用" if status == 1 else "禁用",
        "create_time": d.create_time,
    }


def _parse_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None
