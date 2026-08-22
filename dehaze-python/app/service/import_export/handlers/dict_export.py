"""
字典导出处理器
"""

from __future__ import annotations

import io

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import BATCH_SIZE
from app.repository.dict_repository import dict_repository
from app.service.import_export.file_generator import write_csv, write_excel
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler


class DictExportHandler(ExportHandler):
    def get_module(self) -> str:
        return "dict"

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        keywords = query_params.get("keywords")
        type_code = query_params.get("typeCode")
        _, total = await dict_repository.get_page(
            db,
            page=1,
            page_size=1,
            keywords=keywords,
            type_code=type_code,
        )
        return int(total)

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
        type_code = params.get("typeCode")
        total = ctx.total_count or await self.estimate_count(db, params)

        page = 1
        page_size = BATCH_SIZE
        all_rows: list[dict] = []
        while True:
            items, _ = await dict_repository.get_page(
                db,
                page=page,
                page_size=page_size,
                keywords=keywords,
                type_code=type_code,
            )
            if not items:
                break
            all_rows.extend(_dict_to_row(d) for d in items)
            processed = page * page_size
            await progress_cb(min(processed, total), total)
            if await cancel_cb():
                break
            page += 1

        fields = self.filter_fields(ctx.selected_fields)
        if ctx.format == "csv":
            write_csv(fields, all_rows, output)
        else:
            write_excel(fields, all_rows, output)

    def get_field_configs(self) -> list[ExportFieldConfig]:
        return [
            ExportFieldConfig(field="id", label="ID", order=1),
            ExportFieldConfig(field="type_code", label="字典类型编码", order=2),
            ExportFieldConfig(field="name", label="字典名称", order=3),
            ExportFieldConfig(field="value", label="字典值", order=4),
            ExportFieldConfig(field="sort", label="排序", order=5),
            ExportFieldConfig(field="status_label", label="状态", order=6),
            ExportFieldConfig(field="defaulted_label", label="是否默认", order=7),
            ExportFieldConfig(field="remark", label="备注", order=8),
            ExportFieldConfig(
                field="create_time", label="创建时间", order=9, date_format="%Y-%m-%d %H:%M:%S"
            ),
        ]


def _dict_to_row(d) -> dict:
    status = int(d.status if d.status is not None else 1)
    defaulted = int(d.defaulted if d.defaulted is not None else 0)
    return {
        "id": d.id,
        "type_code": d.type_code or "",
        "name": d.name or "",
        "value": d.value or "",
        "sort": d.sort if d.sort is not None else "",
        "status_label": "启用" if status == 1 else "禁用",
        "defaulted_label": "是" if defaulted == 1 else "否",
        "remark": d.remark or "",
        "create_time": d.create_time,
    }
