"""
角色导出处理器
"""

from __future__ import annotations

import io

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import BATCH_SIZE
from app.repository.role_repository import role_repository
from app.service.import_export.file_generator import write_csv, write_excel
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler


class RoleExportHandler(ExportHandler):
    def get_module(self) -> str:
        return "role"

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        keywords = query_params.get("keywords")
        filters = {"deleted": 0}
        search_fields = None
        if keywords:
            search_fields = [
                ("name", "like", f"%{keywords}%"),
                ("code", "like", f"%{keywords}%"),
            ]
        _, total = await role_repository.get_list(
            db,
            filters=filters,
            search_fields=search_fields,
            order_by="sort",
            page=1,
            page_size=1,
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
        filters = {"deleted": 0}
        search_fields = None
        if keywords:
            search_fields = [
                ("name", "like", f"%{keywords}%"),
                ("code", "like", f"%{keywords}%"),
            ]
        total = ctx.total_count or await self.estimate_count(db, params)

        page = 1
        page_size = BATCH_SIZE
        all_rows: list[dict] = []
        while True:
            roles, _ = await role_repository.get_list(
                db,
                filters=filters,
                search_fields=search_fields,
                order_by="sort",
                page=page,
                page_size=page_size,
            )
            if not roles:
                break
            all_rows.extend(_role_to_row(r) for r in roles)
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
            ExportFieldConfig(field="name", label="角色名称", order=2),
            ExportFieldConfig(field="code", label="角色编码", order=3),
            ExportFieldConfig(field="sort", label="排序", order=4),
            ExportFieldConfig(field="status_label", label="状态", order=5),
            ExportFieldConfig(field="data_scope_label", label="数据权限", order=6),
            ExportFieldConfig(
                field="create_time", label="创建时间", order=7, date_format="%Y-%m-%d %H:%M:%S"
            ),
        ]


_DATA_SCOPE_LABELS = {
    0: "全部数据",
    1: "部门及子部门数据",
    2: "本部门数据",
    3: "本人数据",
}


def _role_to_row(r) -> dict:
    status = int(r.status if r.status is not None else 1)
    data_scope = int(r.data_scope if r.data_scope is not None else 0)
    return {
        "id": r.id,
        "name": r.name or "",
        "code": r.code or "",
        "sort": r.sort if r.sort is not None else "",
        "status_label": "启用" if status == 1 else "禁用",
        "data_scope_label": _DATA_SCOPE_LABELS.get(data_scope, ""),
        "create_time": r.create_time,
    }
