"""
菜单导出处理器
"""
from __future__ import annotations

import io

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import BATCH_SIZE
from app.repository.menu_repository import menu_repository
from app.service.import_export.file_generator import write_csv, write_excel
from app.service.import_export.models import ExportContext, ExportFieldConfig
from app.service.import_export.registry import ExportHandler

_MENU_TYPE_LABELS = {1: "目录", 2: "菜单", 3: "外链", 4: "按钮"}


class MenuExportHandler(ExportHandler):

    def get_module(self) -> str:
        return "menu"

    async def estimate_count(self, db: AsyncSession, query_params: dict) -> int:
        keywords = query_params.get("keywords")
        menus = await menu_repository.get_list(db, keyword=keywords)
        return len(menus)

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
        menus = await menu_repository.get_list(db, keyword=keywords)
        total = ctx.total_count or len(menus)

        all_rows: list[dict] = []
        for i, m in enumerate(menus, 1):
            all_rows.append(_menu_to_row(m))
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
            ExportFieldConfig(field="name", label="菜单名称", order=2),
            ExportFieldConfig(field="parent_id", label="父菜单ID", order=3),
            ExportFieldConfig(field="type_label", label="类型", order=4),
            ExportFieldConfig(field="path", label="路由路径", order=5),
            ExportFieldConfig(field="component", label="组件路径", order=6),
            ExportFieldConfig(field="perm", label="权限标识", order=7),
            ExportFieldConfig(field="visible_label", label="是否可见", order=8),
            ExportFieldConfig(field="sort", label="排序", order=9),
            ExportFieldConfig(field="icon", label="图标", order=10),
            ExportFieldConfig(field="redirect", label="跳转路径", order=11),
            ExportFieldConfig(field="create_time", label="创建时间", order=12, date_format="%Y-%m-%d %H:%M:%S"),
        ]


def _menu_to_row(m) -> dict:
    type_value = int(m.type) if m.type is not None else None
    visible = int(m.visible) if m.visible is not None else 1
    return {
        "id": m.id,
        "name": m.name or "",
        "parent_id": m.parent_id if m.parent_id is not None else "",
        "type_label": _MENU_TYPE_LABELS.get(type_value, "") if type_value is not None else "",
        "path": m.path or "",
        "component": m.component or "",
        "perm": m.perm or "",
        "visible_label": "显示" if visible == 1 else "隐藏",
        "sort": m.sort if m.sort is not None else "",
        "icon": m.icon or "",
        "redirect": m.redirect or "",
        "create_time": m.create_time,
    }
