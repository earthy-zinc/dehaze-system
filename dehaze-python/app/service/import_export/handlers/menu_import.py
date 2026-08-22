"""
菜单导入处理器
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_menu import SysMenu
from app.repository.menu_repository import menu_repository
from app.service.import_export.models import (
    ImportError,
    ImportFieldConfig,
    ImportOptions,
    ImportResult,
)
from app.service.import_export.registry import ImportHandler

_MENU_TYPE_VALUES = {"目录": 1, "菜单": 2, "外链": 3, "按钮": 4}


class MenuImportHandler(ImportHandler):
    def get_module(self) -> str:
        return "menu"

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [
            ImportFieldConfig(field="name", label="菜单名称", required=True, max_length=64),
            ImportFieldConfig(field="parent_id", label="父菜单ID(0为顶级)"),
            ImportFieldConfig(field="type_label", label="类型(目录/菜单/外链/按钮)", required=True),
            ImportFieldConfig(field="path", label="路由路径"),
            ImportFieldConfig(field="component", label="组件路径"),
            ImportFieldConfig(field="perm", label="权限标识"),
            ImportFieldConfig(field="visible_label", label="是否可见(显示/隐藏)"),
            ImportFieldConfig(field="sort", label="排序"),
            ImportFieldConfig(field="icon", label="图标"),
            ImportFieldConfig(field="redirect", label="跳转路径"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "name": "用户管理",
                "parent_id": "0",
                "type_label": "菜单",
                "path": "/system/user",
                "component": "system/user/index",
                "perm": "sys:user:list",
                "visible_label": "显示",
                "sort": "1",
                "icon": "user",
                "redirect": "",
            }
        ]

    async def import_batch(
        self,
        db: AsyncSession,
        rows: list[dict],
        options: ImportOptions,
        progress_cb,
        cancel_cb,
    ) -> ImportResult:
        partial = options.mode == "partial"
        errors: list[ImportError] = []
        success_count = 0
        failure_count = 0
        total = len(rows)

        for i, row in enumerate(rows):
            row_num = i + 2
            try:
                name = _get_str(row, "name")
                if not name:
                    raise ValueError("菜单名称为空")
                type_label = _get_str(row, "type_label")
                if not type_label:
                    raise ValueError("菜单类型为空")
                type_value = _MENU_TYPE_VALUES.get(type_label)
                if type_value is None:
                    raise ValueError(f"菜单类型无效(应为 目录/菜单/外链/按钮): {type_label}")

                menu = SysMenu(
                    name=name,
                    parent_id=_parse_int(row, "parent_id", 0) or 0,
                    type=type_value,
                    path=_get_str(row, "path") or "",
                    component=_get_str(row, "component"),
                    perm=_get_str(row, "perm"),
                    visible=_parse_visible(row, "visible_label", 1),
                    sort=_parse_int(row, "sort", 0),
                    icon=_get_str(row, "icon") or "",
                    redirect=_get_str(row, "redirect"),
                )
                await menu_repository.create_menu(db, menu)
                success_count += 1
            except Exception as e:
                failure_count += 1
                errors.append(ImportError(row=row_num, message=str(e)))
                if not partial:
                    return ImportResult(
                        total_rows=total,
                        success_count=success_count,
                        failure_count=failure_count,
                        skipped_count=0,
                        errors=errors,
                    )
            if (i + 1) % 100 == 0:
                await progress_cb(i + 1, total)
                if await cancel_cb():
                    break

        return ImportResult(
            total_rows=total,
            success_count=success_count,
            failure_count=failure_count,
            skipped_count=0,
            errors=errors,
        )


def _get_str(row: dict, key: str) -> str | None:
    v = row.get(key)
    if v is None:
        return None
    return str(v).strip()


def _parse_int(row: dict, key: str, default: int) -> int:
    v = _get_str(row, key)
    if not v:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def _parse_visible(row: dict, key: str, default: int) -> int:
    v = _get_str(row, key)
    if not v:
        return default
    if v == "显示":
        return 1
    if v == "隐藏":
        return 0
    return default
