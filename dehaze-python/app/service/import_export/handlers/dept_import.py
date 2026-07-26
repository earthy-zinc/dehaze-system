"""
部门导入处理器
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_dept import SysDept
from app.repository.dept_repository import dept_repository
from app.service.import_export.models import (ImportError, ImportFieldConfig,
                                              ImportOptions, ImportResult)
from app.service.import_export.registry import ImportHandler


class DeptImportHandler(ImportHandler):

    def get_module(self) -> str:
        return "dept"

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [
            ImportFieldConfig(field="name", label="部门名称", required=True, max_length=64),
            ImportFieldConfig(field="parent_id", label="父部门ID(0为顶级)"),
            ImportFieldConfig(field="sort", label="排序"),
            ImportFieldConfig(field="status_label", label="状态(启用/禁用)"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "name": "研发部",
                "parent_id": "0",
                "sort": "1",
                "status_label": "启用",
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
                    raise ValueError("部门名称为空")
                parent_id = _parse_int(row, "parent_id", 0) or 0
                if await dept_repository.check_name_exists(db, name, parent_id=parent_id):
                    raise ValueError(f"同层级下部门名称已存在: {name}")
                tree_path = await dept_repository.generate_tree_path(db, parent_id)

                dept = SysDept(
                    name=name,
                    parent_id=parent_id,
                    tree_path=tree_path,
                    sort=_parse_int(row, "sort", 0),
                    status=_parse_status(row, "status_label", 1),
                )
                await dept_repository.create(db, dept)
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


def _parse_status(row: dict, key: str, default: int) -> int:
    label = _get_str(row, key)
    if not label:
        return default
    if label == "启用":
        return 1
    if label == "禁用":
        return 0
    return default
