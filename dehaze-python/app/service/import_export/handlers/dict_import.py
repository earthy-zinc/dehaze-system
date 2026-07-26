"""
字典导入处理器
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.dict_repository import dict_repository
from app.service.import_export.models import (ImportError, ImportFieldConfig,
                                              ImportOptions, ImportResult)
from app.service.import_export.registry import ImportHandler


class DictImportHandler(ImportHandler):

    def get_module(self) -> str:
        return "dict"

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [
            ImportFieldConfig(field="type_code", label="字典类型编码", required=True, max_length=50),
            ImportFieldConfig(field="name", label="字典名称", required=True, max_length=50),
            ImportFieldConfig(field="value", label="字典值", required=True, max_length=50),
            ImportFieldConfig(field="sort", label="排序"),
            ImportFieldConfig(field="status_label", label="状态(启用/禁用)"),
            ImportFieldConfig(field="defaulted_label", label="是否默认(是/否)"),
            ImportFieldConfig(field="remark", label="备注"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "type_code": "gender",
                "name": "男",
                "value": "1",
                "sort": "1",
                "status_label": "启用",
                "defaulted_label": "否",
                "remark": "",
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
                type_code = _get_str(row, "type_code")
                if not type_code:
                    raise ValueError("字典类型编码为空")
                name = _get_str(row, "name")
                if not name:
                    raise ValueError("字典名称为空")
                value = _get_str(row, "value")
                if not value:
                    raise ValueError("字典值为空")
                if await dict_repository.get_by_type_code_and_value(db, type_code, value):
                    raise ValueError(f"同类型下字典值已存在: {type_code}/{value}")

                await dict_repository.create_dict(db, {
                    "typeCode": type_code,
                    "name": name,
                    "value": value,
                    "sort": _parse_int(row, "sort", 0),
                    "status": _parse_status(row, "status_label", 1),
                    "defaulted": _parse_defaulted(row, "defaulted_label", 0),
                    "remark": _get_str(row, "remark") or "",
                })
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


def _parse_defaulted(row: dict, key: str, default: int) -> int:
    v = _get_str(row, key)
    if not v:
        return default
    if v == "是":
        return 1
    if v == "否":
        return 0
    return default
