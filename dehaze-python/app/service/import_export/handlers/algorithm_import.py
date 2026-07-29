"""
算法导入处理器
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import (AlgorithmStatus,
                                                 algorithm_repository)
from app.service.import_export.models import (ImportError, ImportFieldConfig,
                                              ImportOptions, ImportResult)
from app.service.import_export.registry import ImportHandler


class AlgorithmImportHandler(ImportHandler):

    def get_module(self) -> str:
        return "algorithms"

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [
            ImportFieldConfig(field="name", label="算法名称", required=True, max_length=50),
            ImportFieldConfig(field="type", label="算法类型", required=True),
            ImportFieldConfig(field="parent_id", label="父算法ID(0为顶级)"),
            ImportFieldConfig(field="path", label="模型文件路径"),
            ImportFieldConfig(field="import_path", label="导入路径"),
            ImportFieldConfig(field="description", label="描述"),
            ImportFieldConfig(field="version", label="版本"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "name": "示例去雾算法",
                "type": "image_dehaze",
                "parent_id": "0",
                "path": "/models/example.pth",
                "import_path": "algorithms.example",
                "description": "示例算法",
                "version": "1.0.0",
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

        all_names = [n for n in (_get_str(r, "name") for r in rows) if n]
        existing_algos = await algorithm_repository.get_list_with_keywords(db, None)
        existing_names = {a.name for a in existing_algos}
        seen: set[str] = set()

        for i, row in enumerate(rows):
            row_num = i + 2
            try:
                name = _get_str(row, "name")
                if not name:
                    raise ValueError("算法名称为空")
                type_ = _get_str(row, "type")
                if not type_:
                    raise ValueError("算法类型为空")
                if name in existing_names or name in seen:
                    raise ValueError(f"算法名称已存在: {name}")
                seen.add(name)

                algorithm = SysAlgorithm(
                    name=name,
                    type=type_,
                    parent_id=_parse_int(row, "parent_id", 0) or 0,
                    path=_get_str(row, "path") or "",
                    import_path=_get_str(row, "import_path"),
                    description=_get_str(row, "description"),
                    version=_get_str(row, "version"),
                    status=AlgorithmStatus.DRAFT,
                )
                await algorithm_repository.create(db, algorithm)
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
