"""
角色导入处理器
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_user import SysRole
from app.repository.role_repository import BUILTIN_ROLE_CODES, role_repository
from app.service.import_export.models import (
    ImportError,
    ImportFieldConfig,
    ImportOptions,
    ImportResult,
)
from app.service.import_export.registry import ImportHandler


class RoleImportHandler(ImportHandler):
    def get_module(self) -> str:
        return "role"

    def get_field_configs(self) -> list[ImportFieldConfig]:
        return [
            ImportFieldConfig(field="name", label="角色名称", required=True, max_length=64),
            ImportFieldConfig(field="code", label="角色编码", required=True, max_length=32),
            ImportFieldConfig(field="sort", label="排序"),
            ImportFieldConfig(field="status_label", label="状态(启用/禁用)"),
        ]

    def get_template_sample_data(self) -> list[dict]:
        return [
            {
                "name": "普通用户",
                "code": "user",
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
                    raise ValueError("角色名称为空")
                code = _get_str(row, "code")
                if not code:
                    raise ValueError("角色编码为空")
                if code in BUILTIN_ROLE_CODES:
                    raise ValueError(f"内置角色编码不可导入: {code}")
                if await role_repository.check_code_exists(db, code):
                    raise ValueError(f"角色编码已被历史记录占用: {code}")

                role = SysRole(
                    name=name,
                    code=code,
                    sort=_parse_int(row, "sort", 0),
                    status=_parse_status(row, "status_label"),
                    data_scope=2,
                )
                await role_repository.create(db, role)
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

        if success_count:
            # 导入新增角色影响下拉选项，提交后失效 role:options 缓存
            from app.database import defer_after_commit
            from app.dependencies.redis import get_redis_client
            from app.infrastructure.cache.cache import CacheService
            from app.service.role_service import role_service

            async def _invalidate_options():
                redis = await get_redis_client()
                await CacheService(redis).delete(role_service.ROLE_OPTIONS_CACHE_KEY)

            defer_after_commit(db, _invalidate_options)

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


def _parse_status(row: dict, key: str) -> int:
    label = _get_str(row, key)
    if not label:
        return 1
    if label == "启用":
        return 1
    if label == "禁用":
        return 0
    raise ValueError(f"状态值无效，必须为 启用/禁用: {label}")
