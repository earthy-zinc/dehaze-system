"""
通用导入导出路由

对齐 Java GenericImportExportController：
- GET  /api/v1/{module}/_export  同步/异步导出（简单查询条件）
- POST /api/v1/{module}/_export  同步/异步导出（复杂查询条件）
- POST /api/v1/{module}/_import  同步/异步导入
- GET  /api/v1/{module}/template  下载导入模板
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.service.import_export_service import ImportExportService
from redis.asyncio import Redis

router = APIRouter(prefix="/api/v1", tags=["通用导入导出接口"])

_FRAMEWORK_QUERY_KEYS = {"format", "async", "fields"}
_FRAMEWORK_IMPORT_KEYS = {"file", "mode", "async"}


def _build_query_params(request: Request) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if key in _FRAMEWORK_QUERY_KEYS:
            continue
        if key in params:
            existing = params[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                params[key] = [existing, value]
        else:
            params[key] = value
    return params


async def _build_extra_params(request: Request) -> dict[str, Any]:
    params: dict[str, Any] = {}
    form = await request.form()
    for key, value in form.multi_items():
        if key in _FRAMEWORK_IMPORT_KEYS:
            continue
        if key in params:
            existing = params[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                params[key] = [existing, value]
        else:
            params[key] = value
    return params


def _check_module_permission(user: UserContext, module: str, action: str) -> None:
    if user.is_root:
        return
    required = f"sys:{module}:{action}"
    if required not in user.permissions and "*" not in user.permissions:
        from fastapi import HTTPException, status
        from app.core.code import ResultCode
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ResultCode.FORBIDDEN_OPERATION.msg,
        )


@router.get("/{module}/_export", summary="导出数据（GET，简单查询条件）")
async def export_get(
    module: str,
    request: Request,
    format: str = Query(default="excel", description="文件格式: excel/csv"),
    async_flag: Optional[bool] = Query(default=None, alias="async", description="是否强制异步"),
    fields: Optional[str] = Query(default=None, description="导出字段，逗号分隔"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    _check_module_permission(user, module, "export")
    query_params = _build_query_params(request)
    field_list = _split_fields(fields)
    result = await ImportExportService.export(
        db=db,
        redis=redis,
        module=module,
        params=query_params,
        format=format,
        async_flag=async_flag,
        fields=field_list,
        user_id=user.id,
    )
    if isinstance(result, StreamingResponse):
        return result
    return success(result)


@router.post("/{module}/_export", summary="导出数据（POST，复杂查询条件）")
async def export_post(
    module: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    _check_module_permission(user, module, "export")
    fmt = body.get("format") or "excel"
    async_flag = body.get("async")
    fields = body.get("fields")
    query_params = body.get("queryParams") or {}
    result = await ImportExportService.export(
        db=db,
        redis=redis,
        module=module,
        params=query_params,
        format=fmt,
        async_flag=async_flag,
        fields=fields,
        user_id=user.id,
    )
    if isinstance(result, StreamingResponse):
        return result
    return success(result)


@router.post("/{module}/_import", summary="导入数据")
async def import_data(
    module: str,
    request: Request,
    file: UploadFile = File(..., description="Excel/CSV 文件"),
    mode: str = Form(default="all", description="导入模式: all/partial"),
    async_flag: Optional[bool] = Form(default=None, alias="async", description="是否强制异步"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    _check_module_permission(user, module, "import")
    extra_params = await _build_extra_params(request)
    result = await ImportExportService.import_data(
        db=db,
        redis=redis,
        module=module,
        file=file,
        mode=mode,
        async_flag=async_flag,
        extra_params=extra_params,
        user_id=user.id,
    )
    return success(result)


@router.get("/{module}/template", summary="下载导入模板")
async def download_template(
    module: str,
    format: str = Query(default="excel", description="文件格式: excel/csv"),
    user: UserContext = Depends(get_current_user),
):
    _check_module_permission(user, module, "import")
    return ImportExportService.download_template(module, format)


def _split_fields(fields: Optional[str]) -> Optional[list[str]]:
    if not fields or not fields.strip():
        return None
    return [s.strip() for s in fields.split(",") if s.strip()]
