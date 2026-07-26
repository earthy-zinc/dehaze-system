from typing import Optional

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.common import PageResult
from app.models.schema.dict import (DictForm, DictFormVO, DictOptionVO,
                                    DictPageVO, DictTypeForm, DictTypeFormVO,
                                    DictTypePageVO)
from app.service.dict_service import DictService, DictTypeService
from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/dict", tags=["字典管理"], dependencies=[Depends(get_current_user)])


@router.get("/types/page", response_model=Result[PageResult[DictTypePageVO]], summary="字典类型分页列表")
async def get_dict_type_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keywords: Optional[str] = Query(default=None, description="关键词(名称/编码)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """获取字典类型分页列表"""
    items, total = await DictTypeService.get_dict_type_page(db, pageNum, pageSize, keywords)

    type_list = [
        {
            "id": item.id,
            "name": item.name,
            "code": item.code,
            "status": item.status,
            "remark": item.remark,
            "createTime": item.create_time.strftime("%Y-%m-%d %H:%M:%S") if item.create_time else None,
        }
        for item in items
    ]

    return success({"list": type_list, "total": total})


@router.get("/types/{type_id}/form", response_model=Result[DictTypeFormVO | None], summary="字典类型表单数据", dependencies=[Depends(get_current_user)])
async def get_dict_type_form(
    type_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
):
    """获取字典类型表单数据"""
    dict_type_data = await DictTypeService.get_dict_type_form(db, type_id)
    if not dict_type_data:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典类型不存在")
    return success(dict_type_data)


@router.post("/types", response_model=Result[None], summary="新增字典类型")
@require_permission("sys:dict:type:add")
async def create_dict_type(
    body: DictTypeForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await DictTypeService.create_dict_type(db, body.model_dump(exclude_none=True))
    return success(msg="新增成功")


@router.put("/types/{type_id}", response_model=Result[None], summary="修改字典类型")
@require_permission("sys:dict:type:edit")
async def update_dict_type(
    type_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
    body: DictTypeForm = Body(...),
):
    await DictTypeService.update_dict_type(db, redis, type_id, body.model_dump(exclude_none=True))
    return success(msg="修改成功")


@router.delete("/types/{type_ids}", response_model=Result[None], summary="删除字典类型", description="force=true 时级联删除关联字典数据")
@require_permission("sys:dict:type:delete")
async def delete_dict_types(
    type_ids: str = Path(...),
    force: bool = Query(default=False, description="是否强制删除关联的字典数据"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    try:
        id_list = [int(i) for i in type_ids.split(",")]
    except ValueError:
        raise BusinessException(ResultCode.PARAM_ERROR, "参数错误")
    await DictTypeService.delete_dict_types(db, redis, id_list, force=force)
    return success(msg="删除成功")


@router.get("/page", response_model=Result[PageResult[DictPageVO]], summary="字典分页列表")
async def get_dict_page(
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    keywords: Optional[str] = Query(default=None, description="关键词"),
    typeCode: Optional[str] = Query(default=None, description="字典类型编码"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """获取字典分页列表"""
    if not typeCode:
        raise BusinessException(ResultCode.PARAM_IS_NULL, "字典类型编码不能为空")
    items, total = await DictService.get_dict_page(db, pageNum, pageSize, keywords, typeCode)

    dict_list = [
        {
            "id": item.id,
            "typeCode": item.type_code,
            "name": item.name,
            "value": item.value,
            "status": item.status,
            "defaulted": item.defaulted,
            "sort": item.sort,
            "remark": item.remark,
            "createTime": item.create_time.strftime("%Y-%m-%d %H:%M:%S") if item.create_time else None,
        }
        for item in items
    ]

    return success({"list": dict_list, "total": total})


@router.get("/{dict_id}/form", response_model=Result[DictFormVO | None], summary="字典表单数据", dependencies=[Depends(get_current_user)])
async def get_dict_form(
    dict_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
):
    """获取字典表单数据"""
    dict_data = await DictService.get_dict_form(db, dict_id)
    if not dict_data:
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "字典数据项不存在")
    return success(dict_data)


@router.post("", response_model=Result[None], summary="新增字典")
@require_permission("sys:dict:data:add")
async def create_dict(
    body: DictForm,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    await DictService.create_dict(db, redis, body.model_dump(exclude_none=True))
    return success(msg="新增成功")


@router.put("/{dict_id}", response_model=Result[None], summary="修改字典")
@require_permission("sys:dict:data:edit")
async def update_dict(
    dict_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
    body: DictForm = Body(...),
):
    await DictService.update_dict(db, redis, dict_id, body.model_dump(exclude_none=True))
    return success(msg="修改成功")


@router.delete("/{dict_ids}", response_model=Result[None], summary="删除字典", description="多个ID以逗号分隔")
@require_permission("sys:dict:data:delete")
async def delete_dict(
    dict_ids: str = Path(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    try:
        id_list = [int(i) for i in dict_ids.split(",")]
    except ValueError:
        raise BusinessException(ResultCode.PARAM_ERROR, "参数错误")
    await DictService.delete_dict(db, redis, id_list)
    return success(msg="删除成功")


@router.get("/{type_code}/options", response_model=Result[list[DictOptionVO]], summary="字典下拉列表", dependencies=[Depends(get_current_user)])
async def list_dict_options(
    type_code: str = Path(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    options = await DictService.list_dict_options(db, redis, type_code)
    return success(options)
