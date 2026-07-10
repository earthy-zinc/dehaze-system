from typing import Optional

from app.core.code import ResultCode
from app.core.result import Result, error, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.dept import DeptForm, DeptFormVO, DeptOptionVO, DeptVO
from app.service.dept_service import DeptService
from fastapi import APIRouter, Body, Depends, Path, Query
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    prefix="/api/v1/dept",
    tags=["部门管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("", response_model=Result[list[DeptVO]], summary="获取部门列表")
async def list_depts(
    keywords: Optional[str] = Query(default=None),
    status: Optional[int] = Query(default=None, ge=0, le=1),
    db: AsyncSession = Depends(get_db),
):
    depts = await DeptService.get_dept_list(db, keywords, status)
    return success(depts)


@router.get("/options", response_model=Result[list[DeptOptionVO]], summary="获取部门下拉选项")
async def list_dept_options(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    options = await DeptService.get_dept_options(db, redis)
    return success(options)


@router.get("/{dept_id}/form", response_model=Result[DeptFormVO], summary="获取部门表单数据")
async def get_dept_form(
    dept_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
):
    dept_form = await DeptService.get_dept_form(db, dept_id)
    if not dept_form:
        return error("部门不存在", ResultCode.RESOURCE_NOT_FOUND.code)
    return success(dept_form)


@router.post("", response_model=Result[dict[str, int]], summary="新增部门")
@require_permission("sys:dept:add")
async def create_dept(
    body: DeptForm,  # type: ignore
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    dept_id = await DeptService.create_dept(db, redis, body.model_dump(exclude_none=True))
    return success({"id": dept_id}, msg="部门创建成功")


@router.put("/{dept_id}", response_model=Result[dict[str, int]], summary="修改部门")
@require_permission("sys:dept:edit")
async def update_dept(
    dept_id: int = Path(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    body: DeptForm = Body(...),
    user: UserContext = Depends(get_current_user),
):
    updated_id = await DeptService.update_dept(db, redis, dept_id, body.model_dump(exclude_none=True))
    return success({"id": updated_id}, msg="部门更新成功")


@router.delete("/{ids}", response_model=Result[None], summary="删除部门")
@require_permission("sys:dept:delete")
async def delete_depts(
    ids: str = Path(...),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    dept_ids = [int(i) for i in ids.split(",")]
    await DeptService.delete_depts(db, redis, dept_ids)
    return success(msg="部门删除成功")
