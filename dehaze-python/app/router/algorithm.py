from typing import Optional

from app.core.code import ResultCode
from app.core.result import Result, error, success
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.schema.algorithm import (AlgorithmDeleteResultVO,
                                         AlgorithmForm, AlgorithmIdVO,
                                         AlgorithmOptionVO, AlgorithmVO)
from app.service.algorithm_service import AlgorithmService
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    prefix="/api/v1/algorithm",
    tags=["算法管理"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/", response_model=Result[list[AlgorithmVO]], summary="获取算法树形表格")
async def list_algorithms(
    keywords: Optional[str] = Query(default=None, description="关键词"),
    db: AsyncSession = Depends(get_db),
):
    algorithms = await AlgorithmService.get_algorithm_list(db, keywords)
    return success(algorithms)


@router.get(
    "/options", response_model=Result[list[AlgorithmOptionVO]], summary="获取算法下拉选项"
)
async def get_algorithm_options(
    db: AsyncSession = Depends(get_db),
):
    options = await AlgorithmService.get_algorithm_options(db)
    return success(options)


@router.get(
    "/{algorithm_id}", response_model=Result[AlgorithmVO], summary="获取算法详情"
)
async def get_algorithm(
    algorithm_id: int,
    db: AsyncSession = Depends(get_db),
):
    algorithm = await AlgorithmService.get_algorithm_by_id(db, algorithm_id)
    if algorithm:
        return success(algorithm)
    return error("算法不存在", ResultCode.RESOURCE_NOT_FOUND.code)


@router.post(
    "/", response_model=Result[AlgorithmIdVO], summary="新增算法"
)
async def create_algorithm(
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
):
    algorithm_id = await AlgorithmService.create_algorithm(db, body.model_dump(exclude_none=True))
    return success(AlgorithmIdVO(id=algorithm_id), msg="算法创建成功")


@router.put("/{algorithm_id}", response_model=Result[None], summary="修改算法")
async def update_algorithm(
    algorithm_id: int,
    body: AlgorithmForm,
    db: AsyncSession = Depends(get_db),
):
    await AlgorithmService.update_algorithm(db, algorithm_id, body.model_dump(exclude_none=True))
    return success(msg="算法更新成功")


@router.delete(
    "/", response_model=Result[AlgorithmDeleteResultVO], summary="批量删除算法"
)
async def delete_algorithms(
    ids: str = Query(..., description="算法ID，多个以逗号分隔"),
    db: AsyncSession = Depends(get_db),
):
    algorithm_ids = [int(i) for i in ids.split(",")]
    count = await AlgorithmService.delete_algorithms(db, algorithm_ids)
    return success(AlgorithmDeleteResultVO(count=count), msg="算法删除成功")
