"""智能体评测路由（AI对话 - 智能体管理）

端点前缀 /api/v1/ai/agents/{agent_id}/eval：
- 评测集 CRUD、样本 CRUD、评测执行记录查询
- 手动触发评测（trigger_type=manual）

权限：ai:agent:manage
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis_client
from app.models.schema.ai_agent import (
    EvalDatasetCreate,
    EvalDatasetResult,
    EvalDatasetUpdate,
    EvalRunResult,
    EvalSampleCreate,
    EvalSampleResult,
    EvalSampleUpdate,
)
from app.models.schema.common import PageResult
from app.service.ai_eval_service import eval_service

router = APIRouter(prefix="/api/v1/ai/agents/{agent_id}/eval", tags=["AI对话-智能体评测"])


# ── 评测集 ──────────────────────────────────────────────────────


@router.post("/datasets", response_model=Result[EvalDatasetResult], summary="创建评测集")
@require_permission("ai:agent:manage")
async def create_dataset(
    agent_id: int,
    form: EvalDatasetCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_service.create_dataset(db, agent_id, form)
    return success(EvalDatasetResult.model_validate(result))


@router.get("/datasets", response_model=Result[list[EvalDatasetResult]], summary="评测集列表")
@require_permission("ai:agent:manage")
async def list_datasets(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_service.list_datasets(db, agent_id)
    return success([EvalDatasetResult.model_validate(d) for d in result])


@router.patch(
    "/datasets/{dataset_id}", response_model=Result[EvalDatasetResult], summary="更新评测集"
)
@require_permission("ai:agent:manage")
async def update_dataset(
    agent_id: int,
    dataset_id: int,
    form: EvalDatasetUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_service.update_dataset(db, dataset_id, form)
    return success(EvalDatasetResult.model_validate(result))


@router.delete("/datasets/{dataset_id}", response_model=Result[None], summary="删除评测集")
@require_permission("ai:agent:manage")
async def delete_dataset(
    agent_id: int,
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await eval_service.delete_dataset(db, dataset_id)
    return success(msg="一切ok")


# ── 样本 ────────────────────────────────────────────────────────


@router.post(
    "/datasets/{dataset_id}/samples",
    response_model=Result[EvalSampleResult],
    summary="创建评测样本",
)
@require_permission("ai:agent:manage")
async def create_sample(
    agent_id: int,
    dataset_id: int,
    form: EvalSampleCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_service.create_sample(db, dataset_id, form)
    return success(EvalSampleResult.model_validate(result))


@router.get(
    "/datasets/{dataset_id}/samples",
    response_model=Result[list[EvalSampleResult]],
    summary="评测样本列表",
)
@require_permission("ai:agent:manage")
async def list_samples(
    agent_id: int,
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_service.list_samples(db, dataset_id)
    return success([EvalSampleResult.model_validate(s) for s in result])


@router.patch(
    "/samples/{sample_id}", response_model=Result[EvalSampleResult], summary="更新评测样本"
)
@require_permission("ai:agent:manage")
async def update_sample(
    agent_id: int,
    sample_id: int,
    form: EvalSampleUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_service.update_sample(db, sample_id, form)
    return success(EvalSampleResult.model_validate(result))


@router.delete("/samples/{sample_id}", response_model=Result[None], summary="删除评测样本")
@require_permission("ai:agent:manage")
async def delete_sample(
    agent_id: int,
    sample_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    await eval_service.delete_sample(db, sample_id)
    return success(msg="一切ok")


# ── 评测执行 ────────────────────────────────────────────────────


@router.post("/runs", response_model=Result[dict], summary="手动触发评测（回归集）")
@require_permission("ai:agent:manage")
async def run_manual_eval(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
    redis=Depends(get_redis_client),
):
    result = await eval_service.run_regression(db, redis, agent_id, trigger_type="manual")
    return success(result)


@router.get("/runs", response_model=Result[PageResult[EvalRunResult]], summary="评测执行记录")
@require_permission("ai:agent:manage")
async def list_runs(
    agent_id: int,
    pageNum: int = Query(default=1, ge=1),
    pageSize: int = Query(default=10, ge=1, le=100),
    datasetId: int | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result, total = await eval_service.list_runs(
        db, agent_id, pageNum, pageSize, dataset_id=datasetId
    )
    return success(PageResult(list=[EvalRunResult.model_validate(r) for r in result], total=total))
