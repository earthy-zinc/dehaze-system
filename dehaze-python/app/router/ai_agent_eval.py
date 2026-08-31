"""智能体评测路由（AI对话 - 智能体管理）

端点前缀 /api/v1/ai/agents/{agent_id}/eval：
- 评测集 CRUD、样本 CRUD、评测执行记录查询
- 手动触发评测（trigger_type=manual）

端点前缀 /api/v1/ai/eval-center（评测中心，跨 Agent 聚合）：
- 评测总览 / 历史趋势 / 两次 run 对比 / 判分状态 / 人工复核

权限：ai:agent:manage
"""

from datetime import datetime

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
from app.models.schema.ai_eval_center import (
    EvalAgentOverviewItem,
    EvalReviewQueueResult,
    EvalReviewSubmitForm,
    EvalRunCompareResult,
    EvalRunGateResult,
    EvalTrendItem,
    JudgeStatusResult,
)
from app.models.schema.common import PageResult
from app.service.ai_eval_center_service import eval_center_service
from app.service.ai_eval_service import eval_service

router = APIRouter(prefix="/api/v1/ai/agents/{agent_id}/eval", tags=["AI对话-智能体评测"])
center_router = APIRouter(prefix="/api/v1/ai/eval-center", tags=["AI对话-评测中心"])


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


@router.post("/runs", response_model=Result[EvalRunGateResult], summary="手动触发评测（回归集）")
@require_permission("ai:agent:manage")
async def run_manual_eval(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
    redis=Depends(get_redis_client),
):
    result = await eval_service.run_regression(db, redis, agent_id, trigger_type="manual")
    return success(EvalRunGateResult.model_validate(result))


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


# ── 评测中心（跨 Agent 聚合） ────────────────────────────────────


@center_router.get(
    "/overview",
    response_model=Result[list[EvalAgentOverviewItem]],
    summary="评测总览（各 Agent 最近得分/门禁状态/退化标识）",
)
@require_permission("ai:agent:manage")
async def eval_overview(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success([EvalAgentOverviewItem.model_validate(i) for i in await eval_center_service.overview(db)])


@center_router.get(
    "/trends",
    response_model=Result[list[EvalTrendItem]],
    summary="评测历史趋势（按 Agent/时间范围过滤）",
)
@require_permission("ai:agent:manage")
async def eval_trends(
    agentId: int | None = Query(default=None),
    startTime: datetime | None = Query(default=None),
    endTime: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    items = await eval_center_service.trends(
        db, agent_id=agentId, start_time=startTime, end_time=endTime, limit=limit
    )
    return success([EvalTrendItem.model_validate(i) for i in items])


@center_router.get(
    "/runs/{run_id}/compare",
    response_model=Result[EvalRunCompareResult],
    summary="两次评测 run 得分对比（四维差异 + 样本级差异）",
)
@require_permission("ai:agent:manage")
async def eval_run_compare(
    run_id: int,
    baseRunId: int = Query(..., description="基准评测记录ID"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(EvalRunCompareResult.model_validate(
        await eval_center_service.compare_runs(db, run_id, baseRunId)
    ))


@center_router.get(
    "/judge-status",
    response_model=Result[JudgeStatusResult],
    summary="判分模型状态（一致性/漂移/门禁暂停提示）",
)
@require_permission("ai:agent:manage")
async def eval_judge_status(
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    return success(JudgeStatusResult.model_validate(await eval_center_service.judge_status(db)))


@center_router.get(
    "/reviews",
    response_model=Result[EvalReviewQueueResult],
    summary="人工复核队列（失败样本全量 + 通过样本按比例抽样）",
)
@require_permission("ai:agent:manage")
async def eval_reviews(
    status: int | None = Query(default=None, ge=1, le=2, description="复核状态过滤(1:待复核;2:已复核)"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_center_service.list_reviews(db, status)
    return success(EvalReviewQueueResult.model_validate(result))


@center_router.post(
    "/reviews/{review_id}",
    response_model=Result[dict],
    summary="复核结果回填（判定一致/不一致 + 备注）",
)
@require_permission("ai:agent:manage")
async def eval_review_submit(
    review_id: int,
    form: EvalReviewSubmitForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    result = await eval_center_service.submit_review(
        db, review_id, form.agree, form.remark, user.id
    )
    return success(result)
