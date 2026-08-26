"""AI 定时调度路由（F-M08-009）

对齐 API接口.md §2.8.4 的 9 个端点契约。本模块为用户级操作，无独立权限标识，
通过 get_current_user 注入当前用户并做归属校验。

注意：/next-times 是集合级路径，必须注册在 /{schedule_id} 之前，避免被路径参数吞掉。
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db, get_db_session
from app.dependencies.auth import UserContext, get_current_user
from app.models.schema.ai_schedule import (
    NextTimesPreview,
    RunHistoryItem,
    ScheduleCreate,
    ScheduleDetail,
    ScheduleListItem,
    SchedulePageQuery,
    ScheduleStatusForm,
    ScheduleUpdate,
)
from app.models.schema.common import BasePageQuery, PageResult
from app.service.ai.service.ai_schedule_executor import schedule_executor
from app.service.ai.service.ai_schedule_service import scheduled_task_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/ai/scheduled-tasks",
    tags=["AI定时调度"],
    dependencies=[Depends(get_current_user)],
)

# 持有手动触发后台任务引用，防止任务对象被垃圾回收提前取消
_background_tasks: set[asyncio.Task] = set()


def _track_task(task: asyncio.Task) -> None:
    """登记后台任务引用并在结束时移除，避免 GC 提前取消 fire-and-forget 执行。"""
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@router.post("", response_model=Result[ScheduleDetail], summary="创建定时任务")
async def create_schedule(
    form: ScheduleCreate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """创建定时任务，保存后返回下次触发时间预览。"""
    result = await scheduled_task_service.create(db, user.id, form)
    return success(result)


@router.get("", response_model=Result[PageResult[ScheduleListItem]], summary="定时任务列表")
async def list_schedules(
    query: SchedulePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """任务列表：分页、按下次触发时间排序、含最近执行结果摘要。"""
    result = await scheduled_task_service.list_page(db, user.id, query)
    return success(result)


@router.get(
    "/next-times", response_model=Result[NextTimesPreview], summary="Cron 解释与下次执行时间预览"
)
async def preview_next_times(
    cron: str = Query(..., max_length=64, description="Cron 触发规则(5位表达式)"),
    count: int = Query(5, ge=1, le=20, description="返回的触发时间次数(默认5,最大20)"),
    user: UserContext = Depends(get_current_user),
):
    """Cron 表达式解释：人类可读描述 + 接下来 N 次触发时间。"""
    result = await scheduled_task_service.preview_next_times(cron, count)
    return success(result)


@router.get("/{schedule_id}", response_model=Result[ScheduleDetail], summary="定时任务详情")
async def get_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """任务详情：触发规则/输入输出配置/下次触发时间/熔断状态。"""
    result = await scheduled_task_service.get_detail(db, user.id, schedule_id)
    return success(result)


@router.put("/{schedule_id}", response_model=Result[ScheduleDetail], summary="更新定时任务")
async def update_schedule(
    schedule_id: int,
    form: ScheduleUpdate,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """更新任务（触发规则/输入来源/输出目标），变更后重算下次触发时间。"""
    result = await scheduled_task_service.update(db, user.id, schedule_id, form)
    return success(result)


@router.patch("/{schedule_id}/status", response_model=Result[None], summary="启停定时任务")
async def set_schedule_status(
    schedule_id: int,
    form: ScheduleStatusForm,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """启停任务（{enabled: bool}）；熔断停用后可重新启用（core 层重置连续失败计数）。"""
    await scheduled_task_service.set_enabled(db, user.id, schedule_id, form.enabled)
    return success(msg="一切ok")


@router.delete("/{schedule_id}", response_model=Result[None], summary="删除定时任务")
async def delete_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """删除任务（软删除）。"""
    await scheduled_task_service.delete(db, user.id, schedule_id)
    return success(msg="一切ok")


@router.post(
    "/{schedule_id}/run", response_model=Result[dict[str, bool]], summary="手动触发一次执行"
)
async def run_schedule(
    schedule_id: int,
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """手动触发一次执行（验证配置/补跑），不改变原定时规则。

    无人值守执行可能耗时分钟级，采用后台 fire-and-forget：触发受理即返回
    {accepted: true}，具体执行结果由后台任务写入执行历史并通知用户。
    """
    # 先校验任务归属，越权或不存在立即失败
    await scheduled_task_service.get_detail(db, user.id, schedule_id)

    async def _trigger_once() -> None:
        try:
            # 后台任务使用独立 DB 会话与全局 Redis 客户端，避免随请求关闭
            from app.dependencies.redis import get_redis_client

            async with get_db_session() as bg_db:
                redis_client = await get_redis_client()
                await schedule_executor.trigger_once(
                    bg_db, redis_client, schedule_id, user.id, manual=True
                )
        except Exception as exc:  # noqa: BLE001 后台执行异常不影响受理响应
            logger.error("定时任务手动触发后台执行失败: schedule_id=%s err=%s", schedule_id, exc)

    _track_task(asyncio.create_task(_trigger_once()))
    return success({"accepted": True})


@router.get(
    "/{schedule_id}/history", response_model=Result[PageResult[RunHistoryItem]], summary="执行历史"
)
async def list_run_history(
    schedule_id: int,
    query: BasePageQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """执行历史分页（结果/消耗积分/耗时/失败原因/跳过原因）。"""
    result = await scheduled_task_service.list_history(
        db, user.id, schedule_id, query.pageNum, query.pageSize
    )
    return success(result)
