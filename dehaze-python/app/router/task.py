from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.enum.task_enum import TaskStatus
from app.models.schema.task import \
    ExportTaskCreateForm as ExportTaskCreateRequest
from app.models.schema.task import TaskPageVO
from app.models.schema.task import TaskVO as TaskData
from app.service.task_service import TaskServiceAsync
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/tasks", tags=["任务管理"])


def _dict_to_task_data(task_data: dict) -> TaskData:
    """将任务字典转换为 TaskData VO"""
    return TaskData(
        id=task_data["id"],
        taskId=task_data["task_id"],
        taskType=task_data["task_type"],
        status=task_data["status"],
        progress=task_data["progress"],
        totalFiles=task_data.get("total_files", 0),
        processedFiles=task_data.get("processed_files", 0),
        result=task_data.get("result"),
        downloadUrl=task_data.get("result") if task_data.get(
            "status") == TaskStatus.COMPLETED.value else None,
        error=task_data.get("error_message"),
        createdAt=task_data.get("created_at"),
        startedAt=task_data.get("started_at"),
        completedAt=task_data.get("completed_at"),
        expiresAt=task_data.get("expires_at"),
        idempotencyKey=task_data.get("idempotency_key"),
        retryCount=task_data.get("retry_count", 0),
        workerId=task_data.get("worker_id"),
    )


@router.get("", response_model=Result[TaskPageVO], summary="查询任务列表")
async def list_tasks(
    status_filter: str | None = Query(
        default=None, alias="status", description="状态筛选"),
    task_type: str | None = Query(
        default=None, alias="taskType", description="类型筛选"),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """
    查询当前用户的任务列表（分页+状态筛选）

    - **status**: 状态筛选（pending/processing/completed/failed/cancelled）
    - **taskType**: 类型筛选
    - **pageNum**: 页码（从1开始）
    - **pageSize**: 每页数量（最大100）
    """
    result_data = await TaskServiceAsync.list_tasks(
        db=db,
        user_id=user.id,
        status=status_filter,
        task_type=task_type,
        page=pageNum,
        size=pageSize,
    )
    return success(
        TaskPageVO(
            list=[_dict_to_task_data(t) for t in result_data["list"]],
            total=result_data["total"],
        )
    )


@router.post("", response_model=Result[TaskData], summary="创建任务")
async def create_export_task(
    request: ExportTaskCreateRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
    idempotency_key: str | None = Header(
        default=None, alias="Idempotency-Key",
        description="客户端幂等键，相同键返回已有任务，防止重复创建"
    ),
):
    """
    创建新的导出任务，支持批量导出数据集或数据项

    - **type**: 导出类型
      - dataset_export: 数据集导出
      - item_download: 单个数据项下载
      - batch_download: 批量数据项下载
      - custom_export: 自定义导出
    - **targetId**: 单个导出目标ID
    - **targetIds**: 批量导出目标ID列表
    - **options**: 导出选项
    - **Idempotency-Key** (请求头): 客户端幂等键，相同键返回已有任务
    """
    # 转换 options 为字典
    options_dict = None
    if request.options:
        options_dict = request.options.model_dump()

    task_data = await TaskServiceAsync.create_export_task(
        db=db,
        redis=redis,
        task_type=request.type.value,
        target_id=request.targetId,
        target_ids=request.targetIds,
        options=options_dict,
        user_id=user.id,
        idempotency_key=idempotency_key,
    )

    return success(_dict_to_task_data(task_data))


@router.get(
    "/{task_id}",
    response_model=Result[TaskData],
    summary="查询任务状态",
)
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """
    根据任务ID查询任务执行状态和进度

    - **task_id**: 任务ID（UUID格式）
    """
    task_data = await TaskServiceAsync.get_task_status(db, redis, task_id, user_id=user.id)

    if task_data is None:
        return success(None)

    return success(_dict_to_task_data(task_data))


@router.get(
    "/{task_id}/download",
    summary="下载导出文件",
)
async def download_export_file(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """
    下载已完成的导出任务文件（302重定向到文件存储）

    - **task_id**: 任务ID（UUID格式）

    返回 302 重定向到实际的下载链接
    """
    download_url = await TaskServiceAsync.download_export_file(
        db, redis, task_id, user_id=user.id
    )

    if download_url is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务未完成、已过期或下载链接不存在",
        )

    return RedirectResponse(url=download_url, status_code=302)


@router.post(
    "/{task_id}/cancel",
    response_model=Result[None],
    summary="取消任务",
)
async def cancel_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """
    取消正在执行或等待中的任务

    - **task_id**: 任务ID（UUID格式）
    """
    await TaskServiceAsync.cancel_task(db, redis, task_id, user_id=user.id)
    return success(msg="取消成功")


@router.post(
    "/{task_id}/retry",
    response_model=Result[TaskData],
    summary="重试失败的任务",
)
async def retry_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """
    重试失败的任务（重放入口）

    仅允许 FAILED 状态的任务重试，重置重试次数后重新投递到消息队列。

    - **task_id**: 任务ID（UUID格式）
    """
    task_data = await TaskServiceAsync.retry_task(
        db, redis, task_id, user_id=user.id
    )
    return success(_dict_to_task_data(task_data))
