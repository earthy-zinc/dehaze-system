from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.database import get_db
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.enum.task_enum import EXPORT_TASK_TYPES, TaskStatus
from app.models.schema.task import ExportTaskCreateForm as ExportTaskCreateRequest
from app.models.schema.task import TaskPageVO
from app.models.schema.task import TaskVO as TaskData
from app.service.task import task_service
from app.service.file_service import file_service

router = APIRouter(prefix="/api/v1/tasks", tags=["任务管理"])


def _dict_to_task_data(task_data: dict, request: Request) -> TaskData:
    """将任务字典转换为 TaskData VO"""
    download_url = None
    if (
        task_data.get("status") == TaskStatus.COMPLETED.value
        and task_data.get("task_type") in EXPORT_TASK_TYPES
        and task_data.get("result")
    ):
        download_url = (
            f"{str(request.base_url).rstrip('/')}/api/v1/tasks/{task_data['task_id']}/download"
        )
    return TaskData(
        id=task_data["id"],
        taskId=task_data["task_id"],
        taskType=task_data["task_type"],
        status=task_data["status"],
        progress=task_data["progress"],
        totalFiles=task_data.get("total_files", 0),
        processedFiles=task_data.get("processed_files", 0),
        downloadUrl=download_url,
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
    request: Request,
    status_filter: int | None = Query(
        default=None,
        alias="status",
        description="状态筛选(1:待处理;2:处理中;3:已完成;4:失败;5:已取消)",
    ),
    task_type: str | None = Query(default=None, alias="taskType", description="类型筛选"),
    task_category: str | None = Query(
        default=None, alias="taskCategory", description="任务类别筛选(import/export)"
    ),
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),
):
    """
    查询当前用户的任务列表（分页+状态筛选）

    - **status**: 状态筛选（1:待处理/2:处理中/3:已完成/4:失败/5:已取消）
    - **taskType**: 类型筛选
    - **taskCategory**: 类别筛选（import/export）
    - **pageNum**: 页码（从1开始）
    - **pageSize**: 每页数量（最大100）
    """
    result_data = await task_service.list_tasks(
        db=db,
        user_id=user.id,
        status=status_filter,
        task_type=task_type,
        task_category=task_category,
        page=pageNum,
        size=pageSize,
    )
    return success(
        TaskPageVO(
            list=[_dict_to_task_data(t, request) for t in result_data["list"]],
            total=result_data["total"],
        )
    )


@router.post("", response_model=Result[TaskData], summary="创建任务")
async def create_export_task(
    form: ExportTaskCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        description="客户端幂等键，相同键返回已有任务，防止重复创建",
    ),
):
    """
    创建新的任务（由通用导入导出框架内部调用，通常不直接暴露给前端）

    - **type**: 任务类型（user_export/role_export/.../user_import/...）
    - **paramsJson**: 任务参数（JSON 字符串）
    - **Idempotency-Key** (请求头): 客户端幂等键，相同键返回已有任务
    """
    task_data = await task_service.create_task(
        db=db,
        redis=redis,
        task_type=form.type.value,
        params_json=form.params_json,
        user_id=user.id,
        idempotency_key=idempotency_key,
    )

    return success(_dict_to_task_data(task_data, request))


@router.get(
    "/{task_id}",
    response_model=Result[TaskData],
    summary="查询任务状态",
)
async def get_task_status(
    task_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """
    根据任务ID查询任务执行状态和进度

    - **task_id**: 任务ID（UUID格式）
    """
    task_data = await task_service.get_task_status(db, redis, task_id, user_id=user.id)

    if task_data is None:
        raise BusinessException(ResultCode.TASK_NOT_FOUND, f"任务不存在: {task_id}")

    return success(_dict_to_task_data(task_data, request))


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
    下载已完成的导出任务文件（从存储后端流式返回）

    - **task_id**: 任务ID（UUID格式）
    """
    object_name = await task_service.get_export_object_name(db, redis, task_id, user_id=user.id)

    if object_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务未完成、已过期或下载链接不存在",
        )

    return file_service.stream_file_response(object_name, storage="minio")


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
    await task_service.cancel_task(db, redis, task_id, user_id=user.id)
    return success(msg="取消成功")


@router.post(
    "/{task_id}/retry",
    response_model=Result[TaskData],
    summary="重试失败的任务",
)
async def retry_task(
    task_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    """
    重试失败的任务（重放入口）

    仅允许 FAILED 状态的任务重试，重试次数 +1 后重新投递到消息队列。

    - **task_id**: 任务ID（UUID格式）
    """
    task_data = await task_service.retry_task(db, redis, task_id, user_id=user.id)
    return success(_dict_to_task_data(task_data, request))
