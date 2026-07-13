"""
任务服务模块

实现导出任务的异步执行和管理，使用 Redis 存储任务状态
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException, TaskCancelledException
from app.database import get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.metrics.task_metrics import TaskMetricsContext
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus, TaskType
from app.repository.task_repository import task_repository

logger = logging.getLogger(__name__)


class TaskServiceAsync:
    """任务服务类（异步版本）"""

    # Redis 键前缀（与文档对齐）
    TASK_CACHE_PREFIX = "task:cache:"
    TASK_PROGRESS_PREFIX = "task:progress:"
    TASK_CANCEL_PREFIX = "task:cancel:"
    TASK_EXPIRE_HOURS = 24  # 任务缓存 24 小时
    CANCEL_FLAG_TTL = 3600  # 取消标志 TTL 1 小时（与文档对齐）

    # 进度更新频率控制
    _PROGRESS_MIN_INTERVAL = 5.0  # 至少间隔 5 秒
    _PROGRESS_MIN_PERCENT_DELTA = 5  # 至少变化 5%

    @staticmethod
    async def create_export_task(
        db: AsyncSession,
        redis: Redis,
        task_type: str,
        target_id: Optional[int],
        target_ids: Optional[List[int]],
        options: Optional[Dict[str, Any]],
        user_id: int,
    ) -> Dict[str, Any]:
        """
        创建导出任务

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            task_type: 任务类型
            target_id: 单个目标 ID
            target_ids: 批量目标 ID 列表
            options: 导出选项
            user_id: 当前用户 ID

        Returns:
            任务信息字典

        Raises:
            BusinessException: 用户未登录、参数错误或超并发限制
        """
        if user_id is None:
            raise BusinessException("用户未登录")

        # 校验任务类型
        valid_types = [t.value for t in TaskType]
        if task_type not in valid_types:
            raise BusinessException(ResultCode.TASK_TYPE_UNSUPPORTED)

        # 生成任务 ID
        task_id = str(uuid.uuid4())

        # 创建任务实体
        now = datetime.now()
        sys_task = SysTask(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING.value,
            progress=0,
            total_files=0,
            processed_files=0,
            params=json.dumps({
                'type': task_type,
                'targetId': target_id,
                'targetIds': target_ids or [],
                'options': options or {}
            }),
            created_by=user_id,
            created_at=now,
            expires_at=now +
            timedelta(hours=TaskServiceAsync.TASK_EXPIRE_HOURS)
        )

        # 保存任务到数据库
        db.add(sys_task)
        await db.flush()
        await db.refresh(sys_task)
        await db.commit()

        # 缓存任务信息到 Redis
        cache_key = TaskServiceAsync.TASK_CACHE_PREFIX + task_id
        task_dict = TaskServiceAsync._task_to_dict(sys_task)
        await redis.setex(
            cache_key,
            TaskServiceAsync.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

        # 提交异步任务：优先通过 RabbitMQ 发布，不可用时 fallback 到 asyncio.Task
        await TaskServiceAsync._dispatch_task(
            db_task_id=sys_task.id,
            task_id=task_id,
            task_type=task_type,
            target_id=target_id,
            target_ids=target_ids,
            options=options,
            user_id=user_id,
        )

        logger.info(
            f"创建导出任务成功: taskId={task_id}, type={task_type}, userId={user_id}")

        return task_dict

    @staticmethod
    async def get_task_status(
        db: AsyncSession,
        redis: Redis,
        task_id: str,
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        查询任务状态

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            task_id: 任务 ID
            user_id: 当前用户 ID（权限校验）

        Returns:
            任务信息字典，如果任务不存在则返回 None
        """
        if not task_id:
            raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

        cache_key = TaskServiceAsync.TASK_CACHE_PREFIX + task_id

        # 先从 Redis 缓存查询
        cached_task = await redis.get(cache_key)
        if cached_task:
            try:
                task_data = json.loads(cached_task)
                # 权限校验
                if task_data.get("created_by") != user_id:
                    raise BusinessException(ResultCode.TASK_UNAUTHORIZED)
                return task_data
            except (json.JSONDecodeError, Exception) as e:
                if isinstance(e, BusinessException):
                    raise
                logger.warning(f"解析缓存数据失败: {e}")

        # 从数据库查询（使用 repository）
        sys_task = await task_repository.get_by_task_id(db, task_id)

        if sys_task is None:
            return None

        # 权限校验
        if sys_task.created_by != user_id:
            raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

        # 更新缓存
        task_dict = TaskServiceAsync._task_to_dict(sys_task)
        await redis.setex(
            cache_key,
            TaskServiceAsync.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

        return task_dict

    @staticmethod
    async def list_tasks(
        db: AsyncSession,
        user_id: int,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        page: int = 1,
        size: int = 10,
    ) -> Dict[str, Any]:
        """
        查询当前用户的任务列表（分页+筛选）

        Args:
            db: 异步数据库会话
            user_id: 当前用户 ID
            status: 状态筛选
            task_type: 类型筛选
            page: 页码
            size: 每页数量

        Returns:
            分页结果 {"list": [...], "total": N}
        """
        items, total = await task_repository.get_user_tasks_paginated(
            db, user_id, status=status, task_type=task_type, page=page, size=size
        )
        return {
            "list": [TaskServiceAsync._task_to_dict(t) for t in items],
            "total": total,
        }

    @staticmethod
    async def download_export_file(
        db: AsyncSession,
        redis: Redis,
        task_id: str,
        user_id: int,
    ) -> Optional[str]:
        """
        下载导出文件

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            task_id: 任务 ID
            user_id: 当前用户 ID（权限校验）

        Returns:
            下载链接，如果任务未完成或已过期则返回 None
        """
        if not task_id:
            raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

        sys_task = await task_repository.get_by_task_id(db, task_id)
        if sys_task is None:
            raise BusinessException(ResultCode.TASK_NOT_FOUND)

        # 权限校验
        if sys_task.created_by != user_id:
            raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

        # 检查任务状态
        if sys_task.status != TaskStatus.COMPLETED.value:
            raise BusinessException(
                ResultCode.TASK_STATUS_INVALID, "任务未完成，无法下载")

        # 检查任务是否过期
        if sys_task.expires_at and sys_task.expires_at < datetime.now():
            raise BusinessException(
                ResultCode.TASK_STATUS_INVALID, "任务已过期，无法下载")

        # 从 result 字段获取下载链接
        if not sys_task.result:
            logger.warning(f"任务结果为空: taskId={task_id}")
            return None

        return sys_task.result

    @staticmethod
    async def cancel_task(
        db: AsyncSession,
        redis: Redis,
        task_id: str,
        user_id: int,
    ) -> bool:
        """
        取消导出任务

        Args:
            db: 异步数据库会话
            redis: Redis 异步客户端
            task_id: 任务 ID
            user_id: 当前用户 ID（权限校验）

        Returns:
            是否取消成功
        """
        if not task_id:
            raise BusinessException(ResultCode.TASK_PARAM_ERROR, "任务ID不能为空")

        sys_task = await task_repository.get_by_task_id(db, task_id)
        if sys_task is None:
            raise BusinessException(ResultCode.TASK_NOT_FOUND)

        # 权限校验
        if sys_task.created_by != user_id:
            raise BusinessException(ResultCode.TASK_UNAUTHORIZED)

        # 检查任务状态
        if sys_task.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
            raise BusinessException(
                ResultCode.TASK_STATUS_INVALID, "任务已完成或失败，无法取消")

        if sys_task.status == TaskStatus.CANCELLED.value:
            return True

        # 更新任务状态
        sys_task.status = TaskStatus.CANCELLED.value
        sys_task.completed_at = datetime.now()

        # 更新缓存
        cache_key = TaskServiceAsync.TASK_CACHE_PREFIX + task_id
        task_dict = TaskServiceAsync._task_to_dict(sys_task)
        await redis.setex(
            cache_key,
            TaskServiceAsync.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

        # 设置取消标志位（通知执行器停止），TTL = 1 小时
        cancel_key = TaskServiceAsync.TASK_CANCEL_PREFIX + task_id
        await redis.setex(cancel_key, TaskServiceAsync.CANCEL_FLAG_TTL, 'true')

        await db.commit()
        logger.info(f"取消导出任务成功: taskId={task_id}")
        return True

    @staticmethod
    async def _dispatch_task(
        db_task_id: int,
        task_id: str,
        task_type: str,
        target_id: Optional[int],
        target_ids: Optional[List[int]],
        options: Optional[Dict[str, Any]],
        user_id: int,
    ) -> None:
        """
        分发异步任务：优先通过 RabbitMQ 发布，不可用时 fallback 到 asyncio.Task

        MQ 模式：消息持久化 + ACK 确认 + 死信队列，进程崩溃后可恢复
        Fallback 模式：asyncio.Task + TaskTracker，进程崩溃则任务丢失
        """
        from app.infrastructure.mq.connection import get_publisher

        publisher = get_publisher()

        if publisher is not None and publisher.is_connected:
            # MQ 模式：发布消息到 task.execute 队列
            try:
                await publisher.publish(
                    routing_key="task.execute",
                    body={
                        "db_task_id": db_task_id,
                        "task_id": task_id,
                        "task_type": task_type,
                        "target_id": target_id,
                        "target_ids": target_ids or [],
                        "options": options or {},
                    },
                )
                logger.info(f"任务已发布到 RabbitMQ: taskId={task_id}")
                return
            except Exception as e:
                logger.warning(f"RabbitMQ 发布失败，降级为本地执行: {e}")

        # Fallback：asyncio.Task + TaskTracker
        background_task = asyncio.create_task(
            TaskServiceAsync._execute_export_task_background(
                db_task_id, task_id, task_type, target_id, target_ids, options
            )
        )

        try:
            from app.service.task_tracker import get_task_tracker
            tracker = get_task_tracker()
            await tracker.register(
                task_id=task_id,
                task=background_task,
                task_type=task_type,
                metadata={
                    "db_task_id": db_task_id,
                    "user_id": user_id,
                },
            )
        except Exception as e:
            logger.warning(f"任务追踪注册失败（不影响执行）: {e}")

    # ==================== 私有方法 ====================

    @staticmethod
    def _task_to_dict(task: SysTask) -> Dict[str, Any]:
        """将任务实体转换为字典"""
        return {
            'id': task.id,
            'task_id': task.task_id,
            'task_type': task.task_type,
            'status': task.status,
            'progress': task.progress,
            'total_files': task.total_files,
            'processed_files': task.processed_files,
            'result': task.result,
            'error_message': task.error_message,
            'created_by': task.created_by,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'expires_at': task.expires_at.isoformat() if task.expires_at else None
        }

    @staticmethod
    async def _update_task_status(
        db: AsyncSession,
        redis: Redis,
        db_task_id: int,
        status: str,
        result: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """更新任务状态"""
        sys_task = await task_repository.get_by_id(db, db_task_id)

        if sys_task:
            sys_task.status = status
            if result:
                sys_task.result = result
            if error_message:
                sys_task.error_message = error_message
            if status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
                sys_task.completed_at = datetime.now()
            await db.commit()

            # 更新缓存
            await TaskServiceAsync._update_cache(redis, sys_task)

    @staticmethod
    async def _update_task_progress(
        db: AsyncSession,
        redis: Redis,
        db_task_id: int,
        processed_files: int,
        total_files: int,
        *,
        _last_update: Dict[str, Any] | None = None,
    ) -> None:
        """更新任务进度（带频率控制）"""
        progress = int((processed_files * 100 / total_files)
                       ) if total_files > 0 else 100

        # 频率控制：至少间隔 5 秒或进度变化 5%（P2-06）
        if _last_update is not None:
            elapsed = time.monotonic() - _last_update.get("time", 0)
            last_progress = _last_update.get("progress", 0)
            if (
                elapsed < TaskServiceAsync._PROGRESS_MIN_INTERVAL
                and abs(progress - last_progress) < TaskServiceAsync._PROGRESS_MIN_PERCENT_DELTA
                and progress < 100
            ):
                return

        sys_task = await task_repository.get_by_id(db, db_task_id)

        if sys_task:
            await task_repository.update_progress(
                db, sys_task.task_id, progress, processed_files, total_files
            )
            sys_task.progress = progress
            sys_task.processed_files = processed_files
            await db.commit()

            # 更新独立进度缓存（P1-05）
            progress_key = TaskServiceAsync.TASK_PROGRESS_PREFIX + sys_task.task_id
            await redis.setex(
                progress_key,
                TaskServiceAsync.TASK_EXPIRE_HOURS * 3600,
                json.dumps(
                    {"progress": progress, "processed": processed_files, "total": total_files})
            )

            # 更新主缓存
            await TaskServiceAsync._update_cache(redis, sys_task)

            # 更新频率控制状态
            if _last_update is not None:
                _last_update["time"] = time.monotonic()
                _last_update["progress"] = progress

    @staticmethod
    async def _update_cache(redis: Redis, sys_task: SysTask) -> None:
        """更新缓存"""
        cache_key = TaskServiceAsync.TASK_CACHE_PREFIX + sys_task.task_id
        task_dict = TaskServiceAsync._task_to_dict(sys_task)
        await redis.setex(
            cache_key,
            TaskServiceAsync.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

    @staticmethod
    async def _is_task_cancelled(redis: Redis, task_id: str) -> bool:
        """检查任务是否已被取消"""
        cancel_key = TaskServiceAsync.TASK_CANCEL_PREFIX + task_id
        is_cancelled = await redis.get(cancel_key)
        if isinstance(is_cancelled, bytes):
            is_cancelled = is_cancelled.decode('utf-8')
        return is_cancelled == 'true'

    # ==================== 后台任务执行 ====================

    @staticmethod
    async def _execute_export_task_background(
        db_task_id: int,
        task_id: str,
        task_type: str,
        target_id: Optional[int],
        target_ids: Optional[List[int]],
        options: Optional[Dict[str, Any]],
    ) -> None:
        """
        后台执行导出任务

        Args:
            db_task_id: 数据库任务 ID
            task_id: 任务 UUID
            task_type: 任务类型
            target_id: 单个目标 ID
            target_ids: 批量目标 ID 列表
            options: 导出选项
        """
        redis = await get_redis_client()
        metrics_enabled = settings.PROMETHEUS_ENABLED

        try:
            async with get_db_session() as db:
                logger.info(f"开始执行导出任务: taskId={task_id}, type={task_type}")

                # 查询任务（使用 repository）
                sys_task = await task_repository.get_by_id(db, db_task_id)

                if sys_task is None:
                    logger.error(f"任务不存在: taskId={task_id}")
                    return

                try:
                    # 更新任务状态为 processing
                    sys_task.status = TaskStatus.PROCESSING.value
                    sys_task.started_at = datetime.now()
                    await db.commit()
                    await TaskServiceAsync._update_cache(redis, sys_task)

                    # 使用 TaskMetricsContext 自动管理指标
                    metrics_cm = TaskMetricsContext(
                        task_type) if metrics_enabled else contextlib.AsyncExitStack()
                    async with metrics_cm as metrics_ctx:

                        # 委托给策略执行器
                        from app.service.task.factory import \
                            TaskStrategyFactory
                        strategy = TaskStrategyFactory.get_strategy(task_type)

                        # 进度频率控制状态
                        progress_state = {"time": 0.0, "progress": 0}

                        async def progress_callback(processed: int, total: int) -> None:
                            """进度回调（带频率控制）"""
                            await TaskServiceAsync._update_task_progress(
                                db, redis, db_task_id, processed, total,
                                _last_update=progress_state,
                            )

                        async def cancel_checker() -> bool:
                            """取消检测回调"""
                            return await TaskServiceAsync._is_task_cancelled(redis, task_id)

                        download_url = await strategy.execute(
                            db=db,
                            sys_task=sys_task,
                            target_id=target_id,
                            target_ids=target_ids,
                            options=options or {},
                            progress_callback=progress_callback,
                            cancel_checker=cancel_checker,
                        )

                        if download_url:
                            await TaskServiceAsync._update_task_status(
                                db, redis, db_task_id, TaskStatus.COMPLETED.value, download_url, None
                            )
                            logger.info(
                                f"导出任务完成: taskId={task_id}, downloadUrl={download_url}")
                        else:
                            if metrics_enabled and isinstance(metrics_ctx, TaskMetricsContext):
                                metrics_ctx.set_status("failed")
                            await TaskServiceAsync._update_task_status(
                                db, redis, db_task_id, TaskStatus.FAILED.value, None, "导出失败"
                            )

                except asyncio.CancelledError:
                    # 任务被取消（优雅关闭时）
                    logger.warning(f"导出任务被取消（服务关闭）: taskId={task_id}")
                    await TaskServiceAsync._update_task_status(
                        db, redis, db_task_id, TaskStatus.FAILED.value, None, "服务关闭，任务中断"
                    )
                    raise

                except TaskCancelledException:
                    logger.warning(f"导出任务被取消: taskId={task_id}")
                    await TaskServiceAsync._update_task_status(
                        db, redis, db_task_id, TaskStatus.CANCELLED.value, None, None
                    )

                except Exception as e:
                    logger.error(f"导出任务执行失败: taskId={task_id}", exc_info=True)
                    await TaskServiceAsync._update_task_status(
                        db, redis, db_task_id, TaskStatus.FAILED.value, None, str(
                            e)
                    )

        except Exception as e:
            logger.error(f"后台任务执行异常: {e}", exc_info=True)
