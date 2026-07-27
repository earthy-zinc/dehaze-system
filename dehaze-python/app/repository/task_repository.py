"""
任务数据访问层
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import get_audit_update_values
from app.models.entity.sys_task import SysTask
from app.models.enum.task_enum import TaskStatus
from app.repository.base import BaseRepository


class TaskRepository(BaseRepository[SysTask]):
    """任务数据访问层"""

    model = SysTask

    async def get_by_task_id(
        self,
        db: AsyncSession,
        task_id: str,
    ) -> SysTask | None:
        """根据任务 UUID 查询任务"""
        stmt = select(SysTask).where(SysTask.task_id == task_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(
        self,
        db: AsyncSession,
        idempotency_key: str,
    ) -> SysTask | None:
        """根据客户端幂等键查询任务（用于幂等去重）"""
        stmt = select(SysTask).where(SysTask.idempotency_key == idempotency_key)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_retry_count(
        self,
        db: AsyncSession,
        task_id: str,
        retry_count: int,
        worker_id: Optional[str] = None,
    ) -> int:
        """更新任务重试次数和 Worker 标识"""
        values: dict = {"retry_count": retry_count}
        if worker_id is not None:
            values["worker_id"] = worker_id
        values.update(get_audit_update_values())
        stmt = (
            update(SysTask)
            .where(SysTask.task_id == task_id)
            .values(**values)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount

    async def update_progress(
        self,
        db: AsyncSession,
        task_id: str,
        progress: int,
        processed_files: int,
        total_files: int,
    ) -> int:
        """更新任务进度"""
        values = {
            "progress": progress,
            "processed_files": processed_files,
            "total_files": total_files,
        }
        values.update(get_audit_update_values())
        stmt = (
            update(SysTask)
            .where(SysTask.task_id == task_id)
            .values(**values)
        )
        result = await db.execute(stmt)
        await db.flush()
        return result.rowcount

    async def get_user_tasks_paginated(
        self,
        db: AsyncSession,
        user_id: int,
        status: Optional[int] = None,
        task_type: Optional[str] = None,
        task_category: Optional[str] = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysTask], int]:
        """获取用户的任务列表（分页+筛选）"""
        stmt = select(SysTask).where(SysTask.create_by == user_id)

        if status:
            stmt = stmt.where(SysTask.status == status)
        if task_type:
            stmt = stmt.where(SysTask.task_type == task_type)
        if task_category:
            if task_category == 'import':
                stmt = stmt.where(SysTask.task_type.like('%_import'))
            elif task_category == 'export':
                stmt = stmt.where(SysTask.task_type.like('%_export'))

        stmt = stmt.order_by(SysTask.create_time.desc())
        return await self.paginate(db, stmt, page, size)

    async def get_terminated_task_ids(
        self,
        db: AsyncSession,
        before: datetime,
    ) -> list[str]:
        """获取指定时间之前已终止（非 pending/processing）的任务 ID 列表"""
        stmt = select(SysTask.task_id).where(
            and_(
                SysTask.status.not_in([
                    TaskStatus.PENDING.value,
                    TaskStatus.PROCESSING.value,
                ]),
                SysTask.create_time < before,
            )
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]


# 单例
task_repository = TaskRepository()
