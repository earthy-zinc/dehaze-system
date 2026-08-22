"""AI 定时调度执行历史仓库（F-M08-009）

只追加日志表，不逻辑删除；保留 30 天由 cleanup_before 物理清理。
幂等防重入依赖 uk_schedule_window(schedule_id, window_start) 唯一约束，
服务重启/多实例并发扫描/时钟漂移均不产生重复执行。
"""

from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_schedule_run import SysAiScheduleRun
from app.repository.base import BaseRepository


class AiScheduleRunRepository(BaseRepository[SysAiScheduleRun]):
    model = SysAiScheduleRun

    async def create_with_window(
        self,
        db: AsyncSession,
        entity: SysAiScheduleRun,
    ) -> SysAiScheduleRun | None:
        """幂等插入执行批次。

        依赖 uk_schedule_window(schedule_id, window_start) 唯一约束：同窗口已存在记录时
        捕获 IntegrityError，返回已存在记录或 None，供调用方记 skip_reason=idempotent。

        用 SAVEPOINT（begin_nested）隔离回滚：IntegrityError 仅回滚本次插入，
        不级联回滚调用方外层事务中已有的未提交变更。
        """
        try:
            async with db.begin_nested():
                db.add(entity)
                await db.flush()
            return entity
        except IntegrityError:
            existing = await self.get_by_window(db, entity.schedule_id, entity.window_start)
            return existing

    async def get_by_window(
        self,
        db: AsyncSession,
        schedule_id: int,
        window_start: datetime,
    ) -> SysAiScheduleRun | None:
        """按幂等键查询同窗口记录。"""
        stmt = select(SysAiScheduleRun).where(
            SysAiScheduleRun.schedule_id == schedule_id,
            SysAiScheduleRun.window_start == window_start,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_stale_running(
        self,
        db: AsyncSession,
        before: datetime,
        limit: int = 200,
    ) -> list[SysAiScheduleRun]:
        """查询执行中超时未完成的僵尸批次（进程崩溃残留）。

        status=0 且 create_time 早于 before，供扫描任务回收为失败。
        """
        stmt = (
            select(SysAiScheduleRun)
            .where(SysAiScheduleRun.status == 0, SysAiScheduleRun.create_time < before)
            .order_by(SysAiScheduleRun.id)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_running_window(
        self,
        db: AsyncSession,
        schedule_id: int,
        window_start: datetime,
    ) -> SysAiScheduleRun | None:
        """查同任务当前窗口是否有执行中（status=0）的执行记录。

        供并发控制（防任务重叠）使用：存在则说明上一次执行尚未结束，本次触发应跳过。
        """
        stmt = select(SysAiScheduleRun).where(
            SysAiScheduleRun.schedule_id == schedule_id,
            SysAiScheduleRun.window_start == window_start,
            SysAiScheduleRun.status == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_latest_by_schedule_ids(
        self,
        db: AsyncSession,
        schedule_ids: list[int],
    ) -> dict[int, SysAiScheduleRun]:
        """批量取各任务最近一条执行记录（列表摘要聚合，避免 N+1）。

        Returns:
            {schedule_id: 最近一次执行记录}
        """
        if not schedule_ids:
            return {}
        max_ids = (
            select(func.max(SysAiScheduleRun.id))
            .where(SysAiScheduleRun.schedule_id.in_(schedule_ids))
            .group_by(SysAiScheduleRun.schedule_id)
            .scalar_subquery()
        )
        stmt = select(SysAiScheduleRun).where(SysAiScheduleRun.id.in_(max_ids))
        result = await db.execute(stmt)
        return {run.schedule_id: run for run in result.scalars().all()}

    async def page_by_schedule(
        self,
        db: AsyncSession,
        schedule_id: int,
        page: int,
        size: int,
    ) -> tuple[list[SysAiScheduleRun], int]:
        """执行历史分页（按创建时间倒序）。"""
        stmt = (
            select(SysAiScheduleRun)
            .where(SysAiScheduleRun.schedule_id == schedule_id)
            .order_by(SysAiScheduleRun.create_time.desc(), SysAiScheduleRun.id.desc())
        )
        return await self.paginate(db, stmt, page, size)

    async def cleanup_before(self, db: AsyncSession, before: datetime) -> int:
        """物理清理早于指定时间的执行历史（保留 30 天由定时任务调用）。"""
        stmt = delete(SysAiScheduleRun).where(SysAiScheduleRun.create_time < before)
        result = await db.execute(stmt)
        return result.rowcount


ai_schedule_run_repository = AiScheduleRunRepository()
