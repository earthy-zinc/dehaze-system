"""AI 定时调度配置仓库（F-M08-009）

提供到期任务扫描、单用户上限计数、启停、软删、熔断与下次触发时间维护。
逻辑删除字段由全局 do_orm_execute 事件自动过滤（继承 SoftDeleteMixin）。
"""

from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_ai_schedule import SysAiSchedule
from app.repository.base import BaseRepository


class AiScheduleRepository(BaseRepository[SysAiSchedule]):
    model = SysAiSchedule

    async def get_due_tasks(
        self,
        db: AsyncSession,
        now: datetime,
        limit: int,
    ) -> list[SysAiSchedule]:
        """扫描到期任务：启用、正常状态、未删、下次触发时间不晚于当前，按触发时间排序。

        供调度引擎扫描触发使用（服务重启/多实例并发扫描均由执行历史幂等键兜底防重）。
        """
        stmt = (
            select(SysAiSchedule)
            .where(
                SysAiSchedule.enabled == 1,
                SysAiSchedule.status == 1,
                SysAiSchedule.next_trigger_time <= now,
            )
            .order_by(SysAiSchedule.next_trigger_time.asc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count_by_user(self, db: AsyncSession, user_id: int) -> int:
        """统计单用户定时任务数（不含已删，用于创建时上限 20 校验）。"""
        stmt = (
            select(func.count()).select_from(SysAiSchedule).where(SysAiSchedule.user_id == user_id)
        )
        return (await db.execute(stmt)).scalar() or 0

    async def set_enabled(self, db: AsyncSession, schedule_id: int, enabled: int) -> None:
        """仅设置用户启停标志（enabled=0/1），不动任务状态。"""
        stmt = update(SysAiSchedule).where(SysAiSchedule.id == schedule_id).values(enabled=enabled)
        await db.execute(stmt)

    async def soft_delete(self, db: AsyncSession, schedule_id: int) -> int:
        """软删除单个任务（逻辑删除，删除后不可恢复）。"""
        return await self.soft_delete_by_ids(db, [schedule_id])

    async def mark_circuit(self, db: AsyncSession, schedule_id: int) -> None:
        """熔断停用：置 status=2（连续失败达阈值后调用）。"""
        stmt = update(SysAiSchedule).where(SysAiSchedule.id == schedule_id).values(status=2)
        await db.execute(stmt)

    async def reset_circuit(self, db: AsyncSession, schedule_id: int) -> None:
        """重新启用：恢复 status=1 并清零连续失败计数（用户修复后手动启用）。"""
        stmt = (
            update(SysAiSchedule)
            .where(SysAiSchedule.id == schedule_id)
            .values(status=1, circuit_streak=0)
        )
        await db.execute(stmt)

    async def update_next_trigger(
        self,
        db: AsyncSession,
        schedule_id: int,
        next_trigger_time: datetime,
    ) -> None:
        """重算并写入下次触发时间（创建/更新/启用/执行完成后调用）。"""
        stmt = (
            update(SysAiSchedule)
            .where(SysAiSchedule.id == schedule_id)
            .values(next_trigger_time=next_trigger_time)
        )
        await db.execute(stmt)


ai_schedule_repository = AiScheduleRepository()
