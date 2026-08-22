"""
预测日志 / 评估日志 Repository
"""

import json
from datetime import datetime

from sqlalchemy import desc, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_log import SysEvalLog, SysPredLog
from app.models.enum.log_status import LogStatus
from app.repository.base import BaseRepository


class PredLogRepository(BaseRepository[SysPredLog]):
    """预测日志 Repository"""

    model = SysPredLog

    async def count_by_algorithm(self, db: AsyncSession, algorithm_id: int) -> int:
        """算法使用次数（预测日志记录数）"""
        from sqlalchemy import func

        stmt = select(func.count(SysPredLog.id)).where(
            SysPredLog.algorithm_id == algorithm_id
        )
        return (await db.execute(stmt)).scalar() or 0

    async def list_ids_by_file(self, db: AsyncSession, file_id: int) -> list[int]:
        """按预测结果/原图文件反查预测日志 id（文件删除联动失效产物用）"""
        stmt = select(SysPredLog.id).where(
            or_(SysPredLog.pred_file_id == file_id, SysPredLog.origin_file_id == file_id)
        )
        result = await db.execute(stmt)
        return [int(row) for row in result.scalars().all()]

    async def create_pending_log(
        self,
        db: AsyncSession,
        algorithm_id: int,
        origin_md5: str,
        origin_url: str,
        origin_file_id: int | None = None,
    ) -> SysPredLog:
        """创建 processing 状态的预测日志，返回 log_id 供异步任务更新"""
        log = SysPredLog(
            algorithm_id=algorithm_id,
            origin_file_id=origin_file_id,
            origin_md5=origin_md5,
            origin_url=origin_url,
            pred_md5="",
            pred_url="",
            time=0,
            status=LogStatus.PROCESSING.value,
        )
        return await self.create(db, log)

    async def update_result(
        self,
        db: AsyncSession,
        log_id: int,
        pred_md5: str,
        pred_url: str,
        time_ms: int,
        pred_file_id: int | None = None,
    ) -> None:
        """更新预测日志为 completed 并写入结果"""
        values = {
            "status": LogStatus.COMPLETED.value,
            "pred_md5": pred_md5,
            "pred_url": pred_url,
            "time": time_ms // 1000,
        }
        if pred_file_id is not None:
            values["pred_file_id"] = pred_file_id
        stmt = update(SysPredLog).where(SysPredLog.id == log_id).values(**values)
        await db.execute(stmt)
        await db.commit()

    async def update_status(
        self,
        db: AsyncSession,
        log_id: int,
        status: int,
        error_message: str,
        time_ms: int,
    ) -> None:
        """更新预测日志状态为 failed 并写入错误信息"""
        stmt = (
            update(SysPredLog)
            .where(SysPredLog.id == log_id)
            .values(
                status=status,
                error_message=error_message,
                time=time_ms // 1000,
            )
        )
        await db.execute(stmt)
        await db.commit()

    async def create_log(
        self,
        db: AsyncSession,
        algorithm_id: int,
        origin_md5: str,
        origin_url: str,
        pred_md5: str,
        pred_url: str,
        time_ms: int,
        origin_file_id: int | None = None,
        pred_file_id: int | None = None,
    ) -> SysPredLog:
        """创建已完成的预测日志（缓存命中场景）"""
        log = SysPredLog(
            algorithm_id=algorithm_id,
            origin_file_id=origin_file_id,
            origin_md5=origin_md5,
            origin_url=origin_url,
            pred_file_id=pred_file_id,
            pred_md5=pred_md5,
            pred_url=pred_url,
            time=time_ms // 1000,
            status=LogStatus.COMPLETED.value,
        )
        return await self.create(db, log)

    async def get_paginated(
        self,
        db: AsyncSession,
        algorithm_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysPredLog], int]:
        """分页查询预测日志"""
        stmt = select(SysPredLog)
        if algorithm_id is not None:
            stmt = stmt.where(SysPredLog.algorithm_id == algorithm_id)
        stmt = stmt.order_by(desc(SysPredLog.id))
        return await self.paginate(db, stmt, page, size)

    async def mark_stuck_as_failed(
        self,
        db: AsyncSession,
        threshold: datetime,
    ) -> int:
        """将超时的 processing 记录标记为 failed（僵尸任务恢复）"""
        stmt = (
            update(SysPredLog)
            .where(
                SysPredLog.status == LogStatus.PROCESSING.value,
                SysPredLog.update_time < threshold,
            )
            .values(
                status=LogStatus.FAILED.value,
                error_message="任务执行超时，服务可能已重启",
            )
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount


class EvalLogRepository(BaseRepository[SysEvalLog]):
    """评估日志 Repository"""

    model = SysEvalLog

    async def list_ids_by_file(self, db: AsyncSession, file_id: int) -> list[int]:
        """按预测结果/真值图文件反查评估日志 id（文件删除联动失效产物用）"""
        stmt = select(SysEvalLog.id).where(
            or_(SysEvalLog.pred_file_id == file_id, SysEvalLog.gt_file_id == file_id)
        )
        result = await db.execute(stmt)
        return [int(row) for row in result.scalars().all()]

    async def create_pending_log(
        self,
        db: AsyncSession,
        algorithm_id: int,
        pred_md5: str,
        pred_url: str,
        gt_md5: str,
        gt_url: str,
        pred_file_id: int | None = None,
        gt_file_id: int | None = None,
    ) -> SysEvalLog:
        """创建 processing 状态的评估日志，返回 log_id 供异步任务更新"""
        log = SysEvalLog(
            algorithm_id=algorithm_id,
            pred_file_id=pred_file_id,
            pred_md5=pred_md5,
            pred_url=pred_url,
            gt_file_id=gt_file_id,
            gt_md5=gt_md5,
            gt_url=gt_url,
            time=0,
            status=LogStatus.PROCESSING.value,
        )
        return await self.create(db, log)

    async def update_result(
        self,
        db: AsyncSession,
        log_id: int,
        result: dict,
        time_ms: int,
    ) -> None:
        """更新评估日志为 completed 并写入结果"""
        stmt = (
            update(SysEvalLog)
            .where(SysEvalLog.id == log_id)
            .values(
                status=LogStatus.COMPLETED.value,
                result=json.dumps(result) if isinstance(result, dict) else result,
                time=time_ms // 1000,
            )
        )
        await db.execute(stmt)
        await db.commit()

    async def update_status(
        self,
        db: AsyncSession,
        log_id: int,
        status: int,
        error_message: str,
        time_ms: int,
    ) -> None:
        """更新评估日志状态为 failed 并写入错误信息"""
        stmt = (
            update(SysEvalLog)
            .where(SysEvalLog.id == log_id)
            .values(
                status=status,
                error_message=error_message,
                time=time_ms // 1000,
            )
        )
        await db.execute(stmt)
        await db.commit()

    async def get_paginated(
        self,
        db: AsyncSession,
        algorithm_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询评估日志"""
        stmt = select(SysEvalLog)
        if algorithm_id is not None:
            stmt = stmt.where(SysEvalLog.algorithm_id == algorithm_id)
        stmt = stmt.order_by(desc(SysEvalLog.id))
        return await self.paginate(db, stmt, page, size)

    async def mark_stuck_as_failed(
        self,
        db: AsyncSession,
        threshold: datetime,
    ) -> int:
        """将超时的 processing 记录标记为 failed（僵尸任务恢复）"""
        stmt = (
            update(SysEvalLog)
            .where(
                SysEvalLog.status == LogStatus.PROCESSING.value,
                SysEvalLog.update_time < threshold,
            )
            .values(
                status=LogStatus.FAILED.value,
                error_message="任务执行超时，服务可能已重启",
            )
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount


pred_log_repository = PredLogRepository()
eval_log_repository = EvalLogRepository()
