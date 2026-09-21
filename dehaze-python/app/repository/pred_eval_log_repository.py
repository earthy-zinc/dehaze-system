"""
预测日志 / 评估日志 Repository
"""

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

        stmt = select(func.count(SysPredLog.id)).where(SysPredLog.algorithm_id == algorithm_id)
        return (await db.execute(stmt)).scalar() or 0

    async def list_recent_pred_urls(
        self, db: AsyncSession, algorithm_id: int, limit: int = 3
    ) -> list[str]:
        """算法最近成功预测的结果图 URL（算法选择详情样例图）"""
        stmt = (
            select(SysPredLog.pred_url)
            .where(
                SysPredLog.algorithm_id == algorithm_id,
                SysPredLog.status == LogStatus.COMPLETED.value,
            )
            .order_by(desc(SysPredLog.id))
            .limit(limit)
        )
        result = await db.execute(stmt)
        return [url for url in result.scalars().all() if url]

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
        recommended_by: int | None = None,
    ) -> SysPredLog:
        """创建 processing 状态的预测日志，返回 log_id 供异步任务更新"""
        log = SysPredLog(
            algorithm_id=algorithm_id,
            origin_file_id=origin_file_id,
            origin_md5=origin_md5,
            origin_url=origin_url,
            recommended_by=recommended_by,
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
    ) -> bool:
        """更新预测日志为 completed 并写入结果。

        仅允许 processing → completed 流转；任务已被取消（终态）时返回 False，
        防止后台推理完成后覆盖已取消状态。
        """
        values = {
            "status": LogStatus.COMPLETED.value,
            "pred_md5": pred_md5,
            "pred_url": pred_url,
            "time": time_ms // 1000,
        }
        if pred_file_id is not None:
            values["pred_file_id"] = pred_file_id
        stmt = (
            update(SysPredLog)
            .where(SysPredLog.id == log_id, SysPredLog.status == LogStatus.PROCESSING.value)
            .values(**values)
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

    async def update_status(
        self,
        db: AsyncSession,
        log_id: int,
        status: int,
        error_message: str,
        time_ms: int,
    ) -> bool:
        """将预测日志从 processing 流转到指定终态（failed/cancelled），返回是否流转成功。

        仅允许 processing 起始的流转，保证终态不可被并发写入覆盖，
        并以此作为配额回滚的防重依据（两个并发取消只有一个生效）。
        """
        stmt = (
            update(SysPredLog)
            .where(
                SysPredLog.id == log_id,
                SysPredLog.status == LogStatus.PROCESSING.value,
                SysPredLog.status != status,
            )
            .values(
                status=status,
                error_message=error_message,
                time=time_ms // 1000,
            )
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0

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
        recommended_by: int | None = None,
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
            recommended_by=recommended_by,
            time=time_ms // 1000,
            status=LogStatus.COMPLETED.value,
        )
        return await self.create(db, log)

    async def get_paginated(
        self,
        db: AsyncSession,
        algorithm_id: int | None = None,
        user_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysPredLog], int]:
        """分页查询预测日志（user_id 提供时仅返回该用户的日志）"""
        stmt = select(SysPredLog)
        if algorithm_id is not None:
            stmt = stmt.where(SysPredLog.algorithm_id == algorithm_id)
        if user_id is not None:
            stmt = stmt.where(SysPredLog.create_by == user_id)
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

    async def count_recommended(self, db: AsyncSession, start, end) -> int:
        """推荐采纳数：带推荐来源（recommended_by 非空）的预测记录数"""
        from sqlalchemy import func

        stmt = select(func.count()).where(SysPredLog.recommended_by.isnot(None))
        if start is not None:
            stmt = stmt.where(SysPredLog.create_time >= start)
        if end is not None:
            stmt = stmt.where(SysPredLog.create_time <= end)
        return (await db.execute(stmt)).scalar() or 0

    async def select_daily_recommended(self, db: AsyncSession, start, end) -> list[dict]:
        """按日统计推荐采纳数（日期 + 带推荐来源的预测记录数）"""
        from sqlalchemy import func

        stmt = select(
            func.date(SysPredLog.create_time).label("date"),
            func.count().label("count"),
        ).where(SysPredLog.recommended_by.isnot(None))
        if start is not None:
            stmt = stmt.where(SysPredLog.create_time >= start)
        if end is not None:
            stmt = stmt.where(SysPredLog.create_time <= end)
        stmt = stmt.group_by(func.date(SysPredLog.create_time)).order_by(
            func.date(SysPredLog.create_time)
        )
        result = await db.execute(stmt)
        return [{"date": str(row[0]), "count": int(row[1] or 0)} for row in result.all()]


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
            task_type="evaluation",
        )
        return await self.create(db, log)

    async def update_result(
        self,
        db: AsyncSession,
        log_id: int,
        result: dict,
        time_ms: int,
    ) -> None:
        """更新评估日志为 completed 并写入结果

        result 为 JSON 类型列，SQLAlchemy 绑定时自动序列化；
        此处不能手动 json.dumps（会造成双重编码，读取时得到字符串标量）。
        """
        stmt = (
            update(SysEvalLog)
            .where(SysEvalLog.id == log_id)
            .values(
                status=LogStatus.COMPLETED.value,
                result=result,
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
        user_id: int,
        algorithm_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询指定用户的评估日志"""
        stmt = select(SysEvalLog).where(SysEvalLog.create_by == user_id)
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
