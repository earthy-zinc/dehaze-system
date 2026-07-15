"""
预测日志 / 评估日志 Repository
"""

from typing import Optional

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.sys_log import SysPredLog, SysEvalLog
from app.repository.base import BaseRepository


class PredLogRepository(BaseRepository[SysPredLog]):
    """预测日志 Repository"""

    model = SysPredLog

    async def create_log(
        self,
        db: AsyncSession,
        algorithm_id: int,
        origin_md5: str,
        origin_url: str,
        pred_md5: str,
        pred_url: str,
        time_ms: int,
        origin_file_id: Optional[int] = None,
        pred_file_id: Optional[int] = None,
    ) -> SysPredLog:
        """创建预测日志"""
        log = SysPredLog(
            algorithm_id=algorithm_id,
            origin_file_id=origin_file_id,
            origin_md5=origin_md5,
            origin_url=origin_url,
            pred_file_id=pred_file_id,
            pred_md5=pred_md5,
            pred_url=pred_url,
            time=time_ms // 1000,
        )
        return await self.create(db, log)

    async def get_paginated(
        self,
        db: AsyncSession,
        algorithm_id: Optional[int] = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysPredLog], int]:
        """分页查询预测日志"""
        stmt = select(SysPredLog)
        if algorithm_id is not None:
            stmt = stmt.where(SysPredLog.algorithm_id == algorithm_id)
        stmt = stmt.order_by(desc(SysPredLog.id))
        return await self.paginate(db, stmt, page, size)


class EvalLogRepository(BaseRepository[SysEvalLog]):
    """评估日志 Repository"""

    model = SysEvalLog

    async def create_log(
        self,
        db: AsyncSession,
        algorithm_id: int,
        pred_md5: str,
        pred_url: str,
        gt_md5: str,
        gt_url: str,
        result: dict,
        time_ms: int,
        pred_file_id: Optional[int] = None,
        gt_file_id: Optional[int] = None,
    ) -> SysEvalLog:
        """创建评估日志"""
        import json
        log = SysEvalLog(
            algorithm_id=algorithm_id,
            pred_file_id=pred_file_id,
            pred_md5=pred_md5,
            pred_url=pred_url,
            gt_file_id=gt_file_id,
            gt_md5=gt_md5,
            gt_url=gt_url,
            time=time_ms // 1000,
            result=json.dumps(result) if isinstance(result, dict) else result,
        )
        return await self.create(db, log)

    async def get_paginated(
        self,
        db: AsyncSession,
        algorithm_id: Optional[int] = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询评估日志"""
        stmt = select(SysEvalLog)
        if algorithm_id is not None:
            stmt = stmt.where(SysEvalLog.algorithm_id == algorithm_id)
        stmt = stmt.order_by(desc(SysEvalLog.id))
        return await self.paginate(db, stmt, page, size)


pred_log_repository = PredLogRepository()
eval_log_repository = EvalLogRepository()
