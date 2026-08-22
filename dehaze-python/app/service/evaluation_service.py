"""
评估服务 —— 编排去雾效果评估流程（异步任务模式）

对比预测结果与参考图像，计算 PSNR/SSIM/LPIPS/NIQE/Entropy 等多维指标，
并基于阈值判定是否合格。

状态机：processing → completed/failed
POST 立即返回 logId + status=processing，asyncio.create_task 后台执行
"""

import asyncio
import logging
import time

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from algorithm.metrics import calculate as calculate_metrics
from app.database import get_db_session
from app.models.entity.sys_log import SysEvalLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.prediction_service import prediction_service
from app.utils.file import calculate_bytes_md5

logger = logging.getLogger(__name__)

QUALIFIED_THRESHOLDS = {
    "psnr": 30.0,
    "ssim": 0.8,
    "lpips": 0.3,
    "niqe": 5.0,
}


def _is_qualified(metrics: dict[str, float]) -> bool:
    """基于阈值判定是否合格"""
    if not metrics:
        return False
    psnr = metrics.get("psnr", 0)
    ssim = metrics.get("ssim", 0)
    lpips = metrics.get("lpips", 1.0)
    niqe = metrics.get("niqe", 99.0)
    return (
        psnr >= QUALIFIED_THRESHOLDS["psnr"]
        and ssim >= QUALIFIED_THRESHOLDS["ssim"]
        and lpips <= QUALIFIED_THRESHOLDS["lpips"]
        and niqe <= QUALIFIED_THRESHOLDS["niqe"]
    )


class EvaluationService:
    """去雾效果评估服务（异步任务模式）"""

    async def evaluate(
        self,
        algorithm_id: int,
        pred_url: str,
        gt_url: str,
        user_id: int | None = None,
        skip_quota_check: bool = False,
        conv_id: int | None = None,
        msg_id: int | None = None,
    ) -> dict:
        """
        提交效果评估任务（异步）

        流程：
        1. 权益校验 + Redis 原子扣减评估配额
        2. 校验算法、并行下载 pred/gt 图片、计算 MD5
        3. 创建 processing 日志
        4. 提交 asyncio.create_task 后台执行
        5. 立即返回 {logId, status: "processing"}
        """
        logger.debug("评估请求: algorithmId=%s", algorithm_id)

        from app.database import get_db_session
        from app.service.member_service import MemberService

        if user_id is not None and not skip_quota_check:
            async with get_db_session() as db:
                await MemberService.check_and_deduct_quota(db, user_id, "evaluate")

        # 1. 校验算法存在
        await prediction_service.get_algorithm(algorithm_id)

        # 2. 并行下载预测图和参考图
        pred_bytes, gt_bytes = await asyncio.gather(
            prediction_service.download_image(pred_url),
            prediction_service.download_image(gt_url),
        )

        # 3. 计算图片 MD5（CPU 密集型，移至线程池）
        pred_md5, gt_md5 = await asyncio.gather(
            asyncio.to_thread(calculate_bytes_md5, pred_bytes),
            asyncio.to_thread(calculate_bytes_md5, gt_bytes),
        )

        # 4. 创建 processing 日志
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        try:
            async with get_db_session() as db:
                log = await eval_log_repository.create_pending_log(
                    db=db,
                    algorithm_id=algorithm_id,
                    pred_md5=pred_md5,
                    pred_url=pred_url,
                    gt_md5=gt_md5,
                    gt_url=gt_url,
                )
                log_id = log.id
        finally:
            set_current_user_id(None)

        # 5. 提交异步任务（不等待完成）
        loop = asyncio.get_running_loop()
        background_task = loop.create_task(
            self._execute_async(
                log_id=log_id,
                algorithm_id=algorithm_id,
                pred_bytes=pred_bytes,
                gt_bytes=gt_bytes,
                user_id=user_id,
                conv_id=conv_id,
                msg_id=msg_id,
            )
        )

        # 注册到 TaskTracker，支持优雅关闭与全局任务视图
        try:
            from app.service.task_tracker import get_task_tracker

            await get_task_tracker().register(
                task_id=f"eval:{log_id}",
                task=background_task,
                task_type="evaluation",
                metadata={"log_id": log_id, "algorithm_id": algorithm_id, "user_id": user_id},
            )
        except Exception as e:
            logger.warning("评估任务追踪注册失败（不影响执行）: %s", e)

        # 6. 立即返回 processing
        return {
            "logId": log_id,
            "status": LogStatus.PROCESSING.value,
        }

    async def list_logs(
        self,
        db: AsyncSession,
        algorithm_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询评估日志（管理视图，全量；用户隔离见 list_completed_metrics）"""
        return await eval_log_repository.get_paginated(
            db, algorithm_id=algorithm_id, page=page, size=size
        )

    async def get_log(
        self,
        db: AsyncSession,
        log_id: int,
    ) -> SysEvalLog:
        """按 ID 取评估日志，不存在抛 A0401"""
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

        log = await eval_log_repository.get_by_id(db, log_id)
        if not log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评估任务不存在")
        return log

    async def list_completed_metrics(
        self,
        db: AsyncSession,
        user_id: int,
        algorithm_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询当前用户的已完成评估记录（仅返回 completed，按用户隔离）。"""
        stmt = select(SysEvalLog).where(
            SysEvalLog.create_by == user_id,
            SysEvalLog.status == LogStatus.COMPLETED.value,
        )
        if algorithm_id is not None:
            stmt = stmt.where(SysEvalLog.algorithm_id == algorithm_id)
        stmt = stmt.order_by(desc(SysEvalLog.id))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        paged_stmt = stmt.offset((page - 1) * size).limit(size)
        result = await db.execute(paged_stmt)
        return list(result.scalars().all()), total

    async def _execute_async(
        self,
        log_id: int,
        algorithm_id: int,
        pred_bytes,
        gt_bytes,
        user_id: int | None = None,
        conv_id: int | None = None,
        msg_id: int | None = None,
    ) -> None:
        """异步执行评估任务，完成后更新日志状态，会话场景下注册 metric_report 产物"""
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        start_time = time.time()
        try:
            # 1. 调用评估（图像处理 CPU 密集型，移至线程池避免阻塞事件循环）
            pred_bytes.seek(0)
            gt_bytes.seek(0)
            metrics_list = await asyncio.to_thread(calculate_metrics, pred_bytes, gt_bytes)
            metrics = {m["label"]: m["value"] for m in metrics_list}

            elapsed_ms = int((time.time() - start_time) * 1000)
            qualified = _is_qualified(metrics)
            logger.info(
                "异步评估完成: logId=%s, algorithmId=%s, time=%sms, qualified=%s, metrics=%s",
                log_id,
                algorithm_id,
                elapsed_ms,
                qualified,
                metrics,
            )

            # 2. 更新日志为 completed，并在会话场景下注册 metric_report 产物
            async with get_db_session() as db:
                await eval_log_repository.update_result(
                    db=db,
                    log_id=log_id,
                    result=metrics,
                    time_ms=elapsed_ms,
                )
                if conv_id is not None and msg_id is not None:
                    from app.service.ai_artifact_service import ai_artifact_service

                    await ai_artifact_service.register_artifact(
                        db,
                        conv_id=conv_id,
                        msg_id=msg_id,
                        artifact_type="metric_report",
                        ref_type="sys_eval_log",
                        ref_id=log_id,
                        summary=metrics,
                    )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(
                "异步评估失败: logId=%s, algorithmId=%s, error=%s",
                log_id,
                algorithm_id,
                error_msg,
                exc_info=True,
            )
            try:
                async with get_db_session() as db:
                    await eval_log_repository.update_status(
                        db=db,
                        log_id=log_id,
                        status=LogStatus.FAILED.value,
                        error_message=error_msg,
                        time_ms=elapsed_ms,
                    )
                    if user_id is not None:
                        from app.service.member_service import MemberService

                        await MemberService.restore_quota(db, user_id, "evaluate")
            except Exception as update_err:
                logger.error(
                    "更新评估日志失败状态失败: logId=%s, error=%s",
                    log_id,
                    update_err,
                )
        finally:
            set_current_user_id(None)


evaluation_service = EvaluationService()
