"""
评估服务 —— 编排去雾效果评估流程（异步任务模式）

对比预测结果与参考图像，由 dehaze-modal 计算 PSNR/SSIM/LPIPS/NIQE 等多维指标，
编排侧基于阈值判定是否合格。

状态机：processing → completed/failed
POST 立即返回 logId + status=processing，asyncio.create_task 后台执行
"""

import asyncio
import json
import logging
import time

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db_session
from app.infrastructure.modal import engine_client
from app.models.entity.sys_log import SysEvalLog
from app.models.enum.log_status import LogStatus
from app.repository.pred_eval_log_repository import eval_log_repository
from app.service.prediction.image_source import fetch_image
from app.service.prediction.prediction_service import prediction_service
from app.utils.file import calculate_bytes_md5

logger = logging.getLogger(__name__)

# 指标计算上限（秒）：编排侧新增（设计 §3.4），引擎侧无自限、逐指标隔离失败。
# 引擎侧（dehaze-modal）首次评估需加载 pyiqa 指标模型，取值需覆盖该冷启动。
METRICS_TIMEOUT_SECONDS = 120

QUALIFIED_THRESHOLDS = {
    "psnr": 30.0,
    "ssim": 0.8,
    "lpips": 0.3,
    "niqe": 5.0,
}


async def _metric_catalog() -> list[dict]:
    """指标集目录：由 dehaze-modal 能力发现下发（指标名、展示标签、是否需要参考图）"""
    items = await engine_client.describe_models()
    return [item for item in items if item["type"] == "metrics"]


def _is_qualified(metrics: dict[str, float]) -> bool:
    """基于阈值判定是否合格（metrics 以指标 label 为 key，如 "PSNR"，统一转小写比对）"""
    if not metrics:
        return False
    normalized = {key.lower(): value for key, value in metrics.items()}
    psnr = normalized.get("psnr", 0)
    ssim = normalized.get("ssim", 0)
    lpips = normalized.get("lpips", 1.0)
    niqe = normalized.get("niqe", 99.0)
    return (
        psnr >= QUALIFIED_THRESHOLDS["psnr"]
        and ssim >= QUALIFIED_THRESHOLDS["ssim"]
        and lpips <= QUALIFIED_THRESHOLDS["lpips"]
        and niqe <= QUALIFIED_THRESHOLDS["niqe"]
    )


def parse_eval_result(value) -> dict | None:
    """评估结果归一化：result 列为 JSON 类型，正常读取即 dict；
    存量行（双重编码写入）与 Java 写入路径返回 JSON 字符串，此处统一解析为 dict。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("评估结果 result 列解析失败（非合法 JSON 字符串），按空处理")
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class EvaluationService:
    """去雾效果评估服务（异步任务模式）"""

    async def evaluate(
        self,
        model_id: str,
        pred_url: str,
        gt_url: str,
        user_id: int | None = None,
        conv_id: int | None = None,
        msg_id: int | None = None,
    ) -> dict:
        """
        提交效果评估任务（异步）

        流程：
        1. 校验算法、并行下载 pred/gt 图片、计算 MD5
        2. 创建 processing 日志
        3. 提交 asyncio.create_task 后台执行
        4. 立即返回 {logId, status: "processing"}

        评估为平台免费能力，不计费（设计 §9.1）。
        """
        logger.debug("评估请求: modelId=%s", model_id)

        # 1. 校验算法存在
        await prediction_service.get_model(model_id)

        # 2. 并行下载预测图和参考图
        pred_bytes, gt_bytes = await asyncio.gather(
            fetch_image(pred_url),
            fetch_image(gt_url),
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
                    model_id=model_id,
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
                model_id=model_id,
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
                metadata={"log_id": log_id, "model_id": model_id, "user_id": user_id},
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
        user_id: int,
        model_id: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询当前用户的评估日志（按用户隔离）"""
        return await eval_log_repository.get_paginated(
            db, user_id=user_id, model_id=model_id, page=page, size=size
        )

    async def get_log(
        self,
        db: AsyncSession,
        log_id: int,
    ) -> SysEvalLog:
        """按 ID 取评估日志，不存在抛 A0401"""
        log = await eval_log_repository.get_by_id(db, log_id)
        if not log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评估任务不存在")
        return log

    async def list_completed_metrics(
        self,
        db: AsyncSession,
        user_id: int,
        model_id: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysEvalLog], int]:
        """分页查询当前用户的已完成评估记录（仅 completed + evaluation 类型，按用户隔离）。

        task_type 过滤排除对比报告任务行（同表存储，result 为 reportHtml 包装对象）。
        """
        stmt = select(SysEvalLog).where(
            SysEvalLog.create_by == user_id,
            SysEvalLog.status == LogStatus.COMPLETED.value,
            SysEvalLog.task_type == "evaluation",
        )
        if model_id is not None:
            stmt = stmt.where(SysEvalLog.model_id == model_id)
        stmt = stmt.order_by(desc(SysEvalLog.id))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        paged_stmt = stmt.offset((page - 1) * size).limit(size)
        result = await db.execute(paged_stmt)
        return list(result.scalars().all()), total

    async def _execute_async(
        self,
        log_id: int,
        model_id: str,
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
            # 1. 调用 dehaze-modal 计算指标：指标名与展示标签由能力发现下发，
            #    落库 key 用 label（跨端契约：前端雷达图按 PSNR/SSIM 等标签取值）
            catalog = await _metric_catalog()
            if not catalog:
                raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "指标集为空，无法评估")
            labels = {item["id"]: item["capabilities"]["label"] for item in catalog}
            payload = await engine_client.compute_metrics(
                model_id="pyiqa",  # 指标集标识，仅用于引擎侧日志与指标维度
                names=list(labels),
                image=pred_bytes,
                reference=gt_bytes
                if any(item["capabilities"]["requires_clear"] for item in catalog)
                else None,
                timeout=METRICS_TIMEOUT_SECONDS,
            )
            if payload["failed"]:
                # 逐指标隔离失败：部分失败仍落已完成项（设计 §6.2）
                logger.warning("评估指标部分失败: logId=%s, failed=%s", log_id, payload["failed"])
            metrics = {labels[name]: value for name, value in payload["metrics"].items()}
            if not metrics:
                raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "评估指标计算全部失败")

            elapsed_ms = int((time.time() - start_time) * 1000)
            qualified = _is_qualified(metrics)
            logger.info(
                "异步评估完成: logId=%s, modelId=%s, time=%sms, qualified=%s, metrics=%s",
                log_id,
                model_id,
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

            await self._award_evaluate_growth(user_id, log_id)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(
                "异步评估失败: logId=%s, modelId=%s, error=%s",
                log_id,
                model_id,
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
            except Exception as update_err:
                logger.error(
                    "更新评估日志失败状态失败: logId=%s, error=%s",
                    log_id,
                    update_err,
                )
        finally:
            set_current_user_id(None)

    async def _award_evaluate_growth(self, user_id: int | None, log_id: int) -> None:
        """效果评估完成激励成长值（每日上限由会员模块控制），失败不阻断评估主流程。"""
        if user_id is None:
            return
        try:
            from app.service.member.growth_service import member_growth_service

            async with get_db_session() as db:
                await member_growth_service.add_behavior_growth(
                    db, user_id, "evaluate", related_id=str(log_id)
                )
        except Exception:
            logger.warning(
                "效果评估成长值激励失败: userId=%s, logId=%s", user_id, log_id, exc_info=True
            )


evaluation_service = EvaluationService()
