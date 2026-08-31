"""预测编排域：单张/批量预测任务提交、日志查询、任务取消。

调用算法模块的 dehaze() 函数，管理输入/输出/存储：
- 可插拔拦截器链（如 WPXNet 预查询）：命中即短路，不调用算法
- 基于 (algorithmId, imageMd5) 的 Redis 缓存（见 cache.py）
- 预测日志写入 sys_pred_log，状态机：processing → completed/failed
- POST 立即返回 logId + status=processing，asyncio.create_task 后台执行
"""

import asyncio
import io
import json
import logging
import time
from datetime import datetime
from typing import Any

import PIL.Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import async_session_factory, get_db_session
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from app.models.schema.prediction import BatchPredictionItem
from app.repository.algorithm_repository import algorithm_repository
from app.repository.file_repository import file_repository
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction.cache import (
    get_cached_prediction,
    invalidate_prediction_cache,
    set_cached_prediction,
)
from app.service.prediction.image_source import fetch_image
from app.service.prediction.interceptor import (
    PredictionContext,
    PredictionInterceptorChain,
)
from app.service.prediction.inference_executor import run_dehaze
from app.service.prediction.result_storage import upload_result
from app.service.prediction.wpxnet_interceptor import WpxNetPredictionInterceptor
from app.service.storage.base import StorageService
from app.utils.file import calculate_bytes_md5

logger = logging.getLogger(__name__)

# 单次预测完成回调注册表：pred_log_id -> async callback(result)
# 供 async_wait 中断链路（async_resume）注册，单张预测任务完成后自动恢复推理。
# 批量处理走 process_batch 的批级回调，不在此注册逐张回调，避免逐张触发 resume。
_prediction_done_callbacks: dict[int, Any] = {}


def register_prediction_done_callback(log_id: int, callback: Any) -> None:
    """注册单次预测完成回调（async_wait 单任务场景）。"""
    _prediction_done_callbacks[log_id] = callback


def _build_interceptor_chain() -> PredictionInterceptorChain:
    """构建预测拦截器责任链（新增插件在此注册）"""
    return PredictionInterceptorChain(
        [
            WpxNetPredictionInterceptor(),
        ]
    )


class PredictionService:
    """模型预测编排服务（异步任务模式）"""

    def __init__(self):
        self._interceptor_chain = _build_interceptor_chain()

    async def predict(
        self,
        algorithm_id: int,
        image_url: str,
        params: dict | None = None,
        user_id: int | None = None,
        file_id: int | None = None,
        skip_quota_check: bool = False,
    ) -> dict:
        """
        提交预测任务（异步）

        流程：
        1. 校验算法
        2. 命中拦截器（如 WPXNet 预查询） → 直接写 completed 日志并返回完整结果
        3. 下载图片、计算 MD5，命中 Redis 缓存 → 直接写 completed 日志并返回完整结果
        4. 未命中 → 创建 processing 日志，提交 asyncio.create_task 后立即返回

        Returns:
            命中：{logId, status: "completed", resultUrl, resultMd5, time}
            未命中：{logId, status: "processing"}
        """
        start = time.time()

        if user_id is not None and not skip_quota_check:
            from app.service.member.quota_service import member_quota_service

            async with get_db_session() as db:
                await member_quota_service.check_and_deduct_quota(db, user_id, "dehaze")

        algorithm = await self.get_algorithm(algorithm_id)

        # 2. fileId 存在时查询原始文件并用其真实 object_name 拼接访问 URL
        #   （对齐 Java resolveImageUrl）
        origin_file: SysFile | None = None
        storage_service: StorageService | None = None
        if file_id is not None:
            async with async_session_factory() as db:
                origin_file = await file_repository.get_by_id(db, file_id)
            if origin_file is None:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"文件不存在: {file_id}")
            # URL 运行时拼接（baseUrl + object_name），不落库
            from app.service.storage.factory import get_storage_by_name

            storage_service = get_storage_by_name(origin_file.storage)
            image_url = storage_service.get_url(origin_file.object_name)

        # 3. 调用拦截器链（命中即短路，不调用算法）
        context = PredictionContext(
            algorithm=algorithm,
            file_id=file_id,
            image_url=image_url,
            origin_file=origin_file,
            params=params,
            start_time_ms=int(start * 1000),
        )
        intercepted = await self._interceptor_chain.intercept(context)
        if intercepted is not None:
            elapsed = int((time.time() - start) * 1000)
            logger.debug(
                "预测拦截器命中: algorithmId=%s, resultUrl=%s",
                algorithm_id,
                intercepted.result_url,
            )
            return await self._write_completed_log(
                algorithm_id=algorithm_id,
                origin_md5=origin_file.md5 if origin_file else "",
                origin_url=image_url,
                pred_md5=intercepted.result_md5,
                pred_url=intercepted.result_url,
                pred_file_id=intercepted.result_file_id,
                origin_file_id=file_id,
                time_ms=elapsed,
                user_id=user_id,
            )

        # 4. 下载输入图片（系统存储文件用 SDK 下载避免 minio 私有 bucket 匿名 GET 403）
        if origin_file is not None:
            # origin_file 仅在 file_id 非空分支赋值，此时 storage_service 必已获取
            assert storage_service is not None
            bucket = settings.MINIO_BUCKET
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None, lambda: storage_service.download(bucket, origin_file.object_name)
            )
            image_bytes = io.BytesIO(raw)
        else:
            image_bytes = await fetch_image(image_url)
        image_md5 = calculate_bytes_md5(image_bytes)

        # 5. 查询 Redis 缓存（基于 algorithmId + imageMd5）
        cache_key = f"prediction:{algorithm_id}:{image_md5}"
        cached = await get_cached_prediction(cache_key)
        if cached is not None:
            elapsed = int((time.time() - start) * 1000)
            logger.debug("预测缓存命中: algorithmId=%s, md5=%s", algorithm_id, image_md5)
            return await self._write_completed_log(
                algorithm_id=algorithm_id,
                origin_md5=image_md5,
                origin_url=image_url,
                pred_md5=cached.get("resultMd5", ""),
                pred_url=cached["resultUrl"],
                origin_file_id=file_id,
                time_ms=elapsed,
                user_id=user_id,
                extra=cached,
            )

        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        try:
            async with get_db_session() as db:
                log = await pred_log_repository.create_pending_log(
                    db=db,
                    algorithm_id=algorithm_id,
                    origin_md5=image_md5,
                    origin_url=image_url,
                    origin_file_id=file_id,
                )
                log_id = log.id
        finally:
            set_current_user_id(None)

        # 7. 提交异步任务（不等待完成）
        loop = asyncio.get_running_loop()
        background_task = loop.create_task(
            self._execute_async(
                log_id=log_id,
                algorithm_id=algorithm_id,
                image_bytes=image_bytes,
                image_md5=image_md5,
                algorithm=algorithm,
                cache_key=cache_key,
                user_id=user_id,
            )
        )

        # 注册到 TaskTracker，支持优雅关闭与全局任务视图
        try:
            from app.service.task_tracker import get_task_tracker

            await get_task_tracker().register(
                task_id=f"pred:{log_id}",
                task=background_task,
                task_type="prediction",
                metadata={"log_id": log_id, "algorithm_id": algorithm_id, "user_id": user_id},
            )
        except Exception as e:
            logger.warning("预测任务追踪注册失败（不影响执行）: %s", e)

        return {
            "logId": log_id,
            "status": LogStatus.PROCESSING.value,
        }

    async def _write_completed_log(
        self,
        algorithm_id: int,
        origin_md5: str,
        origin_url: str,
        pred_md5: str,
        pred_url: str,
        time_ms: int,
        user_id: int | None = None,
        origin_file_id: int | None = None,
        pred_file_id: int | None = None,
        extra: dict | None = None,
    ) -> dict:
        """写 completed 日志并返回完整结果（拦截器命中 / 缓存命中共用）。

        pred_url 参数为运行时拼接的完整 URL（由调用方通过 storage.get_url 生成）。
        """
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        try:
            async with get_db_session() as db:
                log = await pred_log_repository.create_log(
                    db=db,
                    algorithm_id=algorithm_id,
                    origin_md5=origin_md5,
                    origin_url=origin_url,
                    pred_md5=pred_md5,
                    pred_url=pred_url,
                    time_ms=time_ms,
                    origin_file_id=origin_file_id,
                    pred_file_id=pred_file_id,
                )
                log_id = log.id
        finally:
            set_current_user_id(None)

        await self._award_process_growth(user_id, log_id)

        result = {
            "logId": log_id,
            "status": LogStatus.COMPLETED.value,
            "resultUrl": pred_url,
            "resultMd5": pred_md5,
            "resultThumbnailUrl": None,
            "time": time_ms,
        }
        if extra:
            result["resultThumbnailUrl"] = extra.get("resultThumbnailUrl")
        return result

    async def _execute_async(
        self,
        log_id: int,
        algorithm_id: int,
        image_bytes: io.BytesIO,
        image_md5: str,
        algorithm: SysAlgorithm,
        cache_key: str,
        user_id: int | None = None,
    ) -> None:
        """异步执行预测任务，完成后更新日志状态"""
        from app.infrastructure.metrics.inference_metrics import record_inference_metrics
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        start_time = time.time()
        try:
            try:
                with PIL.Image.open(image_bytes) as img:
                    image_size = img.width * img.height
                image_bytes.seek(0)
            except Exception:
                image_size = None

            # 2. 执行去雾推理（CPU 密集型 → 线程池）
            inference_start = time.monotonic()
            inference_status = "success"
            try:
                result_bytes = await run_dehaze(
                    algorithm.import_path or algorithm.name,
                    algorithm.path or "",
                    image_bytes,
                )
            except Exception:
                inference_status = "error"
                raise
            finally:
                record_inference_metrics(
                    algorithm=algorithm.name,
                    duration_seconds=time.monotonic() - inference_start,
                    status=inference_status,
                    image_size=image_size,
                )

            result_object_name = await upload_result(result_bytes, algorithm.name)
            result_md5 = calculate_bytes_md5(result_bytes)
            data_len = len(result_bytes.getvalue())
            elapsed_ms = int((time.time() - start_time) * 1000)

            # 运行时拼接完整 URL（与 wpxnet 拦截器/缓存命中行为一致）
            from app.service.storage.factory import get_storage_by_name

            result_url = get_storage_by_name(settings.FILE_STORAGE_TYPE).get_url(result_object_name)

            # 4. 注册 sys_file（MD5 去重 + upsert 复活软删记录）+ 更新日志为 completed
            async with get_db_session() as db:
                result_file = await file_repository.upsert_by_md5(
                    db,
                    md5=result_md5,
                    type="prediction",
                    name=f"{algorithm.name}.png",
                    object_name=result_object_name,
                    storage=settings.FILE_STORAGE_TYPE,
                    size=f"{data_len}",
                    size_bytes=data_len,
                )
                result_file_id = result_file.id

                await pred_log_repository.update_result(
                    db=db,
                    log_id=log_id,
                    pred_md5=result_md5,
                    pred_url=result_url,
                    time_ms=elapsed_ms,
                    pred_file_id=result_file_id,
                )

            # 5. 写入 Redis 缓存（存完整 URL，与拦截器命中行为一致）
            await set_cached_prediction(
                cache_key,
                {
                    "resultUrl": result_url,
                    "resultMd5": result_md5,
                    "resultThumbnailUrl": None,
                    "format": "png",
                },
            )

            logger.info(
                "异步预测完成: logId=%s, algorithmId=%s, time=%sms",
                log_id,
                algorithm_id,
                elapsed_ms,
            )

            await self._award_process_growth(user_id, log_id)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(
                "异步预测失败: logId=%s, algorithmId=%s, error=%s",
                log_id,
                algorithm_id,
                error_msg,
                exc_info=True,
            )
            try:
                async with get_db_session() as db:
                    await pred_log_repository.update_status(
                        db=db,
                        log_id=log_id,
                        status=LogStatus.FAILED.value,
                        error_message=error_msg,
                        time_ms=elapsed_ms,
                    )
                    if user_id is not None:
                        from app.service.member.quota_service import member_quota_service

                        await member_quota_service.restore_quota(db, user_id, "dehaze")
            except Exception as update_err:
                logger.error(
                    "更新预测日志失败状态失败: logId=%s, error=%s",
                    log_id,
                    update_err,
                )
        finally:
            set_current_user_id(None)
            # async_wait 单任务回调：预测完成（成功/失败）后通知恢复推理
            callback = _prediction_done_callbacks.pop(log_id, None)
            if callback is not None:
                try:
                    await callback(log_id)
                except Exception:
                    logger.warning("预测完成回调执行失败: logId=%s", log_id, exc_info=True)

    async def _award_process_growth(self, user_id: int | None, log_id: int) -> None:
        """图像处理完成激励成长值（每日上限由会员模块控制），失败不阻断处理主流程。"""
        if user_id is None:
            return
        try:
            from app.service.member.growth_service import member_growth_service

            async with get_db_session() as db:
                await member_growth_service.add_behavior_growth(
                    db, user_id, "process", related_id=str(log_id)
                )
        except Exception:
            logger.warning(
                "图像处理成长值激励失败: userId=%s, logId=%s", user_id, log_id, exc_info=True
            )

    async def invalidate_cache(self, algorithm_id: int) -> int:
        """版本更新时失效该算法的所有预测缓存"""
        return await invalidate_prediction_cache(algorithm_id)

    async def list_logs(
        self,
        db: AsyncSession,
        algorithm_id: int | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysPredLog], int]:
        """分页查询预测日志（管理视图，全量）"""
        return await pred_log_repository.get_paginated(
            db, algorithm_id=algorithm_id, page=page, size=size
        )

    async def get_log(self, db: AsyncSession, log_id: int) -> SysPredLog:
        """按 ID 取预测日志，不存在抛 A0401"""
        log = await pred_log_repository.get_by_id(db, log_id)
        if not log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预测任务不存在")
        return log

    async def cancel_task(self, db: AsyncSession, log_id: int, user_id: int) -> dict:
        """取消预测任务（幂等）。

        契约：
        - 仅"处理中(1)"任务可取消：终止后台推理、回滚已扣减配额、状态置为"已取消(4)"。
        - 已完成(2)/已失败(3)/已取消(4)任务调用时幂等返回当前状态，不重复回滚配额。
        - 任务不存在抛 A0401。
        """
        from app.service.member.quota_service import member_quota_service

        log = await pred_log_repository.get_by_id(db, log_id)
        if not log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预测任务不存在")

        if log.status != LogStatus.PROCESSING.value:
            # 幂等：非处理中任务直接返回当前状态，不重复回滚配额
            return {"logId": log.id, "status": log.status}

        await self._cancel_background_task(log_id)

        await pred_log_repository.update_status(
            db=db,
            log_id=log_id,
            status=LogStatus.CANCELLED.value,
            error_message="任务已取消",
            time_ms=0,
        )

        # 3. 回滚已扣减配额（restore_quota 内部有 used>0 保护，m2m/免配额用户不受影响）
        try:
            await member_quota_service.restore_quota(db, user_id, "dehaze")
        except Exception as e:
            logger.warning("取消任务回滚配额失败: logId=%s, error=%s", log_id, e)

        logger.info("预测任务已取消: logId=%s", log_id)
        return {"logId": log.id, "status": LogStatus.CANCELLED.value}

    async def _cancel_background_task(self, log_id: int) -> None:
        """取消本 Worker 中仍在运行的预测后台任务（pred:{log_id}）。"""
        try:
            from app.service.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            task_id = f"pred:{log_id}"
            await tracker.cancel_task(task_id)
        except Exception as e:
            logger.warning("取消后台任务失败（不影响状态落库）: logId=%s, error=%s", log_id, e)

    async def get_algorithm(self, algorithm_id: int) -> SysAlgorithm:
        """从数据库获取算法"""
        async with async_session_factory() as session:
            algorithm = await algorithm_repository.get_by_id(session, algorithm_id)
            if algorithm is None:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在")
            return algorithm

    async def batch_predict(
        self,
        algorithm_id: int,
        items: list[BatchPredictionItem],
        user_id: int,
        skip_quota_check: bool = False,
    ) -> list[dict[str, Any]]:
        """批量处理：校验上限后逐张提交预测"""
        # T-DH-027：空 items 直接参数校验失败
        if not items:
            raise BusinessException(ResultCode.PARAM_ERROR, "批量处理图片列表不能为空")

        if len(items) > 20:
            raise BusinessException(ResultCode.BUSINESS_ERROR, "批量处理最多支持20张图片")

        if not skip_quota_check:
            from app.repository.member_benefit_repository import member_benefit_repository
            from app.repository.member_repository import member_repository

            async with get_db_session() as db:
                member = await member_repository.get_by_user_id(db, user_id)
                if not member:
                    raise BusinessException(ResultCode.MEMBER_NOT_FOUND)
                benefit = await member_benefit_repository.get_by_level_code(db, member.level_code)
                batch_limit = benefit.batch_limit if benefit else 5
                # T-DH-023：超过会员等级上限返回 A0500（对齐文档）
                if len(items) > batch_limit:
                    raise BusinessException(
                        ResultCode.BUSINESS_ERROR,
                        f"批量处理图片数量不能超过{batch_limit}张",
                    )

        results = []
        for item in items:
            file_id = item.fileId
            image_url = item.imageUrl
            raw_params = item.params

            # 图片来源校验：fileId 或 imageUrl 至少提供一个（对齐单张预测）
            if file_id is None and not image_url:
                raise BusinessException(
                    ResultCode.PARAM_IS_NULL, "图片来源不能为空，请提供 fileId 或 imageUrl"
                )

            # 类型归一化：fileId 存在时 predict 内部会用库内原始图 URL 覆盖，None 归一为空串
            image_url = image_url or ""

            try:
                # params 为 JSON 字符串，解析为 dict（对齐单张预测）
                params = None
                if raw_params:
                    try:
                        params = json.loads(raw_params)
                    except json.JSONDecodeError:
                        raise BusinessException(
                            ResultCode.PARAM_ERROR, f"参数格式错误: {raw_params}"
                        ) from None

                result = await self.predict(
                    algorithm_id=algorithm_id,
                    image_url=image_url,
                    params=params,
                    user_id=user_id,
                    file_id=file_id,
                    skip_quota_check=skip_quota_check,
                )
                results.append(
                    {
                        "logId": result.get("logId"),
                        "status": result.get("status", LogStatus.PROCESSING.value),
                        "resultUrl": result.get("resultUrl"),
                        "resultThumbnailUrl": result.get("resultThumbnailUrl"),
                        "time": result.get("time", 0),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "logId": None,
                        "status": LogStatus.FAILED.value,
                        "errorMessage": str(e),
                        "time": 0,
                    }
                )

        return results

    @staticmethod
    async def get_quota(db, user_id: int) -> dict:
        """查询用户本月剩余去雾处理次数。

        resetDate 为配额重置日期：月度配额按下月 1 日重置（对齐会员模块 quota_reset_month）。
        """
        from app.repository.member_repository import member_repository

        member = await member_repository.get_by_user_id(db, user_id)
        if not member:
            raise BusinessException(ResultCode.MEMBER_NOT_FOUND)

        total = member.monthly_dehaze_quota or 0
        used = member.monthly_dehaze_used or 0
        remaining = max(0, total - used)

        now = datetime.now()
        if now.month == 12:
            reset_date = datetime(now.year + 1, 1, 1)
        else:
            reset_date = datetime(now.year, now.month + 1, 1)

        return {
            "total": total,
            "used": used,
            "remaining": remaining,
            "resetDate": reset_date.strftime("%Y-%m-%d"),
        }


prediction_service = PredictionService()
