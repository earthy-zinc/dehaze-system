"""图像任务编排域：单张任务提交、日志查询、任务取消。

推理经统一图像模型适配器执行（inference_executor），本域管理输入/输出/存储：
- 模型路由解析（image_route）：请求 model > 平台默认，校验模型/供应商可用
- 可插拔拦截器链（如 WPXNet 预查询）：命中即短路，不调用模型
- 基于 (model_id, imageMd5) 的 Redis 缓存（见 cache.py）
- 任务日志写入 sys_pred_log，状态机：processing → completed/failed/cancelled
- 提交前余额预校验（不预扣），成功后按张实扣（image_billing_service）；缓存/拦截器命中不计费
- POST 立即返回 taskId + status=processing，asyncio.create_task 后台执行
"""

import asyncio
import io
import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import async_session_factory, get_db_session
from app.models.entity.sys_ai_model import SysAiModel
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from app.repository.file_repository import file_repository
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction.cache import get_cached_prediction, set_cached_prediction
from app.service.prediction.image_billing_service import image_billing_service
from app.service.prediction.image_route import ImageRoute, resolve_image_model
from app.service.prediction.image_source import fetch_image
from app.service.prediction.inference_executor import run_edit
from app.service.prediction.interceptor import (
    PredictionContext,
    PredictionInterceptorChain,
)
from app.service.prediction.result_storage import upload_result
from app.service.prediction.wpxnet_interceptor import WpxNetPredictionInterceptor
from app.service.storage.base import StorageService
from app.service.storage.factory import get_storage_by_name
from app.utils.file import calculate_bytes_md5

logger = logging.getLogger(__name__)

# 单次预测完成回调注册表：pred_log_id -> async callback(result)
# 供 async_wait 中断链路（async_resume）注册，单张预测任务完成后自动恢复推理。
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
    """图像任务编排服务（异步任务模式）"""

    def __init__(self):
        self._interceptor_chain = _build_interceptor_chain()

    async def predict(
        self,
        model: str | None,
        image_url: str,
        params: dict | None = None,
        user_id: int | None = None,
        file_id: int | None = None,
        skip_billing: bool = False,
        recommended_by: int | None = None,
    ) -> dict:
        """
        提交图像任务（异步）

        流程：
        1. 解析图像模型路由，校验模型与供应商可用
        2. 调用前余额预校验（不预扣；欠费/超限/余额不足即拒绝）
        3. 命中拦截器（如 WPXNet 预查询）→ 直接写 completed 日志并返回完整结果（不计费）
        4. 下载图片、计算 MD5，命中 Redis 缓存 → 直接写 completed 日志并返回（不计费）
        5. 未命中 → 创建 processing 日志，提交 asyncio.create_task 后立即返回

        Returns:
            命中：{taskId, status: completed, resultUrl, resultMd5, time}
            未命中：{taskId, status: processing, estimatedCredits}
        """
        start = time.time()

        async with async_session_factory() as db:
            route = await resolve_image_model(db, model)
            estimated_credits = await image_billing_service.estimate_image_credits(
                db, route.model_id, route.provider.id, 1
            )
        billed = user_id is not None and not skip_billing

        if billed:
            assert user_id is not None
            async with get_db_session() as db:
                await image_billing_service.ensure_balance(db, user_id, estimated_credits)

        # 前置准备与日志创建（此阶段失败时尚未计费，无需补偿）
        origin_file: SysFile | None = None
        storage_service: StorageService | None = None
        if file_id is not None:
            async with async_session_factory() as db:
                origin_file = await file_repository.get_by_id(db, file_id)
            if origin_file is None:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"文件不存在: {file_id}")
            # URL 运行时拼接（baseUrl + object_name），不落库
            storage_service = get_storage_by_name(origin_file.storage)
            image_url = storage_service.get_url(origin_file.object_name)

        # 调用拦截器链（命中即短路，不调用模型）
        context = PredictionContext(
            model=route.model,
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
                "预测拦截器命中: model=%s, resultUrl=%s",
                route.model_id,
                intercepted.result_url,
            )
            return await self._write_completed_log(
                model_id=route.model_id,
                provider_id=route.provider.id,
                origin_md5=origin_file.md5 if origin_file else "",
                origin_url=image_url,
                pred_md5=intercepted.result_md5,
                pred_url=intercepted.result_url,
                pred_file_id=intercepted.result_file_id,
                origin_file_id=file_id,
                time_ms=elapsed,
                user_id=user_id,
                recommended_by=recommended_by,
            )

        # 下载输入图片（系统存储文件用 SDK 下载避免 minio 私有 bucket 匿名 GET 403）
        if origin_file is not None:
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

        # 查询 Redis 缓存（基于 model_id + imageMd5）
        cache_key = f"image:{route.model_id}:{image_md5}"
        cached = await get_cached_prediction(cache_key)
        # 缓存只存对象标识（storage + objectName），URL 在命中时按当前配置拼接，
        # 避免存储域名变更后仍返回历史环境的地址
        if cached is not None and cached.get("objectName"):
            elapsed = int((time.time() - start) * 1000)
            logger.debug("图像缓存命中: model=%s, md5=%s", route.model_id, image_md5)
            cached_storage = get_storage_by_name(cached["storage"])
            return await self._write_completed_log(
                model_id=route.model_id,
                provider_id=route.provider.id,
                origin_md5=image_md5,
                origin_url=image_url,
                pred_md5=cached["resultMd5"],
                pred_url=cached_storage.get_url(cached["objectName"]),
                origin_file_id=file_id,
                time_ms=elapsed,
                user_id=user_id,
                recommended_by=recommended_by,
                extra=cached,
            )

        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        try:
            async with get_db_session() as db:
                log = await pred_log_repository.create_pending_log(
                    db=db,
                    model_id=route.model_id,
                    origin_md5=image_md5,
                    origin_url=image_url,
                    origin_file_id=file_id,
                    recommended_by=recommended_by,
                    provider_id=route.provider.id,
                )
                log_id = log.id
        finally:
            set_current_user_id(None)

        # 提交异步任务（不等待完成）
        loop = asyncio.get_running_loop()
        background_task = loop.create_task(
            self._execute_async(
                log_id=log_id,
                route=route,
                image_bytes=image_bytes,
                image_md5=image_md5,
                cache_key=cache_key,
                user_id=user_id,
                params=params,
                billed=billed,
            )
        )

        # 注册到 TaskTracker，支持优雅关闭与全局任务视图
        try:
            from app.service.task_tracker import get_task_tracker

            await get_task_tracker().register(
                task_id=f"pred:{log_id}",
                task=background_task,
                task_type="prediction",
                metadata={"log_id": log_id, "model_id": route.model_id, "user_id": user_id},
            )
        except Exception as e:
            logger.warning("预测任务追踪注册失败（不影响执行）: %s", e)

        return {
            "taskId": log_id,
            "status": LogStatus.PROCESSING.value,
            "estimatedCredits": estimated_credits,
        }

    async def _write_completed_log(
        self,
        model_id: str,
        provider_id: int | None,
        origin_md5: str,
        origin_url: str,
        pred_md5: str,
        pred_url: str,
        time_ms: int,
        user_id: int | None = None,
        origin_file_id: int | None = None,
        pred_file_id: int | None = None,
        recommended_by: int | None = None,
        extra: dict | None = None,
    ) -> dict:
        """写 completed 日志并返回完整结果（拦截器命中 / 缓存命中共用，不计费）。

        pred_url 参数为运行时拼接的完整 URL（由调用方通过 storage.get_url 生成）。
        """
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        try:
            async with get_db_session() as db:
                log = await pred_log_repository.create_log(
                    db=db,
                    model_id=model_id,
                    origin_md5=origin_md5,
                    origin_url=origin_url,
                    pred_md5=pred_md5,
                    pred_url=pred_url,
                    time_ms=time_ms,
                    origin_file_id=origin_file_id,
                    pred_file_id=pred_file_id,
                    recommended_by=recommended_by,
                    provider_id=provider_id,
                )
                log_id = log.id
        finally:
            set_current_user_id(None)

        await self._award_process_growth(user_id, log_id)

        result = {
            "taskId": log_id,
            "status": LogStatus.COMPLETED.value,
            "resultUrl": pred_url,
            "resultMd5": pred_md5,
            "originMd5": origin_md5,
            "resultThumbnailUrl": None,
            "time": time_ms // 1000,
        }
        if extra:
            result["resultThumbnailUrl"] = extra.get("resultThumbnailUrl")
        return result

    async def _execute_async(
        self,
        log_id: int,
        route: ImageRoute,
        image_bytes: io.BytesIO,
        image_md5: str,
        cache_key: str,
        user_id: int | None = None,
        params: dict | None = None,
        billed: bool = False,
    ) -> None:
        """异步执行图像任务，成功后按张实扣并更新日志状态"""
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        start_time = time.time()
        try:
            result_bytes = await run_edit(route, image_bytes.getvalue(), params)

            result_object_name = await upload_result(result_bytes, route.model_id)
            result_md5 = calculate_bytes_md5(io.BytesIO(result_bytes))
            data_len = len(result_bytes)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # 运行时拼接完整 URL（与拦截器/缓存命中行为一致）
            result_url = get_storage_by_name(settings.FILE_STORAGE_TYPE).get_url(result_object_name)

            # 注册 sys_file（MD5 去重 + upsert 复活软删记录）+ 更新日志为 completed
            async with get_db_session() as db:
                result_file = await file_repository.upsert_by_md5(
                    db,
                    md5=result_md5,
                    type="prediction",
                    name=f"{route.model_id}.png",
                    object_name=result_object_name,
                    storage=settings.FILE_STORAGE_TYPE,
                    size=f"{data_len}",
                    size_bytes=data_len,
                )
                result_file_id = result_file.id

                updated = await pred_log_repository.update_result(
                    db=db,
                    log_id=log_id,
                    pred_md5=result_md5,
                    pred_url=result_url,
                    time_ms=elapsed_ms,
                    pred_file_id=result_file_id,
                    provider_id=route.provider.id,
                    actual_model=route.model_id,
                )

            # 任务已被取消（update_result 未流转成功）时丢弃结果：
            # 不写缓存、不计费、不发放成长值，保持已取消终态
            if not updated:
                logger.info("图像任务已取消，丢弃推理结果: logId=%s", log_id)
                return

            # 仅成功结果计费（缓存/拦截器命中的日志不由此路径写入，故天然不计费）
            if billed and user_id is not None:
                async with get_db_session() as db:
                    await image_billing_service.charge_image(
                        db, user_id, route.model_id, route.provider.id, 1
                    )

            # 写入 Redis 缓存（只存与环境无关的对象标识 + MD5，URL 命中时再拼接）
            await set_cached_prediction(
                cache_key,
                {
                    "storage": settings.FILE_STORAGE_TYPE,
                    "objectName": result_object_name,
                    "resultMd5": result_md5,
                    "resultThumbnailUrl": None,
                    "format": "png",
                },
            )

            logger.info(
                "异步图像任务完成: logId=%s, model=%s, time=%sms",
                log_id,
                route.model_id,
                elapsed_ms,
            )

            await self._award_process_growth(user_id, log_id)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(
                "异步图像任务失败: logId=%s, model=%s, error=%s",
                log_id,
                route.model_id,
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
            except Exception as update_err:
                logger.error(
                    "更新图像任务失败状态失败: logId=%s, error=%s",
                    log_id,
                    update_err,
                )
        finally:
            set_current_user_id(None)
            # async_wait 单任务回调：任务完成（成功/失败）后通知恢复推理
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

    async def list_logs(
        self,
        db: AsyncSession,
        user_id: int,
        model_id: str | None = None,
        page: int = 1,
        size: int = 10,
    ) -> tuple[list[SysPredLog], int]:
        """分页查询当前用户的图像任务日志"""
        return await pred_log_repository.get_paginated(
            db, model_id=model_id, user_id=user_id, page=page, size=size
        )

    async def get_log(self, db: AsyncSession, log_id: int) -> SysPredLog:
        """按 ID 取任务日志，不存在抛 A0401"""
        log = await pred_log_repository.get_by_id(db, log_id)
        if not log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图像任务不存在")
        return log

    async def get_model(self, model_id: str) -> SysAiModel:
        """按 model_id 取图像模型（供评估/对比等下游取展示名）"""
        from app.repository.ai_model_repository import ai_model_repository

        async with async_session_factory() as session:
            model = await ai_model_repository.get_by_model_id(session, model_id)
            if model is None:
                raise BusinessException(
                    ResultCode.RESOURCE_NOT_FOUND, f"图像模型不存在: {model_id}"
                )
            return model

    async def cancel_task(self, db: AsyncSession, log_id: int, user_id: int) -> dict:
        """取消图像任务（幂等）。

        契约：
        - 仅本人可取消（create_by 归属校验，他人任务返回 A0401 防枚举）。
        - 仅"处理中(1)"任务可取消：终止后台推理、状态置为"已取消(4)"。
        - 已完成(2)/已失败(3)/已取消(4)任务调用时幂等返回当前状态。
        - 任务不存在抛 A0401。
        - 取消时尚未计费（仅成功计费），故无费用回滚。
        """
        log = await pred_log_repository.get_by_id(db, log_id)
        if not log or log.create_by != user_id:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "图像任务不存在")

        if log.status != LogStatus.PROCESSING.value:
            return {"taskId": log.id, "status": log.status}

        await self._cancel_background_task(log_id)

        await pred_log_repository.update_status(
            db=db,
            log_id=log_id,
            status=LogStatus.CANCELLED.value,
            error_message="任务已取消",
            time_ms=0,
        )

        logger.info("图像任务已取消: logId=%s", log_id)
        return {"taskId": log.id, "status": LogStatus.CANCELLED.value}

    async def _cancel_background_task(self, log_id: int) -> None:
        """取消本 Worker 中仍在运行的预测后台任务（pred:{log_id}）。"""
        try:
            from app.service.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            task_id = f"pred:{log_id}"
            await tracker.cancel_task(task_id)
        except Exception as e:
            logger.warning("取消后台任务失败（不影响状态落库）: logId=%s, error=%s", log_id, e)


prediction_service = PredictionService()
