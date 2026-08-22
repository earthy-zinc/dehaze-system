"""
预测服务 —— 编排算法预测流程（异步任务模式）

调用算法模块的 dehaze() 函数，管理输入/输出/存储
- 可插拔拦截器链（如 WPXNet 预查询）：命中即短路，不调用算法
- 基于 (algorithmId, imageMd5) 的 Redis 缓存（24h TTL）
- 预测日志写入 sys_pred_log，状态机：processing → completed/failed
- POST 立即返回 logId + status=processing，asyncio.create_task 后台执行
"""

import asyncio
import importlib
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import PIL.Image
from sqlalchemy.ext.asyncio import AsyncSession

from algorithm.model_loader import resolve_model_path
from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import async_session_factory, get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.infrastructure.logging import _trace_id_var
from app.infrastructure.metrics.inference_metrics import record_inference_metrics
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_file import SysFile
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from app.repository.algorithm_repository import algorithm_repository
from app.repository.file_repository import file_repository
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction import (
    PredictionContext,
    PredictionInterceptorChain,
)
from app.service.storage.base import StorageService
from app.service.prediction.wpxnet_interceptor import WpxNetPredictionInterceptor
from app.utils.file import calculate_bytes_md5

logger = logging.getLogger(__name__)

# 预测结果缓存 TTL：24 小时
PREDICTION_CACHE_TTL = 24 * 60 * 60

# 单次预测完成回调注册表：pred_log_id -> async callback(result)
# 供 async_wait 中断链路（async_resume）注册，单张预测任务完成后自动恢复推理。
# 批量处理走 process_batch 的批级回调，不在此注册逐张回调，避免逐张触发 resume。
_prediction_done_callbacks: dict[int, Any] = {}


def register_prediction_done_callback(log_id: int, callback: Any) -> None:
    """注册单次预测完成回调（async_wait 单任务场景）。"""
    _prediction_done_callbacks[log_id] = callback


# 算法推理专用线程池：PyTorch 推理为 CPU 密集型同步操作，
# 必须在线程池中执行以避免阻塞 asyncio 事件循环。
# 并发数通过 INFERENCE_THREAD_POOL_SIZE 配置，按 GPU 显存/卡数调整。
_inference_executor = ThreadPoolExecutor(
    max_workers=settings.INFERENCE_THREAD_POOL_SIZE, thread_name_prefix="algo-inference"
)


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
            from app.service.member_service import MemberService

            async with get_db_session() as db:
                await MemberService.check_and_deduct_quota(db, user_id, "dehaze")

        # 1. 从数据库获取算法信息
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
            bucket = settings.MINIO_BUCKET_NAME
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None, lambda: storage_service.download(bucket, origin_file.object_name)
            )
            image_bytes = io.BytesIO(raw)
        else:
            image_bytes = await self.download_image(image_url)
        image_md5 = calculate_bytes_md5(image_bytes)

        # 5. 查询 Redis 缓存（基于 algorithmId + imageMd5）
        cache_key = f"prediction:{algorithm_id}:{image_md5}"
        cached = await self._get_cached_prediction(cache_key)
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

        # 6. 缓存未命中：创建 processing 日志
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

        # 8. 立即返回 processing
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
        from app.models.base import set_current_user_id

        set_current_user_id(user_id)
        start_time = time.time()
        try:
            # 1. 计算输入图像尺寸（用于指标）
            try:
                with PIL.Image.open(image_bytes) as img:
                    image_size = img.width * img.height
                image_bytes.seek(0)
            except Exception:
                image_size = None

            # 2. 执行去雾推理（CPU 密集型 → 线程池）
            loop = asyncio.get_running_loop()
            inference_start = time.monotonic()
            inference_status = "success"
            try:
                result_bytes = await loop.run_in_executor(
                    _inference_executor,
                    self._run_dehaze,
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

            # 3. 上传结果（返回 object_name，URL 运行时拼接）
            result_object_name = await self._upload_result(result_bytes, algorithm.name)
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
            await self._set_cached_prediction(
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
                        from app.service.member_service import MemberService

                        await MemberService.restore_quota(db, user_id, "dehaze")
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

    async def _get_cached_prediction(self, cache_key: str) -> dict | None:
        """从 Redis 读取预测缓存（带降级）"""

        async def _get():
            redis = await get_redis_client()
            data = await redis.get(cache_key)
            if data:
                return json.loads(data)
            return None

        return await redis_operation_with_fallback(
            operation=_get,
            default=None,
            operation_name="prediction_cache_get",
        )

    async def _set_cached_prediction(self, cache_key: str, value: dict) -> None:
        """写入 Redis 预测缓存（带降级）"""

        async def _set():
            redis = await get_redis_client()
            await redis.setex(cache_key, PREDICTION_CACHE_TTL, json.dumps(value))

        await redis_operation_with_fallback(
            operation=_set,
            default=None,
            operation_name="prediction_cache_set",
        )

    async def invalidate_cache(self, algorithm_id: int) -> int:
        """版本更新时失效该算法的所有预测缓存"""

        async def _invalidate():
            redis = await get_redis_client()
            pattern = f"prediction:{algorithm_id}:*"
            keys = []
            async for key in redis.scan_iter(match=pattern, count=100):
                keys.append(key)
            if keys:
                await redis.delete(*keys)
            return len(keys)

        result = await redis_operation_with_fallback(
            operation=_invalidate,
            default=0,
            operation_name="prediction_cache_invalidate",
        )
        return result or 0

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
        from app.core.code import ResultCode
        from app.core.exceptions import BusinessException

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
        from app.service.member_service import MemberService

        log = await pred_log_repository.get_by_id(db, log_id)
        if not log:
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "预测任务不存在")

        if log.status != LogStatus.PROCESSING.value:
            # 幂等：非处理中任务直接返回当前状态，不重复回滚配额
            return {"logId": log.id, "status": log.status}

        # 1. 终止后台推理任务（若仍在本 Worker 运行）
        await self._cancel_background_task(log_id)

        # 2. 状态置为已取消
        await pred_log_repository.update_status(
            db=db,
            log_id=log_id,
            status=LogStatus.CANCELLED.value,
            error_message="任务已取消",
            time_ms=0,
        )

        # 3. 回滚已扣减配额（restore_quota 内部有 used>0 保护，m2m/免配额用户不受影响）
        try:
            await MemberService.restore_quota(db, user_id, "dehaze")
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

    async def download_image(self, url: str) -> io.BytesIO:
        """从URL或本地路径下载图片

        HTTP 下载采用指数退避重试（最多 3 次），仅对网络层错误和 5xx 响应重试，
        4xx 客户端错误不重试。
        """
        # 系统存储 URL：用 SDK 带认证下载，避免 minio 私有 bucket 匿名 GET 403
        from app.service.storage.factory import get_storage_service

        storage_service = get_storage_service()
        base_url = storage_service.base_url.rstrip("/")
        if url.startswith(base_url + "/"):
            object_name = url[len(base_url) + 1 :]
            bucket = settings.MINIO_BUCKET_NAME
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None, lambda: storage_service.download(bucket, object_name)
            )
            return io.BytesIO(raw)

        # 处理绝对本地路径（用于离线算法模型本地推理，非生产链路）
        if not url.startswith("http://") and not url.startswith("https://"):
            local_path = Path(url)
            if local_path.exists():
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._read_file_sync, local_path)
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"图片文件不存在: {url}")

        # HTTP/HTTPS 下载（带指数退避重试）
        headers = {}
        trace_id = _trace_id_var.get("")
        if trace_id:
            headers["X-Trace-Id"] = trace_id

        max_retry = 3
        backoff = 1.0  # 初始退避 1 秒
        last_exc: Exception | None = None

        for attempt in range(max_retry + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    return io.BytesIO(response.content)
            except httpx.HTTPStatusError as e:
                # 4xx 客户端错误不重试（请求格式错误，重试无意义）
                # 5xx 服务端错误可重试
                if 400 <= e.response.status_code < 500:
                    raise BusinessException(
                        ResultCode.RESOURCE_NOT_FOUND,
                        f"图片下载失败 ({e.response.status_code}): {url}",
                    ) from e
                last_exc = e
                logger.warning(
                    "图片下载返回 %s (attempt=%s/%s): %s",
                    e.response.status_code,
                    attempt + 1,
                    max_retry + 1,
                    url,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # 网络层错误（连接超时/拒绝/EOF）→ 可重试
                last_exc = e
                logger.warning(
                    "图片下载网络异常 (attempt=%s/%s): %s - %s",
                    attempt + 1,
                    max_retry + 1,
                    url,
                    e,
                )

            if attempt < max_retry:
                await asyncio.sleep(backoff)
                backoff *= 2  # 指数退避

        # 全部重试失败
        raise BusinessException(f"图片下载失败（已重试 {max_retry} 次）: {url} - {last_exc}")

    @staticmethod
    def _read_file_sync(path: Path) -> io.BytesIO:
        """同步读取文件内容（供 run_in_executor 调用）"""
        with open(path, "rb") as f:
            return io.BytesIO(f.read())

    @staticmethod
    def _run_dehaze(
        import_path: str, model_relative_path: str, image_bytes: io.BytesIO
    ) -> io.BytesIO:
        """
        调用算法去雾

        Args:
            import_path: 算法模块导入路径，如 'algorithm.AECRNet.run'，仅用于 importlib
            model_relative_path: 模型权重文件相对路径（sys_algorithm.path），
                                 如 'AECR-Net/NH_train.pk'，用于通过 model_loader 解析本地路径
        """
        # import_path 仅用于 importlib，不再反推文件目录
        module_name = import_path
        if module_name.startswith("algorithm."):
            module_name = module_name[len("algorithm.") :]
        if module_name.endswith(".run"):
            module_name = module_name[: -len(".run")]

        try:
            algo_module = importlib.import_module(f"algorithm.{module_name}.run")
        except ImportError as e:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"算法模块加载失败: algorithm.{module_name}.run, "
                f"请确认 import_path '{import_path}' 是否正确. "
                f"原始错误: {e}",
            ) from None

        if not hasattr(algo_module, "dehaze"):
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"算法模块 {module_name} 未导出 dehaze() 函数",
            )

        dehaze_fn = algo_module.dehaze

        # 通过 model_loader 解析模型权重文件到本地路径
        # 算法 path 字段为空（如 DCP 无权重）时传空字符串，由算法自行处理
        model_path = ""
        if model_relative_path and model_relative_path.strip():
            try:
                model_path = resolve_model_path(model_relative_path)
            except FileNotFoundError as e:
                raise BusinessException(
                    ResultCode.SYSTEM_EXECUTION_ERROR,
                    f"模型权重加载失败: {e}",
                ) from e

        logger.debug("执行去雾: module=%s, model=%s", module_name, model_path)

        # 调用 dehaze 函数（算法内部自行加载权重；异常统一包装为业务错误，
        # 截取摘要避免泄露绝对路径/完整堆栈）
        try:
            result = dehaze_fn(image_bytes, model_path)
        except BusinessException:
            raise
        except Exception as e:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"算法执行失败: {module_name} - {str(e)[:200]}",
            ) from e

        if isinstance(result, io.BytesIO):
            return result
        elif isinstance(result, PIL.Image.Image):
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)
            return buf
        else:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"dehaze() 返回了不支持的类型: {type(result)}",
            )

    async def _upload_result(self, result_bytes: io.BytesIO, algorithm_name: str) -> str:
        """上传预测结果到 MinIO（与文件管理模块共享存储桶）。

        返回 object_name（对象键），URL 由响应层通过 storage.get_url 拼接。
        """
        from app.infrastructure.storage.minio_client import get_minio_client

        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{algorithm_name}_{int(time.time() * 1000)}.png"
        object_name = f"predictions/{date_str}/{filename}"

        data = result_bytes.getvalue()
        loop = asyncio.get_running_loop()

        def _sync_upload():
            client = get_minio_client()
            bucket_name = settings.MINIO_BUCKET_NAME
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
            client.put_object(
                bucket_name,
                object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type="image/png",
            )

        try:
            await loop.run_in_executor(None, _sync_upload)
        except Exception as e:
            logger.error("预测结果上传存储失败: %s", e, exc_info=True)
            raise BusinessException(
                ResultCode.FILE_STORAGE_ERROR, f"结果存储失败: {str(e)}"
            ) from None

        logger.debug("预测结果已上传到存储: %s", object_name)
        return object_name

    async def batch_predict(
        self,
        algorithm_id: int,
        items: list[dict],
        user_id: int,
        skip_quota_check: bool = False,
    ) -> list[dict]:
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
            file_id = item.get("fileId")
            image_url = item.get("imageUrl")
            params = item.get("params")

            try:
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


# 单例
prediction_service = PredictionService()
