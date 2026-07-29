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
from typing import Optional

import httpx
import PIL.Image

from app.database import async_session_factory, get_db_session
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.infrastructure.logging import _trace_id_var
from app.infrastructure.metrics.inference_metrics import record_inference_metrics
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.models.entity.sys_file import SysFile
from app.models.enum.log_status import LogStatus
from app.repository.algorithm_repository import algorithm_repository
from app.repository.file_repository import file_repository
from app.repository.pred_eval_log_repository import pred_log_repository
from app.service.prediction import (
    InterceptedResult,
    PredictionContext,
    PredictionInterceptorChain,
)
from app.service.prediction.wpxnet_interceptor import WpxNetPredictionInterceptor
from app.utils.file import calculate_bytes_md5
from algorithm.model_loader import resolve_model_path

logger = logging.getLogger(__name__)

# 预测结果缓存 TTL：24 小时
PREDICTION_CACHE_TTL = 24 * 60 * 60

# 算法推理专用线程池：PyTorch 推理为 CPU 密集型同步操作，
# 必须在线程池中执行以避免阻塞 asyncio 事件循环。
# 限制并发数避免线程过多导致 GIL 争抢和内存爆炸。
_inference_executor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="algo-inference")


def _build_interceptor_chain() -> PredictionInterceptorChain:
    """构建预测拦截器责任链（新增插件在此注册）"""
    return PredictionInterceptorChain([
        WpxNetPredictionInterceptor(),
    ])


class PredictionService:
    """模型预测编排服务（异步任务模式）"""

    def __init__(self):
        self._interceptor_chain = _build_interceptor_chain()

    async def predict(
        self,
        algorithm_id: int,
        image_url: str,
        params: Optional[dict] = None,
        user_id: Optional[int] = None,
        file_id: Optional[int] = None,
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

        # 2. fileId 存在时查询原始文件并用其真实 URL（对齐 Java resolveImageUrl）
        origin_file: Optional[SysFile] = None
        if file_id is not None:
            async with async_session_factory() as db:
                origin_file = await file_repository.get_by_id(db, file_id)
            if origin_file is None:
                raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"文件不存在: {file_id}")
            image_url = origin_file.url

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
            logger.info(
                "预测拦截器命中: algorithmId=%s, resultUrl=%s",
                algorithm_id, intercepted.result_url,
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

        # 4. 下载输入图片
        image_bytes = await self.download_image(image_url)
        image_md5 = calculate_bytes_md5(image_bytes)

        # 5. 查询 Redis 缓存（基于 algorithmId + imageMd5）
        cache_key = f"prediction:{algorithm_id}:{image_md5}"
        cached = await self._get_cached_prediction(cache_key)
        if cached is not None:
            elapsed = int((time.time() - start) * 1000)
            logger.info("预测缓存命中: algorithmId=%s, md5=%s", algorithm_id, image_md5)
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
        loop.create_task(self._execute_async(
            log_id=log_id,
            algorithm_id=algorithm_id,
            image_bytes=image_bytes,
            image_md5=image_md5,
            algorithm=algorithm,
            cache_key=cache_key,
            user_id=user_id,
        ))

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
        user_id: Optional[int] = None,
        origin_file_id: Optional[int] = None,
        pred_file_id: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """写 completed 日志并返回完整结果（拦截器命中 / 缓存命中共用）"""
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
        user_id: Optional[int] = None,
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

            # 3. 上传结果
            result_url = await self._upload_result(result_bytes, algorithm.name)
            result_md5 = calculate_bytes_md5(result_bytes)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # 4. 更新日志为 completed
            async with get_db_session() as db:
                await pred_log_repository.update_result(
                    db=db,
                    log_id=log_id,
                    pred_md5=result_md5,
                    pred_url=result_url,
                    time_ms=elapsed_ms,
                )

            # 5. 写入 Redis 缓存
            await self._set_cached_prediction(cache_key, {
                "resultUrl": result_url,
                "resultMd5": result_md5,
                "resultThumbnailUrl": None,
                "format": "png",
            })

            logger.info(
                "异步预测完成: logId=%s, algorithmId=%s, time=%sms",
                log_id, algorithm_id, elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(
                "异步预测失败: logId=%s, algorithmId=%s, error=%s",
                log_id, algorithm_id, error_msg, exc_info=True,
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
                    log_id, update_err,
                )
        finally:
            set_current_user_id(None)

    async def _get_cached_prediction(self, cache_key: str) -> Optional[dict]:
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

    @staticmethod
    async def get_algorithm(algorithm_id: int) -> SysAlgorithm:
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
        # 处理本地文件路径 /api/v1/files/download/... → upload/...
        if url.startswith("/api/v1/files/download/"):
            local_path = url[len("/api/v1/files/download/"):]
            full_path = Path("upload") / local_path
            if full_path.exists():
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, self._read_file_sync, full_path)
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"本地文件不存在: {full_path}")

        # 处理绝对本地路径
        if not url.startswith("http://") and not url.startswith("https://"):
            local_path = Path(url)
            if local_path.exists():
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, self._read_file_sync, local_path)
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, f"图片文件不存在: {url}")

        # HTTP/HTTPS 下载（带指数退避重试）
        headers = {}
        trace_id = _trace_id_var.get("")
        if trace_id:
            headers["X-Trace-Id"] = trace_id

        max_retry = 3
        backoff = 1.0  # 初始退避 1 秒
        last_exc: Optional[Exception] = None

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
                    e.response.status_code, attempt + 1, max_retry + 1, url,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # 网络层错误（连接超时/拒绝/EOF）→ 可重试
                last_exc = e
                logger.warning(
                    "图片下载网络异常 (attempt=%s/%s): %s - %s",
                    attempt + 1, max_retry + 1, url, e,
                )

            if attempt < max_retry:
                await asyncio.sleep(backoff)
                backoff *= 2  # 指数退避

        # 全部重试失败
        raise BusinessException(
            f"图片下载失败（已重试 {max_retry} 次）: {url} - {last_exc}"
        )

    @staticmethod
    def _read_file_sync(path: Path) -> io.BytesIO:
        """同步读取文件内容（供 run_in_executor 调用）"""
        with open(path, "rb") as f:
            return io.BytesIO(f.read())

    @staticmethod
    def _run_dehaze(import_path: str, model_relative_path: str, image_bytes: io.BytesIO) -> io.BytesIO:
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
            module_name = module_name[len("algorithm."):]
        if module_name.endswith(".run"):
            module_name = module_name[:-len(".run")]

        try:
            algo_module = importlib.import_module(f"algorithm.{module_name}.run")
        except ImportError as e:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"算法模块加载失败: algorithm.{module_name}.run, "
                f"请确认 import_path '{import_path}' 是否正确. "
                f"原始错误: {e}",
            )

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

        logger.info("执行去雾: module=%s, model=%s", module_name, model_path)

        # 调用 dehaze 函数
        result = dehaze_fn(image_bytes, model_path)

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
        """上传结果到 MinIO（与文件管理模块共享存储桶），由文件下载接口统一提供服务"""
        from app.config import settings
        from app.service.file_service import get_minio_client

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
                ResultCode.FILE_STORAGE_ERROR, f"结果存储失败: {str(e)}")

        logger.info("预测结果已上传到存储: %s", object_name)
        # 与 Java 端 file.baseUrl 风格一致：FILE_BASE_URL 配置后返回绝对 URL，
        # 留空时回退为相对路径 /api/v1/files/download/...
        base_url = settings.FILE_BASE_URL.rstrip("/")
        if base_url:
            return f"{base_url}/{object_name}"
        return f"/api/v1/files/download/{object_name}"


# 单例
prediction_service = PredictionService()
