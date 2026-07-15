"""
预测服务 —— 编排算法预测流程

调用算法模块的 dehaze() 函数，管理输入/输出/存储
- 基于 (algorithmId, imageMd5) 的 Redis 缓存（24h TTL）
- 预测日志写入 sys_pred_log
"""

import asyncio
import importlib
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import PIL.Image

from app.database import async_session_factory
from app.dependencies.redis import get_redis_client
from app.infrastructure.cache.redis_fallback import redis_operation_with_fallback
from app.infrastructure.logging import _trace_id_var
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_algorithm import SysAlgorithm
from app.repository.algorithm_repository import algorithm_repository
from app.repository.pred_eval_log_repository import pred_log_repository
from app.utils.file import calculate_bytes_md5
from algorithm.config import algorithm_config

logger = logging.getLogger(__name__)

# 预测结果缓存 TTL：24 小时
PREDICTION_CACHE_TTL = 24 * 60 * 60

# 算法推理专用线程池：PyTorch 推理为 CPU 密集型同步操作，
# 必须在线程池中执行以避免阻塞 asyncio 事件循环。
# 限制并发数避免线程过多导致 GIL 争抢和内存爆炸。
_inference_executor = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="algo-inference")


class PredictionService:
    """模型预测编排服务"""

    # 算法模块名 → 目录名的映射
    ALGORITHM_REGISTRY = {
        "DCP": "DCP",
        "AODNet": "AODNet",
        "DehazeFormer": "DehazeFormer",
        "FFANet": "FFANet",
        "GridDehazeNet": "GridDehazeNet",
        "MSBDN": "MSBDN",
        "RIDCP": "RIDCP",
        "DarkChannelPrior": "DCP",
    }

    async def predict(
        self,
        algorithm_id: int,
        image_url: str,
        params: Optional[dict] = None,
        user_id: Optional[int] = None,
    ) -> dict:
        """
        执行单次预测（带缓存）

        Args:
            algorithm_id: 算法ID
            image_url: 输入图片的完整URL
            params: 可选参数字典
            user_id: 用户ID（用于日志记录）

        Returns:
            {
                "resultUrl": str,
                "resultMd5": str,
                "resultThumbnailUrl": Optional[str],
                "format": str,
                "logId": Optional[int],
                "fromCache": bool,
                "time": int (毫秒),
            }
        """
        start = time.time()

        # 1. 从数据库获取算法信息
        algorithm = await self.get_algorithm(algorithm_id)

        # 2. 下载输入图片
        image_bytes = await self.download_image(image_url)
        image_md5 = calculate_bytes_md5(image_bytes)

        # 3. 查询 Redis 缓存（基于 algorithmId + imageMd5）
        cache_key = f"prediction:{algorithm_id}:{image_md5}"
        cached = await self._get_cached_prediction(cache_key)
        if cached is not None:
            elapsed = int((time.time() - start) * 1000)
            logger.info(f"预测缓存命中: algorithmId={algorithm_id}, md5={image_md5}")

            # 写入日志（缓存命中也记录，便于审计）
            log_id = await self._write_pred_log(
                algorithm_id=algorithm_id,
                origin_md5=image_md5,
                origin_url=image_url,
                pred_md5=cached.get("resultMd5", ""),
                pred_url=cached["resultUrl"],
                time_ms=elapsed,
                user_id=user_id,
            )

            return {
                "resultUrl": cached["resultUrl"],
                "resultMd5": cached.get("resultMd5", ""),
                "resultThumbnailUrl": cached.get("resultThumbnailUrl"),
                "format": cached.get("format", "png"),
                "logId": log_id,
                "fromCache": True,
                "time": elapsed,
            }

        # 4. 缓存未命中，调用算法去雾（在线程池中执行，避免阻塞事件循环）
        loop = asyncio.get_running_loop()
        result_bytes = await loop.run_in_executor(
            _inference_executor,
            self._run_dehaze,
            algorithm.import_path or algorithm.name,
            image_bytes,
        )

        # 5. 上传结果图片
        result_url = await self._upload_result(result_bytes, algorithm.name)
        result_md5 = calculate_bytes_md5(result_bytes)

        elapsed = int((time.time() - start) * 1000)

        # 6. 写入预测日志
        log_id = await self._write_pred_log(
            algorithm_id=algorithm_id,
            origin_md5=image_md5,
            origin_url=image_url,
            pred_md5=result_md5,
            pred_url=result_url,
            time_ms=elapsed,
            user_id=user_id,
        )

        # 7. 写入 Redis 缓存
        await self._set_cached_prediction(cache_key, {
            "resultUrl": result_url,
            "resultMd5": result_md5,
            "resultThumbnailUrl": None,
            "format": "png",
        })

        return {
            "resultUrl": result_url,
            "resultMd5": result_md5,
            "resultThumbnailUrl": None,
            "format": "png",
            "logId": log_id,
            "fromCache": False,
            "time": elapsed,
        }

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

    async def _write_pred_log(
        self,
        algorithm_id: int,
        origin_md5: str,
        origin_url: str,
        pred_md5: str,
        pred_url: str,
        time_ms: int,
        user_id: Optional[int] = None,
    ) -> Optional[int]:
        """写入预测日志"""
        from app.models.base import set_current_user_id
        set_current_user_id(user_id)
        try:
            async with async_session_factory() as db:
                log = await pred_log_repository.create_log(
                    db=db,
                    algorithm_id=algorithm_id,
                    origin_md5=origin_md5,
                    origin_url=origin_url,
                    pred_md5=pred_md5,
                    pred_url=pred_url,
                    time_ms=time_ms,
                )
                await db.commit()
                return log.id
        except Exception as e:
            logger.warning(f"写入预测日志失败: {e}")
            return None
        finally:
            set_current_user_id(None)

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
            # 尝试相对于 upload 目录
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
                    f"图片下载返回 {e.response.status_code} (attempt={attempt + 1}/{max_retry + 1}): {url}"
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # 网络层错误（连接超时/拒绝/EOF）→ 可重试
                last_exc = e
                logger.warning(
                    f"图片下载网络异常 (attempt={attempt + 1}/{max_retry + 1}): {url} - {e}"
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
    def _run_dehaze(import_path: str, image_bytes: io.BytesIO) -> io.BytesIO:
        """
        调用算法去雾
        """
        # 映射到实际模块名
        module_name = PredictionService.ALGORITHM_REGISTRY.get(
            import_path, import_path
        )

        try:
            algo_module = importlib.import_module(f"algorithm.{module_name}.run")
        except ImportError:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"算法模块未找到: algorithm.{module_name}.run, "
                f"请确认 import_path '{import_path}' 是否正确",
            )

        if not hasattr(algo_module, "dehaze"):
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR,
                f"算法模块 {module_name} 未导出 dehaze() 函数",
            )

        dehaze_fn = algo_module.dehaze

        # 获取模型路径
        model_dir = Path(algorithm_config.MODEL_PATH) / module_name
        model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
        model_path = str(model_files[0]) if model_files else ""

        logger.info(f"执行去雾: module={module_name}, model={model_path}")

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
        """上传结果到本地文件存储，由文件管理接口提供服务"""
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{algorithm_name}_{int(time.time() * 1000)}.png"

        upload_dir = Path("upload/predictions") / date_str
        loop = asyncio.get_running_loop()
        dest_path = await loop.run_in_executor(
            None, self._write_file_sync, upload_dir, filename, result_bytes.getvalue())

        logger.info(f"预测结果已保存: {dest_path}")
        return f"/api/v1/files/download/predictions/{date_str}/{filename}"

    @staticmethod
    def _write_file_sync(upload_dir: Path, filename: str, data: bytes) -> Path:
        """同步写入文件（供 run_in_executor 调用）"""
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_path = upload_dir / filename
        with open(dest_path, "wb") as f:
            f.write(data)
        return dest_path


# 单例
prediction_service = PredictionService()
