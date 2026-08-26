"""输入图片获取：系统存储 SDK 下载 / 本地路径 / HTTP 下载（指数退避重试）。"""

import asyncio
import io
import logging
from pathlib import Path

import httpx

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.logging import _trace_id_var

logger = logging.getLogger(__name__)


async def fetch_image(url: str) -> io.BytesIO:
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
        bucket = settings.MINIO_BUCKET
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
            return await loop.run_in_executor(None, _read_file_sync, local_path)
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


def _read_file_sync(path: Path) -> io.BytesIO:
    """同步读取文件内容（供 run_in_executor 调用）"""
    with open(path, "rb") as f:
        return io.BytesIO(f.read())
