"""输入图片获取：系统存储 SDK 下载 / HTTP 下载（指数退避重试 + SSRF 防护）。"""

import asyncio
import io
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.logging import _trace_id_var

logger = logging.getLogger(__name__)


async def _assert_public_host(url: str) -> None:
    """SSRF 防护：外部图片 URL 禁止指向内网/回环/链路本地/保留地址。

    系统存储的下载在 fetch_image 中经 base_url 前缀分支优先处理，不经过此校验。
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise BusinessException(ResultCode.PARAM_ERROR, f"无效的图片URL: {url}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise BusinessException(
            ResultCode.RESOURCE_NOT_FOUND, f"图片地址无法解析: {hostname}"
        ) from e

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise BusinessException(ResultCode.PARAM_ERROR, "图片地址不允许访问内网资源")


async def fetch_image(url: str) -> io.BytesIO:
    """从URL下载图片

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

    # 外部 HTTP/HTTPS 下载（带指数退避重试 + 内网地址防护）
    if not url.startswith("http://") and not url.startswith("https://"):
        raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的图片地址: {url}")

    await _assert_public_host(url)

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
