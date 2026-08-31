"""云端 Provider 通用基础设施：API Key 选取、认证头构建、HTTP 请求

云端引擎（CloudAsrProvider / CloudTtsProvider）基于本模块的统一能力实现：
- 密钥经 sys_voice_provider_key 加密存储，运行时解密；多 Key 按 priority 优先选取
- 认证头按 sys_voice_provider.auth_type（bearer / x-api-key / custom）组装
- HTTP 请求封装 httpx async，带超时与统一错误转换

具体厂商协议（请求体格式、流式 WebSocket、SK 原生签名）由各厂商 Provider
在其实现中基于 api_base_url + 本模块能力适配。
"""

import httpx

from app.database import get_db_session
from app.infrastructure.crypto.aes_cipher import decrypt
from app.repository.voice_provider_key_repository import voice_provider_key_repository

_HTTP_TIMEOUT = httpx.Timeout(connect=10, read=60, write=30, pool=10)


class CloudBase:
    """云端引擎通用基类"""

    def __init__(self, provider) -> None:
        self._provider = provider

    async def _resolve_key(self) -> str | None:
        """获取启用的 API Key 明文（按 priority 升序取第一个），无 Key 返回 None"""
        async with get_db_session() as db:
            keys = await voice_provider_key_repository.list_enabled_by_provider(
                db, self._provider.id
            )
        return decrypt(keys[0].key_cipher) if keys else None

    def _auth_headers(self, key: str | None) -> dict[str, str]:
        """按 provider.auth_type 组装认证请求头（custom 时从 default_headers.auth_header 取头名）"""
        headers = dict(self._provider.default_headers or {})
        if not key:
            return headers
        auth = self._provider.auth_type or "bearer"
        if auth == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        elif auth == "x-api-key":
            headers["X-API-Key"] = key
        elif auth == "custom":
            header_name = headers.pop("auth_header", None) or "x-api-key"
            headers[header_name] = key
        return headers

    async def _post_json(
        self, url: str, payload: dict, *, key: str | None = None
    ) -> dict:
        """POST JSON 请求，返回响应 JSON（非 2xx 抛 RuntimeError）"""
        headers = {"Content-Type": "application/json", **self._auth_headers(key)}
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _post_audio(
        self, url: str, data: bytes, *, content_type: str, key: str | None = None
    ) -> bytes:
        """POST 二进制音频/文本，返回响应字节"""
        headers = {"Content-Type": content_type, **self._auth_headers(key)}
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(url, content=data, headers=headers)
            resp.raise_for_status()
            return resp.content
