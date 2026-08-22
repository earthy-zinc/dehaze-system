"""SSRF 防护工具

对外部 URL（Agent Card 地址 / A2A 端点 / 文件引用）做安全校验：
- 仅允许 https（安全默认值，避免明文传输敏感凭证）
- 禁止访问内网地址（127.0.0.1 / 10.x / 172.16-31.x / 192.168.x / ::1 / localhost 等）
- 域名主机解析全部 A 记录，任一命中内网段即拒绝（防 DNS 重绑定绕过）

解析结果做 60s 短 TTL 缓存（防每请求重复 DNS 查询），但不跨请求长期缓存
（DNS 记录会变，长缓存会再次放大重绑定风险）。
"""

import asyncio
import ipaddress
import logging
import socket
import time
from urllib.parse import urlparse

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# 已解析且安全的主机名短缓存：host -> (timestamp, safe)
_RESOLVED_CACHE: dict[str, tuple[float, bool]] = {}
_RESOLVED_TTL_SECONDS = 60


def _is_internal_ip(ip: str) -> bool:
    """判断 IP 是否命中内网/环回/链路本地/保留/组播/未指定地址。"""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _is_hostname(host: str) -> bool:
    """host 是否为非字面 IP 的主机名（IPv4/IPv6 字面量返回 False）。"""
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        return True


def _is_internal_hostname(host: str) -> bool:
    """静态判断：localhost 及字面内网 IP。"""
    host = host.lower().strip("[]")
    if host in ("localhost", "localhost.localdomain", "::1"):
        return True
    if not _is_hostname(host):
        return _is_internal_ip(host)
    return False


async def _resolve_reaches_internal(host: str) -> bool:
    """解析主机名全部地址，任一命中内网段即返回 True；解析失败保守拒绝。"""
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except OSError:
        logger.warning("SSRF 域名解析失败，保守拒绝 %s", host)
        return True
    for info in infos:
        ip = info[4][0]
        if _is_internal_ip(ip):
            return True
    return False


async def is_safe_url(url: str) -> bool:
    """校验 URL 是否安全（https + 非内网），安全返回 True。"""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = parsed.hostname
    if not host:
        return False
    if _is_internal_hostname(host):
        return False
    # 字面公网 IP：已排除内网段，直接放行
    if not _is_hostname(host):
        return True
    # 域名：查短缓存，未命中则解析 A 记录
    now = time.monotonic()
    cached = _RESOLVED_CACHE.get(host)
    if cached and now - cached[0] < _RESOLVED_TTL_SECONDS:
        return cached[1]
    safe = not await _resolve_reaches_internal(host)
    _RESOLVED_CACHE[host] = (now, safe)
    return safe


async def validate_https_url(url: str) -> str:
    """校验并返回安全的外部 URL，不合法抛 400。"""
    if not await is_safe_url(url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL 仅支持 https 且禁止内网地址",
        )
    return url
