"""调试工具库：直接连接 Redis，不依赖本地 docker。

对齐 dehaze-sdk-js/test/utils/redis.ts：
- 单例 Redis 客户端（首次调用时建立连接）
- redis-py 同步客户端（同步 API 适合调试脚本，避免 async 样板）
- decode_responses=True，直接拿到 str 而非 bytes
"""
from __future__ import annotations

import redis

from . import config


_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """获取单例 Redis 客户端。"""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            db=config.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _client


def disconnect_redis() -> None:
    """关闭 Redis 连接。"""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ===== 业务相关便捷操作（按需扩展） =====

def get_captcha(captcha_key: str) -> str | None:
    """根据 captchaKey 从 Redis 读取验证码明文（key 格式: captcha_code:{captchaKey}）。"""
    return get_redis().get(f"captcha_code:{captcha_key}")


def scan_keys(pattern: str, count: int = 1000) -> list[str]:
    """扫描匹配 pattern 的所有 key（用 scan_iter 避免阻塞）。"""
    return list(get_redis().scan_iter(match=pattern, count=count))


def delete_pattern(pattern: str) -> int:
    """删除所有匹配 pattern 的 key，返回删除数量。"""
    keys = scan_keys(pattern)
    if not keys:
        return 0
    return get_redis().delete(*keys)
