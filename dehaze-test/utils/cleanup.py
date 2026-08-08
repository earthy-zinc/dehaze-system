"""调试工具库：限流 / 验证码 / session 清理。

对齐 dehaze-sdk-js/test/utils/cleanup.ts：
- 清理登录限流（rate:limit:login:* 等）
- 清理验证码缓存（captcha_code:*）
- 清理 Redis session（session:*）
- 清理业务缓存（msg:unread:* / role:perms:* / user:auth:*）
"""
from __future__ import annotations

from . import redis


# 限流 key 前缀（与后端 RateLimitAspect 对齐）
RATE_LIMIT_PATTERNS = [
    "rate:limit:login:*",
    "rate:limit:/api/v1/auth/login:*",
    "rate:limit:/api/v1/auth/register:*",
]

# 验证码 key 前缀
CAPTCHA_PATTERNS = ["captcha_code:*"]

# session key 前缀
SESSION_PATTERNS = ["session:*"]

# 业务缓存 key 前缀
BUSINESS_CACHE_PATTERNS = [
    "msg:unread:*",
    "role:perms:*",
    "user:auth:*",
]


def clear_login_rate_limit() -> int:
    """清理登录限流，返回删除的 key 数量。"""
    total = 0
    for pattern in RATE_LIMIT_PATTERNS:
        total += redis.delete_pattern(pattern)
    return total


def clear_captcha() -> int:
    """清理所有验证码缓存。"""
    total = 0
    for pattern in CAPTCHA_PATTERNS:
        total += redis.delete_pattern(pattern)
    return total


def clear_sessions() -> int:
    """清理所有 Redis session。"""
    total = 0
    for pattern in SESSION_PATTERNS:
        total += redis.delete_pattern(pattern)
    return total


def clear_business_cache() -> int:
    """清理业务缓存（消息未读数、角色权限、用户认证信息等）。"""
    total = 0
    for pattern in BUSINESS_CACHE_PATTERNS:
        total += redis.delete_pattern(pattern)
    return total


def clear_all() -> dict[str, int]:
    """清理全部（限流 + 验证码 + session + 业务缓存），返回分类删除数量。"""
    return {
        "rate_limit": clear_login_rate_limit(),
        "captcha": clear_captcha(),
        "session": clear_sessions(),
        "business_cache": clear_business_cache(),
    }
