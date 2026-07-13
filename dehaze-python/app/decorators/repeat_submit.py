"""
防重复提交依赖（基于 Redis SETNX）

原理：以 (用户ID|IP + 路径 + 请求体哈希) 为 key 写入 Redis，
若 key 已存在则判定为重复提交，拒绝请求。

用法:
    from app.decorators import repeat_submit

    @router.post("/create", dependencies=[Depends(repeat_submit(interval=5))])
    async def create(...):
        ...

Redis key 设计:
    repeat_submit:{user_id|ip}:{method}:{path}:{body_hash}  → SETNX + TTL
"""

import hashlib
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.auth import UserContext, get_current_user_optional
from app.dependencies.redis import get_redis_client

logger = logging.getLogger(__name__)


def _hash_request_body(body: bytes) -> str:
    """计算请求体哈希（MD5，取前 16 位）"""
    return hashlib.md5(body).hexdigest()[:16]


async def _read_request_body(request: Request) -> bytes:
    """读取请求体（需兼容中间件已消费 body 的情况）"""
    try:
        return await request.body()
    except Exception:
        return b""


def repeat_submit(interval: int = None):
    """
    防重复提交依赖

    Args:
        interval: 防重复间隔（秒），在该时间窗口内相同请求被视为重复提交。
                  默认使用配置 REPEAT_SUBMIT_DEFAULT_INTERVAL

    用法:
        @router.post("/create", dependencies=[Depends(repeat_submit(interval=5))])
        async def create(...):
            ...
    """
    lock_interval = interval if interval is not None else settings.REPEAT_SUBMIT_DEFAULT_INTERVAL

    async def _check(
        request: Request,
        user: Optional[UserContext] = Depends(get_current_user_optional),
    ):
        if not settings.REPEAT_SUBMIT_ENABLED:
            return

        # GET / DELETE 等无 body 请求不做防重复
        if request.method in ("GET", "DELETE", "HEAD", "OPTIONS"):
            return

        # 构建防重复 key
        if user:
            identity = f"u:{user.id}"
        else:
            ip = request.client.host if request.client else "unknown"
            identity = f"ip:{ip}"

        path = request.url.path
        method = request.method

        # 读取并哈希请求体
        body = await _read_request_body(request)
        body_hash = _hash_request_body(body) if body else "no_body"

        lock_key = f"repeat_submit:{identity}:{method}:{path}:{body_hash}"

        redis = await get_redis_client()
        if not redis:
            # Redis 不可用时放行
            return

        try:
            # SETNX：设置成功返回 True（非重复），已存在返回 False（重复）
            acquired = await redis.set(
                lock_key,
                "1",
                ex=lock_interval,
                nx=True,
            )

            if not acquired:
                logger.warning(
                    f"防重复提交拦截: identity={identity}, path={path}, "
                    f"interval={lock_interval}s"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=ResultCode.REPEAT_SUBMIT_ERROR.msg,
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"防重复提交检查失败，放行: path={path}, error={e}")

    return _check
