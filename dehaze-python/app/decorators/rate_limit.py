"""
接口限流依赖（滑动窗口算法，基于 Redis）

用法:
    from app.decorators import rate_limit

    @router.post("/predict", dependencies=[Depends(rate_limit(times=10, seconds=60))])
    async def predict(...):
        ...

Redis key 设计:
    rate_limit:{user_id|ip}:{method}:{path}  → sorted set，score=时间戳
"""

import logging
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from app.config import settings
from app.core.code import ResultCode
from app.dependencies.auth import UserContext, get_current_user_optional
from app.dependencies.redis import get_redis_client

logger = logging.getLogger(__name__)


def rate_limit(
    times: int = None,
    seconds: int = None,
):
    """
    接口限流依赖（滑动窗口算法）

    Args:
        times: 时间窗口内允许的最大请求次数，默认使用配置 RATE_LIMIT_DEFAULT_TIMES
        seconds: 时间窗口大小（秒），默认使用配置 RATE_LIMIT_DEFAULT_SECONDS

    用法:
        @router.post("/predict", dependencies=[Depends(rate_limit(times=10, seconds=60))])
        async def predict(...):
            ...
    """
    max_times = times if times is not None else settings.RATE_LIMIT_DEFAULT_TIMES
    window_seconds = seconds if seconds is not None else settings.RATE_LIMIT_DEFAULT_SECONDS

    async def _check(
        request: Request,
        user: Optional[UserContext] = Depends(get_current_user_optional),
    ):
        if not settings.RATE_LIMIT_ENABLED:
            return

        # 构建限流 key
        if user:
            identity = f"u:{user.id}"
        else:
            ip = request.client.host if request.client else "unknown"
            identity = f"ip:{ip}"

        path = request.url.path
        method = request.method
        limit_key = f"rate_limit:{identity}:{method}:{path}"

        redis = await get_redis_client()
        if not redis:
            # Redis 不可用时放行（fail open）
            return

        try:
            now = time.time()
            pipe = redis.pipeline()
            # 移除窗口外的旧记录
            pipe.zremrangebyscore(limit_key, 0, now - window_seconds)
            # 添加当前请求
            pipe.zadd(limit_key, {f"{now}": now})
            # 统计窗口内请求数
            pipe.zcard(limit_key)
            # 设置 key 过期时间
            pipe.expire(limit_key, window_seconds)
            results = await pipe.execute()

            count = results[2]
            if count > max_times:
                logger.warning(
                    f"接口限流: identity={identity}, path={path}, "
                    f"count={count}/{max_times}"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=ResultCode.RATE_LIMITING.msg,
                    headers={"Retry-After": str(window_seconds)},
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"限流检查失败，放行: path={path}, error={e}")

    return _check
