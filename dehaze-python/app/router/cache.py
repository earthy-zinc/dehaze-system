"""管理端缓存统一失效入口

存在绕过 CacheService 直接删 Redis 的路径（如运维脚本重建库后的清理），
此时各实例 L1 本地缓存不会失效、Pub/Sub 也不广播，导致"改了数据必须重启后端"。
本接口作为统一失效入口：全部清理动作经 CacheService.delete/delete_pattern 执行
（自动清 L2 + 本进程 L1 + 广播其他实例清 L1）。
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from redis.asyncio import Redis

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.core.result import Result, success
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.infrastructure.cache.cache import CacheService

router = APIRouter(
    prefix="/api/v1/cache",
    tags=["缓存管理"],
    dependencies=[Depends(get_current_user)],
)

# 业务缓存清单（与各服务使用的缓存 key 同步维护；不含 session/限流/任务进度等基础设施 key）
_BUSINESS_CACHE_PATTERNS = [
    "menu:routes",
    "role:perms:*",
    "ai:model:list",
    "user:level:*",
    "ai:provider:list",
    "ai:agent:list:enabled",
    "ai:agent:*:published",
    "ai:config:guardrail_defaults",
    "ai:rate:*",
    "ai:bill:*",
    "dept:tree*",
    "dept:options*",
    "dict:options:*",
    "dict:value:*",
    "kb:list:*",
    "kb:detail:*",
    "kb:config:*",
    "kb:search:*",
    "rating:stats:*",
    "feedback:stats",
    "package:onsale:*",
    "package:detail:*",
    "member:benefit*",
    "order:detail:*",
]


class CacheClearForm(BaseModel):
    key: str | None = None
    pattern: str | None = None


@router.post("/clear", response_model=Result[list[dict]], summary="清除业务缓存（仅 ROOT/ADMIN）")
async def clear_cache(
    form: CacheClearForm | None = None,
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    # 缓存清理是全局运维操作，按角色控制而非权限标识（不新增菜单/权限种子）
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, ResultCode.ACCESS_UNAUTHORIZED.msg)

    if form and form.key and form.pattern:
        raise BusinessException(ResultCode.PARAM_ERROR, "key 与 pattern 二选一，不可同时指定")
    if form and form.pattern == "*":
        # pattern=* 等价于清库内所有 key（含 session/限流等基础设施 key），
        # 全量清业务缓存一律走空 body 的枚举路径
        raise BusinessException(
            ResultCode.PARAM_ERROR, "pattern 不允许为 *；全量清业务缓存请传空 body"
        )

    cache = CacheService(redis)
    results: list[dict] = []
    if form and form.key:
        await cache.delete(form.key)
        results.append({"target": form.key, "deleted": 1})
    elif form and form.pattern:
        deleted = await cache.delete_pattern(form.pattern)
        results.append({"target": form.pattern, "deleted": deleted})
    else:
        # 缺省清全部业务缓存：只枚举已知业务前缀，避免 SCAN 到 session/限流等基础设施 key
        for pattern in _BUSINESS_CACHE_PATTERNS:
            deleted = await cache.delete_pattern(pattern)
            results.append({"target": pattern, "deleted": deleted})
    return success(results)
