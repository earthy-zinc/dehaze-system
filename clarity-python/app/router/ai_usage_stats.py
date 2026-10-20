from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.result import Result, success
from app.database import get_db
from app.decorators import require_permission
from app.dependencies.auth import UserContext, get_current_user
from app.dependencies.redis import get_redis
from app.models.schema.ai_provider import UsageStatsQuery, UsageStatsResult
from app.service.ai_usage_stats_service import ai_usage_stats_service

router = APIRouter(prefix="/api/v1/ai", tags=["AI模型管理"])


@router.get(
    "/usage/stats",
    response_model=Result[UsageStatsResult],
    summary="运营统计(供应商健康看板/模型用量分布/降级与故障)",
)
@require_permission("ai:model:manage")
async def get_usage_stats(
    query: UsageStatsQuery = Depends(),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    user: UserContext = Depends(get_current_user),
):
    result = await ai_usage_stats_service.get_usage_stats(db, redis, query)
    return success(result)
