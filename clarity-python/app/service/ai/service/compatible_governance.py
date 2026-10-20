"""第三方兼容 API 接入治理服务（F-M08-010 §2.3）

Key 级配额（日/月/RPM）与模型白名单治理，与用户积分配额双轨控制。

配额计数走 Redis 固定窗口：
- apikey:{key_id}:daily:{yyyyMMdd}    TTL 48h
- apikey:{key_id}:monthly:{yyyyMM}    TTL 35d
- apikey:{key_id}:rpm:{yyyyMMddHHmm}  TTL 65s

时间窗口使用 Asia/Shanghai 本地日期格式化，与用户配额重置时区一致。
配额检查在 INCR 之后判定：保证计数累计但超限时拒绝执行。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entity.api_key import SysApiKey

_TZ = ZoneInfo("Asia/Shanghai")

_DAILY_TTL = 48 * 3600
_MONTHLY_TTL = 35 * 24 * 3600
_RPM_TTL = 65


class GovernanceError(Exception):
    """接入治理异常：配额超限/白名单拒绝。

    签名由兼容协议适配层（compatible_openai/compatible_claude）依赖，不得更改。
    """

    def __init__(self, status_code: int, error_type: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


class CompatibleGovernanceService:
    @staticmethod
    async def precheck(
        redis: Redis,
        api_key: SysApiKey,
        model: str | None,
        endpoint: str,
    ) -> None:
        """请求入口治理预检：Key 级配额（日/月/RPM）与模型白名单。

        - 配额为 NULL 或 0 表示不限制，跳过对应检查；
        - 配额检查在 INCR 之后判定（计数累计但超限拒绝）；
        - model 为 None 时跳过白名单（走会话默认模型）。
        """
        now = datetime.now(_TZ)
        await CompatibleGovernanceService._check_quota(
            redis,
            api_key,
            f"apikey:{api_key.id}:daily:{now:%Y%m%d}",
            api_key.daily_quota,
            _DAILY_TTL,
        )
        await CompatibleGovernanceService._check_quota(
            redis,
            api_key,
            f"apikey:{api_key.id}:monthly:{now:%Y%m}",
            api_key.monthly_quota,
            _MONTHLY_TTL,
        )
        await CompatibleGovernanceService._check_quota(
            redis, api_key, f"apikey:{api_key.id}:rpm:{now:%Y%m%d%H%M}", api_key.rpm_limit, _RPM_TTL
        )
        if model is not None:
            CompatibleGovernanceService._check_whitelist(api_key, model)

    @staticmethod
    async def _check_quota(
        redis: Redis,
        api_key: SysApiKey,
        key: str,
        quota: int | None,
        ttl: int,
    ) -> None:
        if not quota:
            return
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, ttl)
        if count > quota:
            raise GovernanceError(429, "rate_limit_error", "调用频率超出该 API Key 的配额限制")

    @staticmethod
    def _check_whitelist(api_key: SysApiKey, model: str) -> None:
        whitelist = api_key.model_whitelist
        if not whitelist:
            return
        if model not in whitelist:
            raise GovernanceError(
                403, "permission_error", f"模型 {model} 不在该 API Key 的白名单内"
            )

    @staticmethod
    async def check_model_allowed(db: AsyncSession, api_key: SysApiKey, model: str) -> None:
        """适配层解析出实际模型后的二次白名单校验（同白名单逻辑）。"""
        CompatibleGovernanceService._check_whitelist(api_key, model)

    @staticmethod
    async def filter_models(
        db: AsyncSession,
        api_key: SysApiKey | None,
        models: list,
    ) -> list:
        """/v1/models 白名单过滤：白名单为 NULL 或空时不过滤。

        元素可为字符串 model_id，或含 model_id 属性的模型实体。
        """
        if api_key is None or not api_key.model_whitelist:
            return models
        return [
            m
            for m in models
            if (m if isinstance(m, str) else m.model_id) in api_key.model_whitelist
        ]


compatible_governance_service = CompatibleGovernanceService()
