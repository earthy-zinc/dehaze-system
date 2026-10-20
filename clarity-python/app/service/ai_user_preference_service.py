"""用户 AI 偏好服务：偏好默认对话模型的读写。

偏好的**生效**规则不在本服务：解析链（请求指定 > 会话已存 > 用户偏好 > 平台默认）由
`service/ai/service/model_defaults.py` 的 `resolve_default_chat_model` 统一实现，本服务只负责
"用户能设置什么"——写入时按 `GET /api/v1/ai/models/enabled?modelType=chat` 的同一可见性
规则校验（模型存在 + 启用 + model_type=chat + vip_level 不高于用户等级），不满足抛 A0601；
模型停用/删除后无需清理偏好（解析时取不到即回落平台默认）。
"""

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.ai_user_preference_repository import ai_user_preference_repository
from app.service.ai_model_service import ai_model_service


class AiUserPreferenceService:
    async def get_default_model(self, db: AsyncSession, user_id: int) -> str | None:
        """读取偏好默认模型；未设置返回 None（表示跟随平台默认）"""
        preference = await ai_user_preference_repository.get_by_user_id(db, user_id)
        return preference.default_model if preference else None

    async def set_default_model(
        self,
        db: AsyncSession,
        redis: Redis,
        user_id: int,
        model_id: str | None,
    ) -> None:
        """设置（model_id=None 表示清除）偏好默认模型，即时生效

        普通用户只会从 `/ai/models/enabled?modelType=chat` 里选，故复用该接口的可见性
        规则校验：可见即允许设置（等级内、启用、对话型），越权/不存在/停用/非对话模型
        一律 A0601，避免偏好指向一个用户根本用不了的模型。
        """
        if model_id is not None:
            available = await ai_model_service.list_enabled_models(db, redis, user_id, "chat")
            if model_id not in {model.model_id for model in available}:
                raise BusinessException(
                    ResultCode.AI_MODEL_NOT_AVAILABLE, "模型不存在、未启用或当前等级不可用"
                )
        await ai_user_preference_repository.upsert_default_model(
            db, user_id, model_id, actor_id=user_id
        )


ai_user_preference_service = AiUserPreferenceService()
