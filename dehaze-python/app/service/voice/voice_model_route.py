"""语音统一模型路由（设计 §2.2/§3.1 阶段 2）

asr/tts 从「语音引擎注册表（sys_voice_*）」切到统一模型体系：按模型类型取平台默认
模型（`sys_dict: ai_model.default_model.{asr|tts}`），解析出 `sys_ai_model` +
`sys_ai_provider` 后交 OpenAI Audio 兼容适配器调用。语音用户端不暴露 model 参数
（契约不变，D7），始终走平台默认模型。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_model import SysAiModel
from app.models.entity.sys_ai_provider import SysAiProvider
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_repository import ai_provider_repository
from app.service.dict_service import get_dict_value

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DICT_TYPE = "ai_model"


@dataclass(frozen=True)
class VoiceRoute:
    """一次语音调用的统一模型路由结果"""

    model: SysAiModel
    provider: SysAiProvider

    @property
    def model_id(self) -> str:
        return self.model.model_id


async def resolve_voice_model(db: AsyncSession, model_type: str) -> VoiceRoute:
    """按类型解析平台默认语音模型（asr/tts），模型缺失/停用/供应商不可用均拒绝调用"""
    default_model_id = await get_dict_value(
        db, DEFAULT_MODEL_DICT_TYPE, f"default_model.{model_type}", None
    )
    if not default_model_id:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"平台未配置默认{model_type}模型（字典缺项）"
        )
    model = await ai_model_repository.get_by_model_id(db, default_model_id)
    if model is None or model.deleted or model.status != 1 or model.model_type != model_type:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            f"默认{model_type}模型不可用: {default_model_id}（不存在/已停用/类型不符）",
        )
    provider = await ai_provider_repository.get_by_id(db, model.provider_id)
    if provider is None or provider.deleted or provider.status != 1:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR, f"模型 {default_model_id} 的供应商不可用"
        )
    return VoiceRoute(model=model, provider=provider)
