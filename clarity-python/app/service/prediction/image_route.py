"""图像统一模型路由（设计 §2.2/§2.4 阶段 3）

图像模型（model_type=image_edit）由统一模型体系承载：
按请求 model（规范 slug）或平台默认模型（`sys_dict: ai_model.default_model.image_edit`）
解析出 `sys_ai_model` + `sys_ai_provider`，交 OpenAI Images 兼容适配器调用。
解析次序：请求显式 model > 平台默认（须 status=1）> 抛业务异常，不回退代码常量。
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

IMAGE_MODEL_TYPE = "image_edit"
IMAGE_DEFAULT_MODEL_DICT_KEY = "default_model.image_edit"
DEFAULT_MODEL_DICT_TYPE = "ai_model"


@dataclass(frozen=True)
class ImageRoute:
    """一次图像调用的统一模型路由结果"""

    model: SysAiModel
    provider: SysAiProvider

    @property
    def model_id(self) -> str:
        return self.model.model_id


async def resolve_image_model(db: AsyncSession, model: str | None) -> ImageRoute:
    """解析图像模型路由：请求显式 model 优先，其次平台默认；不可用即拒绝调用"""
    model_id = (model or "").strip()
    if not model_id:
        model_id = await get_dict_value(
            db, DEFAULT_MODEL_DICT_TYPE, IMAGE_DEFAULT_MODEL_DICT_KEY, None
        )
    if not model_id:
        raise BusinessException(
            ResultCode.BUSINESS_ERROR,
            "平台未配置默认图像模型（字典 ai_model.default_model.image_edit 缺项）",
        )
    row = await ai_model_repository.get_by_model_id(db, model_id)
    if row is None or row.deleted or row.status != 1 or row.model_type != IMAGE_MODEL_TYPE:
        raise BusinessException(
            ResultCode.PARAM_ERROR, f"图像模型不可用: {model_id}（不存在/已停用/类型不符）"
        )
    provider = await ai_provider_repository.get_by_id(db, row.provider_id)
    if provider is None or provider.deleted or provider.status != 1:
        raise BusinessException(ResultCode.BUSINESS_ERROR, f"模型 {model_id} 的供应商不可用")
    return ImageRoute(model=row, provider=provider)
