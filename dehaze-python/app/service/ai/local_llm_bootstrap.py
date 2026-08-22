"""内置本地轻量 LLM 的数据播种（幂等）

启动时确保 local provider、占位 API Key、qwen3-0.6b 模型记录存在，
使内置 LLM 开箱即用（AI_DEFAULT_MODEL=qwen3-0.6b 的路由目标）。
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.infrastructure.crypto.aes_cipher import encrypt, hash_key
from app.models.entity import SysAiModel, SysAiProvider, SysAiProviderKey
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_key_repository import ai_provider_key_repository
from app.repository.ai_provider_repository import ai_provider_repository

logger = logging.getLogger(__name__)

LOCAL_PROVIDER_CODE = "local"
LOCAL_MODEL_ID = "qwen3-0.6b"
_LOCAL_KEY_PLAINTEXT = "local-no-key"  # 本地服务不校验鉴权，占位走 Key 轮换链路


async def ensure_local_llm(db: AsyncSession) -> None:
    """幂等播种 local provider / key / 模型；已存在则按需补齐缺失部分"""
    base_url = f"http://{settings.LOCAL_LLM_HOST}:{settings.LOCAL_LLM_PORT}/v1"
    provider = await ai_provider_repository.get_by_provider_code(db, LOCAL_PROVIDER_CODE)
    if provider is None:
        provider = SysAiProvider(
            provider_code=LOCAL_PROVIDER_CODE,
            display_name="内置本地模型",
            api_base_url=base_url,
            protocol_type="openai_compat",
            auth_type="bearer",
            status=1,
        )
        db.add(provider)
        await db.flush()
        logger.info("播种 local provider（id=%s）", provider.id)
    elif provider.api_base_url != base_url:
        provider.api_base_url = base_url  # 端口配置变更时同步
        await db.flush()

    # 占位 Key：让 local provider 走与第三方一致的 Key 轮换/健康度链路
    key_hash = hash_key(_LOCAL_KEY_PLAINTEXT)
    if await ai_provider_key_repository.get_by_hash(db, key_hash) is None:
        db.add(
            SysAiProviderKey(
                provider_id=provider.id,
                name="内置占位（本地服务不校验鉴权）",
                key_hash=key_hash,
                key_prefix="local",
                key_cipher=encrypt(_LOCAL_KEY_PLAINTEXT),
                status=1,
                priority=1,
                weight=1,
            )
        )
        logger.info("播种 local provider 占位 Key")

    if await ai_model_repository.get_by_model_and_provider(db, LOCAL_MODEL_ID, provider.id) is None:
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id=LOCAL_MODEL_ID,
                display_name="Qwen3-0.6B（内置本地）",
                input_rate=0.0,
                output_rate=0.0,
                max_context_tokens=8192,
                max_output_tokens=2048,
                supports_multimodal=0,
                supports_tool_call=1,
                supports_streaming=1,
                supports_prompt_cache=0,
                supports_structured_output=0,
                prompt_cache_prefix_len=0,
                status=1,
                vip_level=0,
            )
        )
        logger.info("播种 qwen3-0.6b 模型（fallback 兜底）")
