"""内置本地模型的数据播种（幂等）

启动时确保 local provider、占位 API Key、内置模型记录存在，使本地模型开箱即用：

- LLM（qwen3-0.6b）：状态启用，供对话路由（AI_DEFAULT_MODEL=qwen3-0.6b）使用；
- Embedding（qwen3-embedding-0.6b）：状态停用，仅作统一模型注册表记录，
  不进入对话模型列表；向量调用由 embedding_client 按 sys_ai_provider
  api_base_url 直连 local provider 的 /v1/embeddings 端点，不依赖该记录。

TTS/ASR 为进程内引擎（Piper/FunASR，库内推理，无供应商路由语义），
由 config 中 VOICE_TTS_* / VOICE_ASR_* 配置驱动，不进入模型注册表。
"""

import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.infrastructure.crypto.aes_cipher import encrypt, hash_key
from app.models.entity import SysAiModel, SysAiModelPrice, SysAiProvider, SysAiProviderKey
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_key_repository import ai_provider_key_repository
from app.repository.ai_provider_repository import ai_provider_repository

logger = logging.getLogger(__name__)

LOCAL_PROVIDER_CODE = "local"
LOCAL_MODEL_ID = "qwen3-0.6b"
LOCAL_EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b"
_LOCAL_KEY_PLAINTEXT = "local-no-key"  # 本地服务不校验鉴权，占位走 Key 轮换链路


async def ensure_local_models(db: AsyncSession) -> None:
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

    # 对话 LLM：状态启用，供模型路由与降级链使用
    if await ai_model_repository.get_by_model_and_provider(db, LOCAL_MODEL_ID, provider.id) is None:
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id=LOCAL_MODEL_ID,
                display_name="Qwen3-0.6B（内置本地）",
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
        await _ensure_local_free_price(db, LOCAL_MODEL_ID, provider.id)
        logger.info("播种 qwen3-0.6b 模型（对话兜底）")

    # Embedding：状态停用，仅登记模型目录（避免出现在对话模型列表）；
    # 实际向量调用由 embedding_client 按 local provider 的 api_base_url 直连
    if await ai_model_repository.get_by_model_and_provider(db, LOCAL_EMBEDDING_MODEL_ID, provider.id) is None:
        db.add(
            SysAiModel(
                provider_id=provider.id,
                model_id=LOCAL_EMBEDDING_MODEL_ID,
                display_name="Qwen3-Embedding-0.6B（内置本地，向量）",
                max_context_tokens=4096,
                max_output_tokens=0,
                supports_multimodal=0,
                supports_tool_call=0,
                supports_streaming=0,
                supports_prompt_cache=0,
                supports_structured_output=0,
                prompt_cache_prefix_len=0,
                status=0,
                vip_level=0,
            )
        )
        await _ensure_local_free_price(db, LOCAL_EMBEDDING_MODEL_ID, provider.id)
        logger.info("播种 qwen3-embedding-0.6b 模型（向量注册表记录）")


async def _ensure_local_free_price(db: AsyncSession, model_id: str, provider_id: int) -> None:
    """本地模型免费：播种全 0 价价格版本（幂等）。

    全 0 单价按不扣积分处理（见 AI模型管理 §2.12），使本地兜底模型结算不抛"未配置售价"。
    """
    if await ai_model_price_repository.get_effective_version(
        db, model_id, provider_id, datetime.now()
    ) is not None:
        return
    price = await ai_model_price_repository.create(
        db,
        SysAiModelPrice(
            model_id=model_id,
            provider_id=provider_id,
            price_version=1,
            unit="credits_per_million",
            effective_from=datetime.now(),
            status=1,
        ),
    )
    await ai_model_price_repository.create_details(
        db, price.id,
        [
            {"token_type": t, "time_slot": s, "min_tokens": 0, "max_tokens": None, "unit_price": 0}
            for t in ("input", "cached", "output")
            for s in ("idle", "peak")
        ],
    )
    logger.info("播种 %s 免费价格版本（全 0 价）", model_id)
