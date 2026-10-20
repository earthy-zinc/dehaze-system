"""本地模型播种测试：embedding/rerank 模型必须显式 model_type；voice/modal provider 与密钥播种。

schema 默认 model_type=chat，播种漏传会把本地向量/重排模型登记成对话类型，
污染模型类型筛选与目录展示，此处锁定播种语义与幂等性；阶段 1 起 local-voice /
local-modal 供应商及其服务密钥由播种保证存在（同一个 provider 各自一行 Key）。
"""

from datetime import datetime
from decimal import Decimal

import pytest
from sqlalchemy import delete, func, select

from app.config import settings
from app.infrastructure.crypto.aes_cipher import hash_key
from app.infrastructure.llm.local.model_seeder import (
    LOCAL_EMBEDDING_MODEL_ID,
    LOCAL_MODAL_PROVIDER_CODE,
    LOCAL_MODEL_ID,
    LOCAL_RERANK_MODEL_ID,
    LOCAL_VOICE_PROVIDER_CODE,
    ensure_local_infra_prices,
    ensure_local_infra_providers,
    ensure_local_models,
)
from app.models.entity import SysAiModel
from app.models.entity.sys_ai_model_price import SysAiModelPrice, SysAiModelPriceDetail
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.repository.ai_model_repository import ai_model_repository
from app.repository.ai_provider_key_repository import ai_provider_key_repository
from app.repository.ai_provider_repository import ai_provider_repository

pytestmark = pytest.mark.requires_db


async def _local_provider_id(db) -> int:
    provider = await ai_provider_repository.get_by_provider_code(db, "local")
    assert provider is not None, "ensure_local_models 应播种 local provider"
    return provider.id


async def test_seeded_embedding_model_is_embedding_type(db):
    await ensure_local_models(db)
    provider_id = await _local_provider_id(db)
    model = await ai_model_repository.get_by_model_and_provider(
        db, LOCAL_EMBEDDING_MODEL_ID, provider_id
    )
    assert model is not None
    assert model.model_type == "embedding"
    assert model.dimension == 1024


async def test_seeded_chat_model_is_chat_type(db):
    await ensure_local_models(db)
    provider_id = await _local_provider_id(db)
    model = await ai_model_repository.get_by_model_and_provider(db, LOCAL_MODEL_ID, provider_id)
    assert model is not None
    assert model.model_type == "chat"


async def test_seeded_rerank_model_is_rerank_type(db):
    await ensure_local_models(db)
    provider_id = await _local_provider_id(db)
    model = await ai_model_repository.get_by_model_and_provider(
        db, LOCAL_RERANK_MODEL_ID, provider_id
    )
    assert model is not None
    assert model.model_type == "rerank"


async def test_seeding_is_idempotent(db):
    await ensure_local_models(db)
    await ensure_local_models(db)
    for model_id in (LOCAL_MODEL_ID, LOCAL_EMBEDDING_MODEL_ID, LOCAL_RERANK_MODEL_ID):
        count = (
            await db.execute(
                select(func.count()).select_from(SysAiModel).where(SysAiModel.model_id == model_id)
            )
        ).scalar()
        assert count == 1


async def test_price_backfilled_for_existing_model_without_price(db):
    """存量环境（模型已播种、价格缺失）重跑播种时补齐免费价格。

    旧版本播种不含价格逻辑，模型已存在的环境永远不会进"创建模型+价格"分支；
    价格补齐必须独立于模型创建，否则计费结算抛"未配置用户售价"。
    """
    await ensure_local_models(db)
    provider_id = await _local_provider_id(db)
    # 模拟存量环境：清掉价格版本与档位（模型保留）
    price_ids = (
        (
            await db.execute(
                select(SysAiModelPrice.id).where(
                    SysAiModelPrice.model_id == LOCAL_MODEL_ID,
                    SysAiModelPrice.provider_id == provider_id,
                )
            )
        )
        .scalars()
        .all()
    )
    await db.execute(
        delete(SysAiModelPriceDetail).where(SysAiModelPriceDetail.price_id.in_(price_ids))
    )
    await db.execute(delete(SysAiModelPrice).where(SysAiModelPrice.model_id == LOCAL_MODEL_ID))
    await db.flush()
    assert (
        await ai_model_price_repository.get_effective_version(
            db, LOCAL_MODEL_ID, provider_id, datetime.now()
        )
        is None
    )

    await ensure_local_models(db)
    price = await ai_model_price_repository.get_effective_version(
        db, LOCAL_MODEL_ID, provider_id, datetime.now()
    )
    assert price is not None
    details = await ai_model_price_repository.list_details(db, price.id)
    assert len(details) == 6  # input/cached/output × idle/peak 全 0 价
    assert all(d.unit_price == 0 for d in details)


async def test_key_change_deactivates_stale_rows(db, monkeypatch):
    """INTERNAL_SERVICE_KEY 变更后重播种：旧密钥行自动停用（自愈），不再被 Key 轮换选中。"""

    await ensure_local_models(db)
    provider_id = await _local_provider_id(db)
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_KEY", "rotated-internal-service-key")

    await ensure_local_models(db)

    keys = await ai_provider_key_repository.list_by_provider(db, provider_id)
    current_hash = hash_key("rotated-internal-service-key")
    assert any(k.key_hash == current_hash and k.status == 1 for k in keys), "应播种新密钥行"
    stale = [k for k in keys if k.key_hash != current_hash]
    assert stale, "前一次播种应留下旧密钥行"
    assert all(k.status == 0 for k in stale), "旧密钥行应全部停用"


async def test_seeds_voice_and_modal_providers(db):
    """阶段 1：local-voice / local-modal provider 以 openai_audio / openai_image 协议注册"""
    await ensure_local_infra_providers(db)

    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    assert voice is not None
    assert voice.api_base_url == f"{settings.VOICE_ENGINE_BASE_URL.rstrip('/')}/v1"
    assert voice.protocol_type == "openai_audio"

    modal = await ai_provider_repository.get_by_provider_code(db, LOCAL_MODAL_PROVIDER_CODE)
    assert modal is not None
    assert modal.api_base_url == f"{settings.MODAL_ENGINE_BASE_URL.rstrip('/')}/v1"
    assert modal.protocol_type == "openai_image"


async def test_service_key_registered_per_provider(db):
    """同一把 INTERNAL_SERVICE_KEY 在 local / local-voice / local-modal 各登记一行启用 Key"""
    await ensure_local_models(db)
    await ensure_local_infra_providers(db)

    key_hash = hash_key(settings.INTERNAL_SERVICE_KEY)
    for code in ("local", LOCAL_VOICE_PROVIDER_CODE, LOCAL_MODAL_PROVIDER_CODE):
        provider = await ai_provider_repository.get_by_provider_code(db, code)
        assert provider is not None
        keys = await ai_provider_key_repository.list_by_provider(db, provider.id)
        assert any(k.key_hash == key_hash and k.status == 1 for k in keys), f"{code} 应播种服务密钥"


async def test_infra_provider_seeding_idempotent(db):
    await ensure_local_infra_providers(db)
    await ensure_local_infra_providers(db)
    for code in (LOCAL_VOICE_PROVIDER_CODE, LOCAL_MODAL_PROVIDER_CODE):
        provider = await ai_provider_repository.get_by_provider_code(db, code)
        assert provider is not None
        keys = await ai_provider_key_repository.list_by_provider(db, provider.id)
        assert len(keys) == 1  # 幂等：不重复登记密钥行


async def _seed_registry_model(db, provider_id: int, model_id: str, model_type: str) -> SysAiModel:
    """登记模型行（价格播种按 provider+model_type 动态取行，模型来源非本用例关注点）。

    config/sql/data 静态种子已含 sensevoice/paraformer/huayan（provider 78），已存在时
    复用该行，避免撞 uk_model_provider 唯一键。
    """
    existing = await ai_model_repository.get_by_model_and_provider(db, model_id, provider_id)
    if existing is not None:
        return existing
    model = SysAiModel(
        provider_id=provider_id,
        model_id=model_id,
        model_type=model_type,
        display_name=f"M-{model_id}",
        max_context_tokens=4096,
        max_output_tokens=0,
        supports_multimodal=0,
        supports_tool_call=0,
        supports_streaming=0,
        supports_prompt_cache=0,
        supports_structured_output=0,
        prompt_cache_prefix_len=0,
        status=1,
        vip_level=0,
    )
    db.add(model)
    await db.flush()
    return model


async def _effective_price(db, model_id: str, provider_id: int):
    return await ai_model_price_repository.get_effective_version(
        db, model_id, provider_id, datetime.now()
    )


async def test_token_price_keeps_million_tokens_and_token_kind(db):
    await ensure_local_models(db)
    provider_id = await _local_provider_id(db)

    price = await _effective_price(db, LOCAL_MODEL_ID, provider_id)
    assert price is not None
    assert price.billing_unit == "million_tokens"
    details = await ai_model_price_repository.list_details(db, price.id)
    assert len(details) == 6  # input/cached/output × idle/peak
    assert all(d.usage_kind == "token" for d in details)
    assert all(d.unit_price == 0 for d in details)


async def test_asr_price_seeded_per_second(db):
    await ensure_local_infra_providers(db)
    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    assert voice is not None
    await _seed_registry_model(db, voice.id, "sensevoice", "asr")

    await ensure_local_infra_prices(db)

    price = await _effective_price(db, "sensevoice", voice.id)
    assert price is not None
    assert price.billing_unit == "second"
    details = await ai_model_price_repository.list_details(db, price.id)
    assert len(details) == 1  # 单行 unit 档，不设峰谷/分段
    assert details[0].usage_kind == "unit"
    assert details[0].token_type == "unit"
    assert details[0].max_tokens is None
    assert details[0].unit_price == Decimal("1")  # 1 积分/秒


async def test_tts_price_seeded_per_ten_thousand_chars(db):
    await ensure_local_infra_providers(db)
    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    assert voice is not None
    await _seed_registry_model(db, voice.id, "huayan", "tts")

    await ensure_local_infra_prices(db)

    price = await _effective_price(db, "huayan", voice.id)
    assert price is not None
    assert price.billing_unit == "ten_thousand_chars"
    details = await ai_model_price_repository.list_details(db, price.id)
    assert len(details) == 1
    assert details[0].usage_kind == "unit"
    assert details[0].unit_price == Decimal("100")  # 每字符 0.01 积分 → 万字符 100 积分


async def test_image_edit_price_seeded_per_image(db):
    await ensure_local_infra_providers(db)
    modal = await ai_provider_repository.get_by_provider_code(db, LOCAL_MODAL_PROVIDER_CODE)
    assert modal is not None
    await _seed_registry_model(db, modal.id, "aecrnet", "image_edit")

    await ensure_local_infra_prices(db)

    price = await _effective_price(db, "aecrnet", modal.id)
    assert price is not None
    assert price.billing_unit == "image"
    details = await ai_model_price_repository.list_details(db, price.id)
    assert len(details) == 1
    assert details[0].usage_kind == "unit"
    assert details[0].unit_price == Decimal("1")  # 本地低积分标定：1 积分/张


async def test_infra_price_seeding_idempotent(db):
    await ensure_local_infra_providers(db)
    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    assert voice is not None
    await _seed_registry_model(db, voice.id, "sensevoice", "asr")

    await ensure_local_infra_prices(db)
    await ensure_local_infra_prices(db)

    count = (
        await db.execute(
            select(func.count())
            .select_from(SysAiModelPrice)
            .where(
                SysAiModelPrice.model_id == "sensevoice",
                SysAiModelPrice.provider_id == voice.id,
            )
        )
    ).scalar()
    assert count == 1


async def test_voice_disabled_by_switch(db, monkeypatch):
    """VOICE_LOCAL_ENABLED=False（本部署由第三方语音引擎提供 ASR/TTS）：
    init-builtin 不初始化 local-voice，且已有供应商行被禁用；modal 不受影响。"""
    await ensure_local_infra_providers(db)
    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    assert voice is not None
    assert voice.status == 1

    monkeypatch.setattr(settings, "VOICE_LOCAL_ENABLED", False)
    await ensure_local_infra_providers(db)

    voice = await ai_provider_repository.get_by_provider_code(db, LOCAL_VOICE_PROVIDER_CODE)
    assert voice is not None
    assert voice.status == 0, "开关关闭时应禁用内置语音供应商行"
    modal = await ai_provider_repository.get_by_provider_code(db, LOCAL_MODAL_PROVIDER_CODE)
    assert modal is not None
    assert modal.status == 1, "modal 不受语音开关影响"


async def test_provider_base_url_not_overwritten(db):
    """配置权威源是 DB：已存在的供应商行地址被管理员修改后，重跑 init-builtin 不覆盖"""
    await ensure_local_models(db)
    provider = await ai_provider_repository.get_by_provider_code(db, "local")
    assert provider is not None
    provider.api_base_url = "http://127.0.0.1:18992/v1"
    await db.flush()

    await ensure_local_models(db)

    provider = await ai_provider_repository.get_by_provider_code(db, "local")
    assert provider is not None
    assert provider.api_base_url == "http://127.0.0.1:18992/v1", "初始化不得覆盖用户改过的地址"
