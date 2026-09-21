"""本地模型播种测试：embedding/rerank 模型必须显式 model_type。

schema 默认 model_type=chat，播种漏传会把本地向量/重排模型登记成对话类型，
污染模型类型筛选与目录展示，此处锁定播种语义与幂等性。
"""

from datetime import datetime

import pytest
from sqlalchemy import delete, func, select

from app.infrastructure.llm.local.model_seeder import (
    LOCAL_EMBEDDING_MODEL_ID,
    LOCAL_MODEL_ID,
    LOCAL_RERANK_MODEL_ID,
    ensure_local_models,
)
from app.models.entity import SysAiModel
from app.models.entity.sys_ai_model_price import SysAiModelPrice, SysAiModelPriceDetail
from app.repository.ai_model_price_repository import ai_model_price_repository
from app.repository.ai_model_repository import ai_model_repository
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
