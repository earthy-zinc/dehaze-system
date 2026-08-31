"""本地模型播种测试：embedding 模型必须显式 model_type=embedding。

schema 默认 model_type=chat，播种漏传会把本地向量模型登记成对话类型，
污染模型类型筛选与目录展示，此处锁定播种语义与幂等性。
"""

import pytest
from sqlalchemy import func, select

from app.infrastructure.llm.local.model_seeder import (
    LOCAL_EMBEDDING_MODEL_ID,
    LOCAL_MODEL_ID,
    ensure_local_models,
)
from app.models.entity import SysAiModel
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
    model = await ai_model_repository.get_by_model_and_provider(
        db, LOCAL_MODEL_ID, provider_id
    )
    assert model is not None
    assert model.model_type == "chat"


async def test_seeding_is_idempotent(db):
    await ensure_local_models(db)
    await ensure_local_models(db)
    for model_id in (LOCAL_MODEL_ID, LOCAL_EMBEDDING_MODEL_ID):
        count = (
            await db.execute(
                select(func.count())
                .select_from(SysAiModel)
                .where(SysAiModel.model_id == model_id)
            )
        ).scalar()
        assert count == 1
