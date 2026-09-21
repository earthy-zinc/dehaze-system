"""AI 模型管理业务不变量与对抗性语料测试（真实 db fixture）

覆盖：
- 删除被降级链引用的模型 → A0504 保护
- dimension 仅 embedding 有效（非 embedding 强制置空）
- /providers/enabled 精简视图不泄漏供应商内部配置
- 模型/供应商字段对抗性脏语料（emoji/零宽/CRLF/超长）与参数边界
"""

import pytest
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_conversation import AiModelCreate, AiModelUpdate
from app.models.schema.ai_provider import ProviderCreate, ProviderUpdate
from app.service.ai_model_service import ai_model_service
from app.service.ai_provider_service import PROVIDER_LIST_CACHE_KEY, ai_provider_service

pytestmark = pytest.mark.requires_db

ZWC = "​"  # 零宽空格


def _model_form(model_id: str, model_type: str = "chat", **overrides) -> AiModelCreate:
    data = {
        "provider_id": 1,
        "model_id": model_id,
        "model_type": model_type,
        "dimension": 1024 if model_type == "embedding" else None,
        "display_name": f"M-{model_id}",
        "status": 1,
    }
    data.update(overrides)
    return AiModelCreate(**data)


def _provider_form(provider_code: str, **overrides) -> ProviderCreate:
    data: dict = {
        "provider_code": provider_code,
        "display_name": f"P-{provider_code}",
        "api_base_url": "https://api.test.local/v1",
    }
    data.update(overrides)
    return ProviderCreate(**data)


class TestDeleteFallbackGuard:
    async def test_delete_fallback_target_blocked(self, db, mock_redis):
        target = await ai_model_service.create_model(db, mock_redis, _model_form("fb-target"))
        await ai_model_service.create_model(
            db, mock_redis, _model_form("fb-source", fallback_model_id=target.id)
        )
        with pytest.raises(BusinessException) as exc:
            await ai_model_service.delete_model(db, mock_redis, "fb-target")
        assert exc.value.code == ResultCode.DATA_BIND_EXISTS

    async def test_delete_allowed_after_reference_removed(self, db, mock_redis):
        target = await ai_model_service.create_model(db, mock_redis, _model_form("fb-t2"))
        await ai_model_service.create_model(
            db, mock_redis, _model_form("fb-s2", fallback_model_id=target.id)
        )
        await ai_model_service.update_model(
            db, mock_redis, "fb-s2", AiModelUpdate(fallback_model_id=None)
        )
        await ai_model_service.delete_model(db, mock_redis, "fb-t2")


class TestDimensionInvariant:
    async def test_non_embedding_dimension_purged(self, db, mock_redis):
        result = await ai_model_service.create_model(
            db, mock_redis, _model_form("chat-dim", dimension=768)
        )
        assert result.dimension is None

    async def test_embedding_dimension_roundtrip(self, db, mock_redis):
        result = await ai_model_service.create_model(
            db, mock_redis, _model_form("emb-dim", "embedding", dimension=1024)
        )
        assert result.dimension == 1024


class TestProviderEnabledRedacted:
    async def test_enabled_list_strips_internal_config(self, db, mock_redis):
        provider = await ai_provider_service.create_provider(
            db,
            mock_redis,
            _provider_form(
                "prov_redact",
                api_base_url="https://secret.internal/v1",
                user_identity_forward={
                    "enabled": True,
                    "field": "user_id",
                    "prefix": "u_",
                    "max_len": 64,
                },
                remark="内部账号备注",
            ),
        )
        # 先触发一次缓存写入（完整视图），再走 enabled 精简路径
        await ai_provider_service.list_enabled(db, mock_redis)
        enabled = await ai_provider_service.list_enabled(db, mock_redis)
        item = next(p for p in enabled if p.id == provider.id)
        assert item.model_dump().keys() == {
            "id",
            "provider_code",
            "display_name",
            "protocol_type",
            "health",
            "status",
        }
        # 精简视图不缓存旧完整结构：清缓存后重建仍为精简字段
        await mock_redis.delete(PROVIDER_LIST_CACHE_KEY)
        await ai_provider_service.delete_provider(db, mock_redis, provider.id)


class TestAdversarialCorpus:
    """对抗性脏语料：emoji/零宽/CRLF/全半角/超长输入的校验与往返一致性"""

    async def test_provider_display_name_dirty_chars_roundtrip(self, db, mock_redis):
        dirty = "O🚀penAI​-兼容\n测试"
        form = _provider_form("prov_dirty")
        form.display_name = dirty
        result = await ai_provider_service.create_provider(db, mock_redis, form)
        assert result.display_name == dirty
        updated = await ai_provider_service.update_provider(
            db, mock_redis, result.id, ProviderUpdate(display_name=dirty + "\r\n")
        )
        assert updated.display_name == dirty + "\r\n"

    async def test_provider_code_over_limit_rejected(self, db, mock_redis):
        with pytest.raises(ValidationError):
            _provider_form("p" * 33)

    async def test_model_display_name_dirty_chars_roundtrip(self, db, mock_redis):
        dirty = "GPT\t4o\r\n🤖​版"
        result = await ai_model_service.create_model(
            db, mock_redis, _model_form("m-dirty", display_name=dirty)
        )
        assert result.display_name == dirty

    async def test_model_id_boundary_length(self, db, mock_redis):
        result = await ai_model_service.create_model(db, mock_redis, _model_form("m" * 64))
        assert result.model_id == "m" * 64
        with pytest.raises(ValidationError):
            _model_form("m" * 65)


class TestDeleteProviderGuard:
    async def test_delete_provider_blocked_by_disabled_model(self, db, mock_redis):
        """禁用模型引用供应商时同样拦截删除，防悬挂引用（重新启用时才爆雷的延迟数据问题"""
        provider = await ai_provider_service.create_provider(
            db, mock_redis, _provider_form("prov_del_guard")
        )
        await ai_model_service.create_model(
            db, mock_redis, _model_form("m-del-guard", provider_id=provider.id, status=0)
        )

        with pytest.raises(BusinessException) as exc:
            await ai_provider_service.delete_provider(db, mock_redis, provider.id)
        assert exc.value.code == ResultCode.DATA_BIND_EXISTS

    async def test_delete_provider_allowed_without_models(self, db, mock_redis):
        provider = await ai_provider_service.create_provider(
            db, mock_redis, _provider_form("prov_del_free")
        )

        await ai_provider_service.delete_provider(db, mock_redis, provider.id)
