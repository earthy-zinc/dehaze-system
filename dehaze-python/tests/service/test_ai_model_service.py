from types import SimpleNamespace
from unittest.mock import patch

import pytest

import app.repository.ai_model_repository as repo_m
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.provider.model_registry import model_registry
from app.models.schema.ai_conversation import AiModelCreate, AiModelUpdate
from app.service import ai_model_service as m
from app.service.ai_model_service import AiModelService

pytestmark = pytest.mark.requires_db

def _model(
    pk=1,
    model_id="gpt-4o",
    provider_id=1,
    multimodal=0,
    tool_call=0,
    streaming=1,
    fallback_pk=None,
    status=1,
    deleted=0,
    display_name="M",
    model_type="chat",
    dimension=None,
):
    return SimpleNamespace(
        id=pk,
        model_id=model_id,
        provider_id=provider_id,
        model_type=model_type,
        dimension=dimension,
        supports_multimodal=multimodal,
        supports_tool_call=tool_call,
        supports_streaming=streaming,
        fallback_model_id=fallback_pk,
        status=status,
        deleted=deleted,
        display_name=display_name,
        max_output_tokens=4096,
        max_context_tokens=8192,
        input_rate=1.0,
        output_rate=1.0,
        cached_rate=1.0,
        supports_prompt_cache=0,
        supports_structured_output=0,
        prompt_cache_prefix_len=0,
        vip_level=0,
    )


class TestDedupIncludeDeleted:
    async def test_get_by_model_and_provider_sets_include_deleted(self):
        captured = {}

        class _FakeDb:
            async def execute(self, stmt):
                captured["options"] = dict(stmt._execution_options)

                class _Result:
                    def scalar_one_or_none(self):
                        return None

                return _Result()

        await repo_m.ai_model_repository.get_by_model_and_provider(_FakeDb(), "gpt-4o", 1)
        assert captured["options"].get("include_deleted") is True

    async def test_create_model_blocks_reuse_after_soft_delete(self):
        class _FakeRepo:
            async def get_by_model_and_provider(self, db, model_id, provider_id):
                return _model(pk=1, model_id="gpt-4o", provider_id=1, deleted=1)

            async def create(self, db, model):
                return model

        svc = AiModelService(ai_model_repository=_FakeRepo())
        form = AiModelCreate(
            provider_id=1,
            model_id="gpt-4o",
            model_type="chat",
            dimension=None,
            display_name="GPT",
            status=1,
            vip_level=0,
        )
        with pytest.raises(BusinessException) as exc:
            await svc.create_model(None, object(), form)
        assert "已被历史记录占用" in str(exc.value)


class TestGetCallRoutes:
    async def test_order_current_providers_then_fallback(self, monkeypatch):
        cur_a = _model(pk=1, model_id="gpt-4o", provider_id=1, fallback_pk=3)
        cur_b = _model(pk=2, model_id="gpt-4o", provider_id=2)
        fb = _model(pk=3, model_id="gpt-4o-mini", provider_id=1, fallback_pk=None)

        def _patch(current, fallbacks):
            async def list_by_model_id(db, model_id):
                return current

            async def list_by_pks(db, pks):
                return fallbacks

            monkeypatch.setattr(
                repo_m.ai_model_repository, "list_enabled_by_model_id", list_by_model_id
            )
            monkeypatch.setattr(repo_m.ai_model_repository, "list_enabled_by_pks", list_by_pks)

        _patch([cur_a, cur_b], [fb])

        routes = await model_registry.get_call_routes(None, "gpt-4o", set())
        assert [r["model_pk"] for r in routes] == [1, 2, 3]

    async def test_capability_filter_skips_unsupported(self, monkeypatch):
        fb_tool = _model(pk=3, model_id="gpt-4o-mini", provider_id=1, tool_call=1)
        fb_no_tool = _model(pk=4, model_id="gpt-4o-nano", provider_id=1, tool_call=0)

        def _patch(current, fallbacks):
            async def list_by_model_id(db, model_id):
                return current

            async def list_by_pks(db, pks):
                return fallbacks

            monkeypatch.setattr(
                repo_m.ai_model_repository, "list_enabled_by_model_id", list_by_model_id
            )
            monkeypatch.setattr(repo_m.ai_model_repository, "list_enabled_by_pks", list_by_pks)

        _patch([_model(pk=1, model_id="gpt-4o", provider_id=1, fallback_pk=3)], [fb_tool, fb_no_tool])

        routes = await model_registry.get_call_routes(None, "gpt-4o", {"tool_call"})
        assert [r["model_pk"] for r in routes] == [1, 3]

    async def test_cycle_guard(self, monkeypatch):
        a = _model(pk=1, model_id="a", provider_id=1, fallback_pk=2)
        b = _model(pk=2, model_id="b", provider_id=1, fallback_pk=1)

        def _patch(current, fallbacks):
            async def list_by_model_id(db, model_id):
                return current

            async def list_by_pks(db, pks):
                return [b] if pks == [2] else [a]

            monkeypatch.setattr(
                repo_m.ai_model_repository, "list_enabled_by_model_id", list_by_model_id
            )
            monkeypatch.setattr(repo_m.ai_model_repository, "list_enabled_by_pks", list_by_pks)

        _patch([a], [b])

        routes = await model_registry.get_call_routes(None, "a", set())
        assert [r["model_pk"] for r in routes] == [1, 2]

    async def test_depth_limit_caps_chain(self, monkeypatch):
        chain = {
            pk: _model(pk=pk, model_id=f"m{pk}", provider_id=1, fallback_pk=pk + 1)
            for pk in range(1, 10)
        }

        def _patch(current, fallbacks):
            async def list_by_model_id(db, model_id):
                return current

            async def list_by_pks(db, pks):
                return [chain[p] for p in pks]

            monkeypatch.setattr(
                repo_m.ai_model_repository, "list_enabled_by_model_id", list_by_model_id
            )
            monkeypatch.setattr(repo_m.ai_model_repository, "list_enabled_by_pks", list_by_pks)

        _patch([chain[1]], [])

        routes = await model_registry.get_call_routes(None, "m1", set())
        assert len(routes) == 6


class TestValidateModelCaps:
    async def test_multimodal_required_raises_a0601(self):
        model = _model(multimodal=0)
        with pytest.raises(BusinessException) as exc:
            await m.ai_model_service.validate_model_caps(
                model, has_attachments=True, need_tools=False
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE

    async def test_tool_call_required_raises_a0601(self):
        model = _model(tool_call=0)
        with pytest.raises(BusinessException) as exc:
            await m.ai_model_service.validate_model_caps(
                model, has_attachments=False, need_tools=True
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE

    async def test_passes_when_caps_satisfied(self):
        model = _model(multimodal=1, tool_call=1)
        result = await m.ai_model_service.validate_model_caps(
            model, has_attachments=True, need_tools=True
        )
        assert result is None


class TestListEnabledModels:
    async def test_is_fallback_target_flag(self, monkeypatch):
        m1 = _model(pk=1, model_id="gpt-4o", provider_id=1, fallback_pk=3)
        m2 = _model(pk=2, model_id="claude", provider_id=1, fallback_pk=None)
        m3 = _model(pk=3, model_id="gpt-4o-mini", provider_id=1, fallback_pk=None)

        class _FakeRepo:
            async def list_enabled(self, db):
                return [m1, m2, m3]

        async def get_user_level(db, redis, uid):
            return 0

        async def health_snapshot(redis, provider_id):
            return {}

        async def get_json(self, key):
            return None

        async def set_json(self, key, value, ttl):
            return None

        monkeypatch.setattr(m.CacheService, "get_json", get_json)
        monkeypatch.setattr(m.CacheService, "set_json", set_json)
        svc = AiModelService(ai_model_repository=_FakeRepo())
        monkeypatch.setattr(m, "_get_user_level", get_user_level)
        monkeypatch.setattr(m, "_provider_health_snapshot", health_snapshot)

        items = await svc.list_enabled_models(None, object(), 1)
        flags = {item.model_id: item.is_fallback_target for item in items}
        assert flags["gpt-4o-mini"] is True
        assert flags["gpt-4o"] is False
        assert flags["claude"] is False


def _create_form(model_id: str, model_type: str = "chat", dimension: int | None = None, vip_level: int = 0) -> AiModelCreate:
    return AiModelCreate(
        provider_id=999,
        model_id=model_id,
        model_type=model_type,
        dimension=dimension,
        display_name=f"M-{model_id}",
        input_rate=1.0,
        output_rate=3.0,
        cached_rate=0.5,
        max_context_tokens=8192,
        max_output_tokens=4096,
        supports_multimodal=False,
        supports_tool_call=False,
        supports_streaming=True,
        supports_prompt_cache=False,
        supports_structured_output=False,
        prompt_cache_prefix_len=0,
        status=1,
        vip_level=vip_level,
    )


class TestModelTypeDimension:
    async def test_create_embedding_returns_dimension(self, db, mock_redis):
        result = await m.ai_model_service.create_model(db, mock_redis, _create_form("emb-model", "embedding", 1024))
        assert result.model_type == "embedding"
        assert result.dimension == 1024

    async def test_embedding_requires_dimension(self, db, mock_redis):
        with pytest.raises(BusinessException) as exc:
            await m.ai_model_service.create_model(db, mock_redis, _create_form("emb-no-dim", "embedding"))
        assert exc.value.code == ResultCode.PARAM_ERROR

    async def test_list_models_filters_by_model_type(self, db, mock_redis):
        await m.ai_model_service.create_model(db, mock_redis, _create_form("chat-a"))
        await m.ai_model_service.create_model(db, mock_redis, _create_form("emb-a", "embedding", 1024))
        page = await m.ai_model_service.list_models(db, 1, 10, model_type="chat")
        assert all(item.model_type == "chat" for item in page.list)
        assert any(item.model_id == "chat-a" for item in page.list)
        assert not any(item.model_id == "emb-a" for item in page.list)

    async def test_dimension_immutable_on_update(self, db, mock_redis):
        await m.ai_model_service.create_model(db, mock_redis, _create_form("emb-fix", "embedding", 1024))
        with pytest.raises(BusinessException) as exc:
            await m.ai_model_service.update_model(db, mock_redis, "emb-fix", AiModelUpdate(dimension=2048))
        assert exc.value.code == ResultCode.DATA_STATE_NOT_ALLOW

    async def test_model_type_immutable_on_update(self, db, mock_redis):
        await m.ai_model_service.create_model(db, mock_redis, _create_form("chat-fix"))
        with pytest.raises(BusinessException) as exc:
            await m.ai_model_service.update_model(db, mock_redis, "chat-fix", AiModelUpdate(model_type="rerank"))
        assert exc.value.code == ResultCode.DATA_STATE_NOT_ALLOW

    async def test_embedding_dimension_le_zero_rejected(self, db, mock_redis):
        with pytest.raises(Exception) as exc:
            _create_form("emb-bad", "embedding", dimension=0)
        assert "gt=0" in str(exc.value) or "dimension" in str(exc.value).lower()


class TestDeleteModel:
    async def test_delete_model_active_session_blocked(self, db, mock_redis):
        await m.ai_model_service.create_model(db, mock_redis, _create_form("del-active"))
        m2 = None
        for _ in range(1):
            m2 = await m.ai_model_service.create_model(db, mock_redis, _create_form("del-other"))

        class _FakeRepo:
            async def get_by_model_id(self, db, model_id):
                return _model(pk=1, model_id=model_id, provider_id=1, status=1)

            async def count_active_conversations(self, db, model_id):
                return 3 if model_id == "del-active" else 0

            async def soft_delete_by_ids(self, db, ids):
                return None

        svc = AiModelService(ai_model_repository=_FakeRepo())
        with pytest.raises(BusinessException) as exc:
            await svc.delete_model(db, mock_redis, "del-active")
        assert exc.value.code == ResultCode.DATA_BIND_EXISTS

    async def test_delete_model_no_active_session(self, db, mock_redis):
        captured = {}

        class _FakeRepo:
            async def get_by_model_id(self, db, model_id):
                return _model(pk=1, model_id=model_id, provider_id=1, status=1)

            async def count_active_conversations(self, db, model_id):
                return 0

            async def soft_delete_by_ids(self, db, ids):
                captured["ids"] = ids
                return None

        svc = AiModelService(ai_model_repository=_FakeRepo())
        await svc.delete_model(db, mock_redis, "del-ok")
        assert captured.get("ids") == [1]

    async def test_delete_model_not_found(self, db, mock_redis):
        class _FakeRepo:
            async def get_by_model_id(self, db, model_id):
                return None

        svc = AiModelService(ai_model_repository=_FakeRepo())
        with pytest.raises(BusinessException) as exc:
            await svc.delete_model(db, mock_redis, "nope")
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND


class TestUpdateModel:
    async def test_update_model_not_found(self, db, mock_redis):
        with pytest.raises(BusinessException) as exc:
            await m.ai_model_service.update_model(
                db, mock_redis, "missing-model", AiModelUpdate(display_name="x")
            )
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_disable_triggers_replacement_notice_with_fallback(self, db, mock_redis, monkeypatch):
        class _FakeRepo:
            async def get_by_model_id(self, db, model_id):
                return _model(pk=1, model_id=model_id, provider_id=1, status=1, fallback_pk=9)

            async def list_enabled_by_pks(self, db, pks):
                return [_model(pk=9, model_id="fallback-m", provider_id=1, status=1, display_name="fallback-m")]

            async def list_active_conversation_users(self, db, model_id):
                return [1001, 1002]

        notified = {}

        class _FakeMessage:
            async def send(self, db, payload):
                notified["payload"] = payload
                return None

        svc = AiModelService(
            ai_model_repository=_FakeRepo(),
            message_service=_FakeMessage(),
        )
        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(db, "flush", _noop)
        monkeypatch.setattr(db, "refresh", _noop)
        async def _clear_noop(redis):
            return None
        with patch.object(m, "_clear_model_cache", _clear_noop):
            result = await svc.update_model(
                db, mock_redis, "chat-a", AiModelUpdate(status=0)
            )
        assert result.status == 0
        assert "即将不可用" in notified["payload"]["title"]
        assert "fallback-m" in notified["payload"]["content"]

    async def test_disable_triggers_replacement_notice_no_fallback(self, db, mock_redis, monkeypatch):
        class _FakeRepo:
            async def get_by_model_id(self, db, model_id):
                return _model(pk=1, model_id=model_id, provider_id=1, status=1)

            async def list_enabled_by_pks(self, db, pks):
                return []

            async def list_active_conversation_users(self, db, model_id):
                return [1001]

        notified = {}

        class _FakeMessage:
            async def send(self, db, payload):
                notified["payload"] = payload
                return None

        svc = AiModelService(
            ai_model_repository=_FakeRepo(),
            message_service=_FakeMessage(),
        )
        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(db, "flush", _noop)
        monkeypatch.setattr(db, "refresh", _noop)
        async def _clear_noop(redis):
            return None
        with patch.object(m, "_clear_model_cache", _clear_noop):
            await svc.update_model(
                db, mock_redis, "chat-b", AiModelUpdate(status=0)
            )
        assert "暂未配置替代模型" in notified["payload"]["content"]


class TestModelType:
    async def test_rerank_type_create_and_list(self, db, mock_redis):
        result = await m.ai_model_service.create_model(db, mock_redis, _create_form("rerank-a", "rerank"))
        assert result.model_type == "rerank"
        page = await m.ai_model_service.list_models(db, 1, 10, model_type="rerank")
        assert any(item.model_id == "rerank-a" for item in page.list)

    async def test_model_type_three_values_query(self, db, mock_redis):
        await m.ai_model_service.create_model(db, mock_redis, _create_form("c1", "chat"))
        await m.ai_model_service.create_model(db, mock_redis, _create_form("e1", "embedding", 1024))
        await m.ai_model_service.create_model(db, mock_redis, _create_form("r1", "rerank"))
        for mt in ("chat", "embedding", "rerank"):
            page = await m.ai_model_service.list_models(db, 1, 10, model_type=mt)
            assert any(item.model_type == mt for item in page.list)


class TestVipLevelFilter:
    async def test_list_enabled_filters_by_vip_level(self, db, mock_redis, monkeypatch):
        await m.ai_model_service.create_model(db, mock_redis, _create_form("vip0", vip_level=0))
        await m.ai_model_service.create_model(db, mock_redis, _create_form("vip1", vip_level=1))
        await m.ai_model_service.create_model(db, mock_redis, _create_form("vip2", vip_level=2))

        async def get_json(self, key):
            return None

        async def set_json(self, key, value, ttl):
            return None

        monkeypatch.setattr(m.CacheService, "get_json", get_json)
        monkeypatch.setattr(m.CacheService, "set_json", set_json)

        for level, expected in ((0, {"vip0"}), (1, {"vip0", "vip1"}), (2, {"vip0", "vip1", "vip2"})):
            async def _get_level(db, redis, uid, _lvl=level):
                return _lvl

            monkeypatch.setattr(m, "_get_user_level", _get_level)
            items = await m.ai_model_service.list_enabled_models(db, mock_redis, 1)
            model_ids = {item.model_id for item in items}
            # 种子数据含 vip_level=0 的模型，仅断言自建模型按等级过滤，不假设列表全集
            assert expected.issubset(model_ids)
            assert model_ids.isdisjoint({"vip1", "vip2"} - expected)
