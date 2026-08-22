from types import SimpleNamespace

import pytest

import app.repository.ai_model_repository as repo_m
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service import ai_model_service as m


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
):
    return SimpleNamespace(
        id=pk,
        model_id=model_id,
        provider_id=provider_id,
        supports_multimodal=multimodal,
        supports_tool_call=tool_call,
        supports_streaming=streaming,
        fallback_model_pk=fallback_pk,
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

    async def test_create_model_blocks_reuse_after_soft_delete(self, monkeypatch):
        async def get_by_model_and_provider(db, model_id, provider_id):
            return _model(pk=1, model_id="gpt-4o", provider_id=1, deleted=1)

        async def create(db, model):
            return model

        async def clear(redis):
            return None

        monkeypatch.setattr(
            repo_m.ai_model_repository, "get_by_model_and_provider", get_by_model_and_provider
        )
        monkeypatch.setattr(repo_m.ai_model_repository, "create", create)
        monkeypatch.setattr(m, "_clear_model_cache", clear)

        form = SimpleNamespace(
            provider_id=1,
            model_id="gpt-4o",
            display_name="GPT",
            input_rate=1.0,
            output_rate=1.0,
            cached_rate=1.0,
            max_context_tokens=4096,
            max_output_tokens=4096,
            supports_multimodal=False,
            supports_tool_call=False,
            supports_streaming=True,
            supports_prompt_cache=False,
            supports_structured_output=False,
            fallback_model_pk=None,
            prompt_cache_prefix_len=0,
            status=1,
            vip_level=0,
        )
        with pytest.raises(BusinessException) as exc:
            await m.AiModelService.create_model(None, object(), form)
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

        routes = await m.AiModelService.get_call_routes(None, "gpt-4o", set())
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

        routes = await m.AiModelService.get_call_routes(None, "gpt-4o", {"tool_call"})
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

        routes = await m.AiModelService.get_call_routes(None, "a", set())
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

        routes = await m.AiModelService.get_call_routes(None, "m1", set())
        assert len(routes) == 6


class TestValidateModelCaps:
    async def test_multimodal_required_raises_a0601(self):
        model = _model(multimodal=0)
        with pytest.raises(BusinessException) as exc:
            await m.AiModelService.validate_model_caps(
                model, has_attachments=True, need_tools=False
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE

    async def test_tool_call_required_raises_a0601(self):
        model = _model(tool_call=0)
        with pytest.raises(BusinessException) as exc:
            await m.AiModelService.validate_model_caps(
                model, has_attachments=False, need_tools=True
            )
        assert exc.value.code == ResultCode.AI_MODEL_NOT_AVAILABLE

    async def test_passes_when_caps_satisfied(self):
        model = _model(multimodal=1, tool_call=1)
        result = await m.AiModelService.validate_model_caps(
            model, has_attachments=True, need_tools=True
        )
        assert result is None


class TestListEnabledModels:
    async def test_is_fallback_target_flag(self, monkeypatch):
        m1 = _model(pk=1, model_id="gpt-4o", provider_id=1, fallback_pk=3)
        m2 = _model(pk=2, model_id="claude", provider_id=1, fallback_pk=None)
        m3 = _model(pk=3, model_id="gpt-4o-mini", provider_id=1, fallback_pk=None)

        async def list_enabled(db):
            return [m1, m2, m3]

        async def get_user_level(db, redis, uid):
            return 0

        async def get_json(self, key):
            return None

        async def set_json(self, key, value, ttl):
            return None

        async def health_snapshot(redis, provider_id):
            return {}

        monkeypatch.setattr(repo_m.ai_model_repository, "list_enabled", list_enabled)
        monkeypatch.setattr(m, "_get_user_level", get_user_level)
        monkeypatch.setattr(m, "_provider_health_snapshot", health_snapshot)
        monkeypatch.setattr(m.CacheService, "get_json", get_json)
        monkeypatch.setattr(m.CacheService, "set_json", set_json)

        items = await m.AiModelService.list_enabled_models(None, object(), 1)
        flags = {item.model_id: item.is_fallback_target for item in items}
        assert flags["gpt-4o-mini"] is True
        assert flags["gpt-4o"] is False
        assert flags["claude"] is False
