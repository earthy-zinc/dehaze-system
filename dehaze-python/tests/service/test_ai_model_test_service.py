"""模型可用性测试服务单测（model_test_service）。

覆盖：探测请求按协议/模型类型的组装分支、成功/HTTP错误/超时三类结果落库口径、
模型/供应商/Key 缺失的业务异常、近 24h 实调统计聚合（真实测试库验证 SQL 可执行）。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import app.service.ai.service.model_test_service as m
from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_model import SysAiModel
from tests.stubs.factories import fake_redis

pytestmark = pytest.mark.requires_db


def _model(pk=1, model_id="gpt-4o", model_type="chat", provider_id=1, status=1) -> SysAiModel:
    return SysAiModel(
        id=pk,
        model_id=model_id,
        model_type=model_type,
        provider_id=provider_id,
        status=status,
        last_test_status=0,
        last_test_at=None,
        last_test_error=None,
    )


def _provider(pid=1, code="openai", protocol="openai_compat", status=1):
    return SimpleNamespace(
        id=pid,
        provider_code=code,
        protocol_type=protocol,
        api_base_url="https://api.example.com/v1",
        auth_type="bearer",
        status=status,
        default_headers={},
    )


class _FakeDb:
    def __init__(self):
        self.flushed = False

    async def flush(self):
        self.flushed = True


class TestProbeRequest:
    def test_openai_chat(self):
        url, payload = m._probe_request("openai_compat", "chat", "https://a.com/v1", "gpt-4o")
        assert url == "https://a.com/v1/chat/completions"
        assert payload["max_tokens"] == 1
        assert payload["stream"] is False

    def test_anthropic_chat(self):
        url, payload = m._probe_request("anthropic", "chat", "https://a.com", "claude-3")
        assert url == "https://a.com/v1/messages"
        assert payload["max_tokens"] == 1

    def test_embedding_default(self):
        url, payload = m._probe_request("openai_compat", "embedding", "https://a.com/v1", "e1")
        assert url == "https://a.com/v1/embeddings"
        assert payload["input"] == "hi"

    def test_embedding_cohere(self):
        url, payload = m._probe_request(
            "openai_compat", "embedding", "https://a.com/v1/embed", "e1"
        )
        assert url == "https://a.com/v1/embed"
        assert payload["input_type"] == "search_query"

    def test_rerank(self):
        url, payload = m._probe_request("openai_compat", "rerank", "https://a.com", "r1")
        assert url == "https://a.com/rerank"
        assert payload["top_n"] == 1
        assert len(payload["documents"]) == 1


class TestTestModel:
    async def _setup(self, monkeypatch, model=None, provider=None, key: str | None = "sk-test"):
        model = model or _model()
        provider = provider or _provider()
        repo = SimpleNamespace(get_by_id=AsyncMock(return_value=model))
        provider_repo = SimpleNamespace(get_by_id=AsyncMock(return_value=provider))
        monkeypatch.setattr(m, "ai_model_repository", repo)
        monkeypatch.setattr(m, "ai_provider_repository", provider_repo)
        selector = SimpleNamespace(select_key=AsyncMock(return_value=key))
        monkeypatch.setattr(m, "provider_key_selector", selector)
        monkeypatch.setattr(m, "_clear_model_cache", AsyncMock())
        return model

    async def test_success_persists_available(self, monkeypatch):
        model = await self._setup(monkeypatch)

        class _Resp:
            status_code = 200

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                return _Resp()

        monkeypatch.setattr(m.httpx, "AsyncClient", lambda **kw: _Client())
        db, redis = _FakeDb(), await fake_redis()

        result = await m.test_model(db, redis, 1)

        assert result["success"] is True
        assert result["latencyMs"] >= 0
        assert model.last_test_status == 1
        assert model.last_test_at is not None
        assert model.last_test_error is None
        assert db.flushed

    async def test_http_error_persists_unavailable_with_detail(self, monkeypatch):
        model = await self._setup(monkeypatch)

        class _Resp:
            status_code = 404

            def json(self):
                return {"error": {"message": "model not found"}}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                return _Resp()

        monkeypatch.setattr(m.httpx, "AsyncClient", lambda **kw: _Client())
        result = await m.test_model(_FakeDb(), await fake_redis(), 1)

        assert result["success"] is False
        assert model.last_test_status == 2
        assert model.last_test_error is not None
        assert "HTTP 404" in model.last_test_error
        assert "model not found" in model.last_test_error

    async def test_timeout_persists_unavailable(self, monkeypatch):
        model = await self._setup(monkeypatch)

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                raise httpx.TimeoutException("t")

        monkeypatch.setattr(m.httpx, "AsyncClient", lambda **kw: _Client())
        db = _FakeDb()
        result = await m.test_model(db, await fake_redis(), 1)

        assert result["success"] is False
        assert model.last_test_status == 2
        assert model.last_test_error is not None
        assert "超时" in model.last_test_error

    async def test_model_not_found_raises(self, monkeypatch):
        repo = SimpleNamespace(get_by_id=AsyncMock(return_value=None))
        monkeypatch.setattr(m, "ai_model_repository", repo)
        with pytest.raises(BusinessException) as exc:
            await m.test_model(_FakeDb(), await fake_redis(), 99)
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_provider_disabled_raises(self, monkeypatch):
        await self._setup(monkeypatch, provider=_provider(status=0))
        with pytest.raises(BusinessException) as exc:
            await m.test_model(_FakeDb(), await fake_redis(), 1)
        assert exc.value.code == ResultCode.RESOURCE_NOT_FOUND

    async def test_no_api_key_raises(self, monkeypatch):
        await self._setup(monkeypatch, key=None)
        with pytest.raises(BusinessException) as exc:
            await m.test_model(_FakeDb(), await fake_redis(), 1)
        assert exc.value.code == ResultCode.OPERATION_NOT_ALLOW


class TestUsageStats24h:
    async def test_aggregates_chat_and_kb_models(self, db):
        """chat 按llm_call聚合成功率；embedding/rerank按计费流水计数；未调用模型不出现在结果"""
        from sqlalchemy import text

        chat_model = _model(pk=910001, model_id="stat-chat-m", model_type="chat")
        embed_model = _model(pk=910002, model_id="stat-embed-m", model_type="embedding")
        cold_model = _model(pk=910003, model_id="stat-cold-m", model_type="chat")

        await db.execute(
            text(
                "INSERT INTO sys_ai_llm_call (trace_id, seq, model, status, duration_ms, "
                "prompt_tokens, completion_tokens, cached_tokens) VALUES "
                "('t-stat-1', 1, 'stat-chat-m', 1, 100, 10, 5, 0),"
                "('t-stat-1', 2, 'stat-chat-m', 1, 100, 10, 5, 0),"
                "('t-stat-1', 3, 'stat-chat-m', 2, 100, 10, 0, 0)"
            )
        )
        await db.execute(
            text(
                "INSERT INTO sys_ai_billing (user_id, model, bill_type, input_tokens, "
                "credits, quota_consumed, pre_deduct) VALUES "
                "(1, 'stat-embed-m', 'embedding', 100, 1, 1, 0),"
                "(1, 'stat-embed-m', 'embedding', 100, 1, 1, 0)"
            )
        )
        await db.flush()

        stats = await m.get_usage_stats_24h(db, [chat_model, embed_model, cold_model])

        chat_stat = stats[910001]
        assert chat_stat["calls_24h"] == 3
        assert chat_stat["success_rate_24h"] == 67  # 2/3 四舍五入
        assert chat_stat["last_call_at"] is not None

        embed_stat = stats[910002]
        assert embed_stat["calls_24h"] == 2
        assert embed_stat["success_rate_24h"] == 100

        assert 910003 not in stats
