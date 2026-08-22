from types import SimpleNamespace

import pytest

from app.core.exceptions import BusinessException
from app.service import ai_agent_service as m
from app.service.ai_agent_service import DEFAULT_AGENT_CODE, agent_service


def _agent(code="dehaze_helper", agent_id=1):
    return SimpleNamespace(id=agent_id, agent_code=code, deleted=0)


@pytest.fixture
def env(monkeypatch):
    redis = object()
    calls = {"soft_delete": []}

    async def get_by_id(d, aid):
        return _agent(agent_id=aid)

    async def get_by_code(d, code):
        return None

    async def count_conv(d, code):
        return 0

    async def count_sub(d, aid):
        return 0

    async def soft_delete(d, ids):
        calls["soft_delete"].extend(ids)

    monkeypatch.setattr(m.ai_agent_repository, "get_by_id", get_by_id)
    monkeypatch.setattr(m.ai_agent_repository, "get_by_code", get_by_code)
    monkeypatch.setattr(m.ai_agent_repository, "count_conversation_references", count_conv)
    monkeypatch.setattr(m.ai_agent_repository, "count_subagent_references", count_sub)
    monkeypatch.setattr(m.ai_agent_repository, "soft_delete_by_ids", soft_delete)

    class _Cache:
        async def delete(self, key):
            return None

    monkeypatch.setattr(m.CacheService, "delete", _Cache.delete)

    return redis, calls


class TestDeleteAgent:
    async def test_default_agent_not_deletable(self, env, monkeypatch):
        redis, calls = env

        async def get_by_id(d, aid):
            return _agent(code=DEFAULT_AGENT_CODE)

        monkeypatch.setattr(m.ai_agent_repository, "get_by_id", get_by_id)
        with pytest.raises(BusinessException) as exc:
            await agent_service.delete_agent(object(), redis, 1)
        assert "默认 Agent" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_deleted_when_referenced_by_conversation(self, env, monkeypatch):
        redis, calls = env

        async def count_conv(d, code):
            return 3

        monkeypatch.setattr(m.ai_agent_repository, "count_conversation_references", count_conv)
        with pytest.raises(BusinessException) as exc:
            await agent_service.delete_agent(object(), redis, 1)
        assert "会话" in str(exc.value) and "3" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_deleted_when_used_as_subagent(self, env, monkeypatch):
        redis, calls = env

        async def count_sub(d, aid):
            return 2

        monkeypatch.setattr(m.ai_agent_repository, "count_subagent_references", count_sub)
        with pytest.raises(BusinessException) as exc:
            await agent_service.delete_agent(object(), redis, 1)
        assert "子 Agent" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_delete_without_references_soft_deletes(self, env):
        redis, calls = env
        await agent_service.delete_agent(object(), redis, 1)
        assert calls["soft_delete"] == [1]

    async def test_delete_nonexistent_raises(self, env, monkeypatch):
        redis, calls = env

        async def get_by_id(d, aid):
            return None

        monkeypatch.setattr(m.ai_agent_repository, "get_by_id", get_by_id)
        with pytest.raises(BusinessException):
            await agent_service.delete_agent(object(), redis, 99)
        assert calls["soft_delete"] == []
