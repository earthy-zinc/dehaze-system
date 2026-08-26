from types import SimpleNamespace

import pytest

from app.core.exceptions import BusinessException
from app.service.ai_agent_service import AgentService, DEFAULT_AGENT_CODE
from app.service import ai_agent_service as m


def _agent(code="dehaze_helper", agent_id=1):
    return SimpleNamespace(id=agent_id, agent_code=code, deleted=0)


def _stub_repo(**methods):
    return SimpleNamespace(**methods)


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

    agent_repo = _stub_repo(
        get_by_id=get_by_id,
        get_by_code=get_by_code,
        count_conversation_references=count_conv,
        count_subagent_references=count_sub,
        soft_delete_by_ids=soft_delete,
        paginate_agents=lambda d, p, s, kw=None, st=None: ([], 0),
        list_enabled=lambda d: [],
        list_skill_names=lambda d, aid: [],
        list_mcp_namespaces=lambda d, aid: [],
        replace_skills=lambda d, aid, names: None,
        replace_mcp_namespaces=lambda d, aid, ns: None,
        replace_subagents=lambda d, aid, items: None,
        create=lambda d, a: None,
    )
    version_repo = _stub_repo(
        get_by_agent_and_version=lambda d, aid, vno: None,
        get_latest_published=lambda d, aid: None,
        next_version_no=lambda d, aid: 1,
        demote_published=lambda d, aid: None,
        list_versions=lambda d, aid: [],
    )
    skill_repo = _stub_repo(list_names_existing=lambda d, names: list(names))

    class _Cache:
        async def delete(self, key):
            return None

    monkeypatch.setattr(m.CacheService, "delete", _Cache.delete)

    # delete_agent 不触达 agent_version_service
    svc = AgentService(
        ai_agent_repository=agent_repo,
        ai_agent_version_repository=version_repo,
        ai_skill_repository=skill_repo,
        agent_version_service=None,
    )
    return redis, calls, svc


class TestDeleteAgent:
    async def test_default_agent_not_deletable(self, env, monkeypatch):
        redis, calls, svc = env

        async def get_by_id(d, aid):
            return _agent(code=DEFAULT_AGENT_CODE)

        svc.ai_agent_repository.get_by_id = get_by_id
        with pytest.raises(BusinessException) as exc:
            await svc.delete_agent(object(), redis, 1)
        assert "默认 Agent" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_deleted_when_referenced_by_conversation(self, env, monkeypatch):
        redis, calls, svc = env

        async def count_conv(d, code):
            return 3

        svc.ai_agent_repository.count_conversation_references = count_conv
        with pytest.raises(BusinessException) as exc:
            await svc.delete_agent(object(), redis, 1)
        assert "会话" in str(exc.value) and "3" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_deleted_when_used_as_subagent(self, env, monkeypatch):
        redis, calls, svc = env

        async def count_sub(d, aid):
            return 2

        svc.ai_agent_repository.count_subagent_references = count_sub
        with pytest.raises(BusinessException) as exc:
            await svc.delete_agent(object(), redis, 1)
        assert "子 Agent" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_delete_without_references_soft_deletes(self, env):
        redis, calls, svc = env
        await svc.delete_agent(object(), redis, 1)
        assert calls["soft_delete"] == [1]

    async def test_delete_nonexistent_raises(self, env, monkeypatch):
        redis, calls, svc = env

        async def get_by_id(d, aid):
            return None

        svc.ai_agent_repository.get_by_id = get_by_id
        with pytest.raises(BusinessException):
            await svc.delete_agent(object(), redis, 99)
        assert calls["soft_delete"] == []
