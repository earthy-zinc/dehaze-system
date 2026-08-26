from types import SimpleNamespace

import pytest

from app.core.exceptions import BusinessException
from app.service.ai_agent_version_service import AgentVersionService
from app.service import ai_agent_version_service as m
from tests.stubs.fakes import StubAsyncSession


def _agent(**kw):
    base = dict(
        id=1,
        name="去雾助手",
        description="",
        system_prompt="sys",
        model_id="m1",
        reasoning_mode="fast",
        config={"max_steps": 30},
        permissions=[],
        is_subagent=False,
        is_team=False,
        is_exposed=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _version(**kw):
    base = dict(
        id=1, agent_id=1, version_no=1, snapshot={}, status=1, change_note="", operator_id=1
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _stub_repo(**methods):
    return SimpleNamespace(**methods)


@pytest.fixture
def env(monkeypatch):
    db = StubAsyncSession()
    redis = object()
    calls = {
        "demote": 0,
        "next_no": 0,
        "delete_cache": [],
        "replace_skills": [],
        "replace_mcp": [],
        "replace_sub": [],
    }

    async def get_agent(d, aid):
        return _agent()

    async def list_skills(d, aid):
        return ["skill_a"]

    async def list_mcp(d, aid):
        return ["ns_a"]

    async def list_sub(d, aid):
        return []

    async def next_no(d, aid):
        calls["next_no"] += 1
        return calls["next_no"] + 2

    async def demote(d, aid):
        calls["demote"] += 1

    async def replace_skills(d, aid, skills):
        calls["replace_skills"].append(skills)

    async def replace_mcp(d, aid, ns):
        calls["replace_mcp"].append(ns)

    async def replace_sub(d, aid, subs):
        calls["replace_sub"].append(subs)

    async def get_version(d, aid, vno):
        return None

    agent_repo = _stub_repo(
        get_by_id=get_agent,
        list_skill_names=list_skills,
        list_mcp_namespaces=list_mcp,
        list_subagents=list_sub,
        replace_skills=replace_skills,
        replace_mcp_namespaces=replace_mcp,
        replace_subagents=replace_sub,
    )
    version_repo = _stub_repo(
        next_version_no=next_no,
        demote_published=demote,
        get_by_agent_and_version=get_version,
        list_versions=lambda d, aid: [],
    )
    skill_repo = _stub_repo(list_names_existing=lambda d, names: list(names))

    class _Cache:
        async def delete(self, key):
            calls["delete_cache"].append(key)

    monkeypatch.setattr(m.CacheService, "delete", _Cache.delete)

    async def fake_resolve(_db, _redis, config, conv=None):
        return {"reasoning": {**(config or {}), "extra": 1}}

    class _Resolver:
        resolve = staticmethod(fake_resolve)

    monkeypatch.setattr(m, "agent_config_resolver", _Resolver)

    async def fake_gate(db, redis, agent_id, trigger_type="publish"):
        return {"passed": True, "failed_samples": [], "run_id": None}

    monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(fake_gate))

    svc = AgentVersionService(
        ai_agent_repository=agent_repo,
        ai_agent_version_repository=version_repo,
    )
    return db, redis, calls, svc


class TestSnapshotContract:
    async def test_snapshot_contains_config_and_resolved_config(self, env):
        db, redis, _, svc = env
        snap = await svc._build_snapshot(db, redis, _agent())
        assert "config" in snap and "resolved_config" in snap
        assert snap["config"] == {"max_steps": 30}
        assert snap["resolved_config"]["reasoning"]["max_steps"] == 30
        assert snap["skills"] == ["skill_a"]
        assert snap["mcp_namespaces"] == ["ns_a"]
        assert snap["subagents"] == []


class TestVersionFlow:
    async def test_save_draft_writes_status_1(self, env):
        db, redis, _, svc = env
        result = await svc.save_draft(db, redis, 1, 1, "draft note")
        assert result.status == 1

    async def test_publish_creates_published_and_demotes_old(self, env):
        db, redis, calls, svc = env
        vno = await svc.publish(db, redis, 1, 1, "release")
        assert calls["demote"] == 1
        assert calls["delete_cache"]
        assert vno == 3

    async def test_publish_gate_failed_raises(self, env, monkeypatch):
        db, redis, _, svc = env

        async def fail_gate(db, redis, agent_id, trigger_type="publish"):
            return {"passed": False, "failed_samples": [{"sample_id": 1}], "run_id": None}

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(fail_gate))
        with pytest.raises(BusinessException) as exc:
            await svc.publish(db, redis, 1, 1, "x")
        assert "门禁" in str(exc.value) or "发布门禁" in str(exc.value)

    async def test_rollback_restores_and_creates_new_published(self, env, monkeypatch):
        db, redis, calls, svc = env
        snapshot = {
            "name": "旧名",
            "config": {"max_steps": 10},
            "system_prompt": "old",
            "skills": ["s1"],
            "mcp_namespaces": ["n1"],
            "subagents": [],
        }

        async def get_version(d, aid, vno):
            return _version(snapshot=snapshot, version_no=vno)

        # 注入版本仓库覆盖 get_by_agent_and_version
        svc.ai_agent_version_repository = _stub_repo(
            get_by_agent_and_version=get_version,
            demote_published=svc.ai_agent_version_repository.demote_published,
            next_version_no=svc.ai_agent_version_repository.next_version_no,
        )

        vno = await svc.rollback(db, redis, 1, 2, 1)
        assert calls["replace_skills"] == [["s1"]]
        assert calls["replace_mcp"] == [["n1"]]
        assert calls["demote"] >= 1
        assert calls["delete_cache"]
        assert vno == 3

    async def test_rollback_restores_config_and_prompt(self, env, monkeypatch):
        db, redis, calls, svc = env
        snapshot = {
            "name": "旧名",
            "config": {"max_steps": 10},
            "system_prompt": "old",
            "skills": [],
            "mcp_namespaces": [],
            "subagents": [],
        }

        async def get_version(d, aid, vno):
            return _version(snapshot=snapshot, version_no=vno)

        svc.ai_agent_version_repository = _stub_repo(
            get_by_agent_and_version=get_version,
            demote_published=svc.ai_agent_version_repository.demote_published,
            next_version_no=svc.ai_agent_version_repository.next_version_no,
        )

        captured = {}

        async def get_agent(d, aid):
            a = _agent()
            captured["name"] = a
            return a

        svc.ai_agent_repository.get_by_id = get_agent

        await svc.rollback(db, redis, 1, 2, 1)
        restored = captured["name"]
        assert restored.config == {"max_steps": 10}
        assert restored.system_prompt == "old"

    async def test_publish_only_one_published(self, env):
        db, redis, calls, svc = env
        await svc.publish(db, redis, 1, 1, "v1")
        await svc.publish(db, redis, 1, 1, "v2")
        assert calls["demote"] == 2
