from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import run_after_commit_callbacks
from app.repository.ai_agent_repository import AiAgentRepository
from app.repository.ai_agent_version_repository import AiAgentVersionRepository
from app.service import ai_agent_version_service as m
from app.service.ai_agent_version_service import AgentVersionService
from tests.stubs.fakes import StubAsyncSession


def _agent(**kw):
    base = {
        "id": 1,
        "name": "去雾助手",
        "description": "",
        "system_prompt": "sys",
        "model_id": "m1",
        "reasoning_mode": "fast",
        "config": {"max_steps": 30},
        "permissions": [],
        "is_subagent": False,
        "is_team": False,
        "is_exposed": True,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _version(**kw):
    base = {
        "id": 1,
        "agent_id": 1,
        "version_no": 1,
        "snapshot": {},
        "status": 1,
        "change_note": "",
        "operator_id": 1,
    }
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
        "audit": [],
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
        list_versions=lambda d, aid, offset, limit: ([], 0),
    )
    _stub_repo(list_names_existing=lambda d, names: list(names))

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

    async def fake_judge(db):
        return {"consistency_state": "normal", "drift_paused": False}

    monkeypatch.setattr(m.eval_center_service, "judge_status", staticmethod(fake_judge))

    monkeypatch.setattr(
        m,
        "mongo_audit_log_repository",
        SimpleNamespace(create_audit_async=lambda **kw: calls["audit"].append(kw)),
    )

    # 回归评测在独立 session 中执行；单测下复用桩 session 观察提交时序
    @asynccontextmanager
    async def fake_session():
        yield db

    monkeypatch.setattr(m, "get_db_session", fake_session)

    # 测试替身：仅实现服务实际调用的仓储方法（多方法结构型桩，改真子类会触发方法覆写告警）
    svc = AgentVersionService(
        ai_agent_repository=cast(AiAgentRepository, agent_repo),  # 替身：多方法结构型仓储桩
        ai_agent_version_repository=cast(AiAgentVersionRepository, version_repo),  # 替身：同上
    )
    return db, redis, calls, svc


class TestSnapshotContract:
    async def test_snapshot_contains_config_and_resolved_config(self, env):
        db, redis, _, svc = env
        snap = await svc._build_snapshot(db, redis, _agent())
        assert "config" in snap
        assert "resolved_config" in snap
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
        await run_after_commit_callbacks(db)
        assert calls["delete_cache"]
        # 发布链路先固化草稿（v3）再写已发布版本（v4）
        assert vno == 4

    async def test_publish_fixates_draft_as_gate_target(self, env, monkeypatch):
        """发布前将可编辑态固化为草稿，门禁评测对象（最新草稿）与即将发布配置一致。"""
        db, redis, _calls, svc = env
        gate_agent_ids = []

        async def gate(db, redis, agent_id, trigger_type="publish"):
            gate_agent_ids.append(agent_id)
            # 草稿在门禁执行前已写入
            drafts = [e for e in db.entities if getattr(e, "status", None) == 1]
            assert drafts
            assert drafts[-1].change_note == "发布评测草稿"
            return {"passed": True, "failed_samples": [], "run_id": None}

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(gate))
        await svc.publish(db, redis, 1, 1, "release")
        assert gate_agent_ids == [1]
        statuses = [e.status for e in db.entities]
        assert 1 in statuses
        assert statuses[-1] == 2

    async def test_publish_gate_failed_raises(self, env, monkeypatch):
        db, redis, _, svc = env

        async def fail_gate(db, redis, agent_id, trigger_type="publish"):
            return {"passed": False, "failed_samples": [{"sample_id": 1}], "run_id": None}

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(fail_gate))
        with pytest.raises(BusinessException) as exc:
            await svc.publish(db, redis, 1, 1, "x")
        assert "门禁" in str(exc.value) or "发布门禁" in str(exc.value)

    async def test_publish_degraded_message_readably(self, env, monkeypatch):
        """退化阻断（degraded=True 且 failed_samples 为空）时不得输出误导性的"失败样本：[]"。"""
        db, redis, calls, svc = env

        async def degraded_gate(db, redis, agent_id, trigger_type="publish"):
            return {"passed": False, "degraded": True, "failed_samples": [], "run_id": 9}

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(degraded_gate))
        with pytest.raises(BusinessException) as exc:
            await svc.publish(db, redis, 1, 1, "x")
        assert "退化" in str(exc.value)
        assert "失败样本：[]" not in str(exc.value)
        assert calls["demote"] == 0

    async def test_publish_insufficient_eval_message_readably(self, env, monkeypatch):
        """回归集已配置但无考题：门禁阻断且不输出误导性的"失败样本：[]"。"""
        db, redis, calls, svc = env

        async def insufficient_gate(db, redis, agent_id, trigger_type="publish"):
            return {
                "passed": False,
                "insufficient_eval": True,
                "degraded": False,
                "failed_samples": [],
                "run_id": None,
            }

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(insufficient_gate))
        with pytest.raises(BusinessException) as exc:
            await svc.publish(db, redis, 1, 1, "x")
        assert exc.value.code == ResultCode.DATA_STATE_NOT_ALLOW
        assert "回归集无考题" in str(exc.value)
        assert "失败样本：[]" not in str(exc.value)
        assert calls["demote"] == 0

    async def _drift_judge(self, monkeypatch):
        async def drifted(db):
            return {"consistency_state": "drifted", "drift_paused": True}

        monkeypatch.setattr(m.eval_center_service, "judge_status", staticmethod(drifted))

    async def test_publish_drifted_blocks(self, env, monkeypatch):
        db, redis, calls, svc = env
        await self._drift_judge(monkeypatch)
        with pytest.raises(BusinessException) as exc:
            await svc.publish(db, redis, 1, 1, "x")
        assert "漂移" in str(exc.value)
        assert calls["demote"] == 0

    async def test_publish_drifted_force_exempts_and_records_note(self, env, monkeypatch):
        db, redis, _calls, svc = env
        await self._drift_judge(monkeypatch)
        await svc.publish(db, redis, 1, 1, "紧急修复", force=True)
        version = db.entities[-1]
        assert version.status == 2
        assert version.change_note == "[漂移豁免]紧急修复"

    async def test_publish_drifted_force_still_blocked_by_regression(self, env, monkeypatch):
        db, redis, calls, svc = env
        await self._drift_judge(monkeypatch)

        async def fail_gate(db, redis, agent_id, trigger_type="publish"):
            return {"passed": False, "failed_samples": [{"sample_id": 1}], "run_id": None}

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(fail_gate))
        with pytest.raises(BusinessException) as exc:
            await svc.publish(db, redis, 1, 1, "x", force=True)
        assert "门禁" in str(exc.value)
        assert calls["demote"] == 0

    async def test_publish_insufficient_data_allows(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def insufficient(db):
            return {"consistency_state": "insufficient_data", "drift_paused": False}

        monkeypatch.setattr(m.eval_center_service, "judge_status", staticmethod(insufficient))
        vno = await svc.publish(db, redis, 1, 1, "x")
        assert vno == 4
        assert calls["demote"] == 1

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
            return _version(snapshot=snapshot, version_no=vno, status=2)

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
        await run_after_commit_callbacks(db)
        assert calls["delete_cache"]
        assert vno == 3

    async def test_rollback_restores_config_and_prompt(self, env, monkeypatch):
        db, redis, _calls, svc = env
        snapshot = {
            "name": "旧名",
            "config": {"max_steps": 10},
            "system_prompt": "old",
            "skills": [],
            "mcp_namespaces": [],
            "subagents": [],
        }

        async def get_version(d, aid, vno):
            return _version(snapshot=snapshot, version_no=vno, status=2)

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


class TestPublishTransactionBoundary:
    """发布链路的事务边界：草稿提交后才跑评测，缓存失效登记到提交后。"""

    async def test_publish_commits_draft_before_regression(self, env, monkeypatch):
        """分钟级回归在独立 session 中执行，且执行前草稿已提交（否则评测读不到草稿）。"""
        db, redis, _, svc = env
        committed_when_gate_ran = []

        async def gate(gate_db, redis, agent_id, trigger_type="publish"):
            committed_when_gate_ran.append(gate_db.committed)
            return {"passed": True, "failed_samples": [], "run_id": None}

        monkeypatch.setattr(m.eval_service, "run_regression", staticmethod(gate))
        await svc.publish(db, redis, 1, 1, "release")
        assert committed_when_gate_ran == [1]

    async def test_publish_defers_published_cache_clear(self, env):
        db, redis, calls, svc = env
        await svc.publish(db, redis, 1, 1, "release")
        assert calls["delete_cache"] == []
        await run_after_commit_callbacks(db)
        assert calls["delete_cache"] == ["ai:agent:1:published"]

    async def test_rollback_defers_published_cache_clear(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def get_version(d, aid, vno):
            return _version(snapshot={}, version_no=vno, status=2)

        svc.ai_agent_version_repository = _stub_repo(
            get_by_agent_and_version=get_version,
            demote_published=svc.ai_agent_version_repository.demote_published,
            next_version_no=svc.ai_agent_version_repository.next_version_no,
        )
        await svc.rollback(db, redis, 1, 2, 1)
        assert calls["delete_cache"] == []
        await run_after_commit_callbacks(db)
        assert calls["delete_cache"] == ["ai:agent:1:published"]


class TestRollbackTarget:
    """回滚目标限定为已发布版本：草稿是发布链路的评测中间态，配置未经门禁验证。"""

    def _version_repo(self, svc, status):
        async def get_version(d, aid, vno):
            return _version(snapshot={}, version_no=vno, status=status)

        return _stub_repo(
            get_by_agent_and_version=get_version,
            demote_published=svc.ai_agent_version_repository.demote_published,
            next_version_no=svc.ai_agent_version_repository.next_version_no,
        )

    async def test_rollback_to_draft_rejected(self, env):
        db, redis, calls, svc = env
        svc.ai_agent_version_repository = self._version_repo(svc, status=1)
        with pytest.raises(BusinessException) as exc:
            await svc.rollback(db, redis, 1, 2, 1)
        assert "已发布" in str(exc.value)
        assert calls["demote"] == 0

    async def test_rollback_to_published_allowed(self, env):
        db, redis, calls, svc = env
        svc.ai_agent_version_repository = self._version_repo(svc, status=2)
        vno = await svc.rollback(db, redis, 1, 2, 1)
        assert calls["demote"] == 1
        assert vno == 3

    async def test_rollback_writes_audit_after_commit(self, env):
        db, redis, calls, svc = env
        svc.ai_agent_version_repository = self._version_repo(svc, status=2)
        await svc.rollback(db, redis, 1, 2, 9)
        assert calls["audit"] == []
        await run_after_commit_callbacks(db)

        assert len(calls["audit"]) == 1
        audit = calls["audit"][0]
        assert audit["operator_id"] == 9
        assert audit["action"] == "rollback"
        assert audit["target_type"] == "ai_agent"
        assert audit["target_id"] == 1
        assert audit["after_value"]["from_version_no"] == 2


class TestForcePublishAudit:
    async def test_force_publish_writes_audit(self, env, monkeypatch):
        db, redis, calls, svc = env
        await svc.publish(db, redis, 1, 3, "紧急修复", force=True)
        await run_after_commit_callbacks(db)

        audits = [a for a in calls["audit"] if a["action"] == "publish_force"]
        assert len(audits) == 1
        assert audits[0]["operator_id"] == 3
        assert audits[0]["after_value"]["drift_exempted"] is False

    async def test_normal_publish_writes_no_audit(self, env):
        db, redis, calls, svc = env
        await svc.publish(db, redis, 1, 3, "常规发布")
        await run_after_commit_callbacks(db)
        assert calls["audit"] == []

    async def test_drifted_force_publish_marks_exemption(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def drifted(db):
            return {"consistency_state": "drifted", "drift_paused": True}

        monkeypatch.setattr(m.eval_center_service, "judge_status", staticmethod(drifted))
        await svc.publish(db, redis, 1, 3, "紧急修复", force=True)
        await run_after_commit_callbacks(db)

        audit = next(a for a in calls["audit"] if a["action"] == "publish_force")
        assert audit["after_value"]["drift_exempted"] is True
        assert audit["after_value"]["change_note"] == "[漂移豁免]紧急修复"


class TestListVersionsPagination:
    async def test_list_versions_pushes_paging_to_sql(self, env):
        """分页下推 SQL：仓储收到 offset/limit，服务层不做内存切片。"""
        db, redis, _, svc = env
        seen = []
        rows = [_version(version_no=3), _version(version_no=2), _version(version_no=1)]

        async def list_versions(d, aid, offset, limit):
            seen.append((offset, limit))
            return rows[:limit], 5

        svc.ai_agent_version_repository = _stub_repo(list_versions=list_versions)
        result = await svc.list_versions(db, redis, 1, page=2, size=2)
        assert seen == [(2, 2)]
        assert result.total == 5
        assert [v.version_no for v in result.list] == [3, 2]
