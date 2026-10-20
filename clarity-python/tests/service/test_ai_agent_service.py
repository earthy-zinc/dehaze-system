from types import SimpleNamespace
from typing import cast

import pytest

from app.core.exceptions import BusinessException
from app.database import run_after_commit_callbacks
from app.models.base import set_current_user_id
from app.models.schema.ai_agent import AgentSubAgentItem, AgentSubAgentsForm
from app.repository.ai_agent_repository import AiAgentRepository
from app.repository.ai_agent_version_repository import AiAgentVersionRepository
from app.repository.ai_skill_repository import AiSkillRepository
from app.service import ai_agent_service as m
from app.service.ai_agent_service import DEFAULT_AGENT_CODE, AgentService
from app.service.ai_agent_version_service import AgentVersionService
from tests.stubs.fakes import StubAsyncSession


def _agent(code="dehaze_helper", agent_id=1):
    return SimpleNamespace(id=agent_id, agent_code=code, name="去雾助手", deleted=0)


def _stub_repo(**methods):
    return SimpleNamespace(**methods)


@pytest.fixture
def env(monkeypatch):
    db = StubAsyncSession()
    redis = object()
    calls = {
        "soft_delete": [],
        "delete_cache": [],
        "audit": [],
        "eval_soft_delete": [],
        "eval_sample_delete": [],
        "eval_run_delete": [],
    }

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

    async def noop(*args):
        return None

    async def empty_list(*args, **kwargs):
        return []

    async def empty_page(*args, **kwargs):
        return [], 0

    async def get_by_ids(d, ids):
        return [_agent(agent_id=i) for i in ids]

    agent_repo = _stub_repo(
        get_by_id=get_by_id,
        get_by_code=get_by_code,
        count_conversation_references=count_conv,
        count_subagent_references=count_sub,
        soft_delete_by_ids=soft_delete,
        paginate_agents=empty_page,
        list_enabled=empty_list,
        list_skill_names=empty_list,
        list_mcp_namespaces=empty_list,
        replace_skills=noop,
        replace_mcp_namespaces=noop,
        replace_subagents=noop,
        create=noop,
        get_by_ids=get_by_ids,
        list_subagents=empty_list,
    )
    version_repo = _stub_repo(
        get_by_agent_and_version=noop,
        get_latest_published=noop,
        next_version_no=lambda d, aid: 1,
        demote_published=noop,
        list_versions=lambda d, aid, offset, limit: ([], 0),
    )

    async def list_names_existing(d, names):
        return list(names)

    skill_repo = _stub_repo(list_names_existing=list_names_existing)

    async def _delete(self, key):
        calls["delete_cache"].append(key)

    monkeypatch.setattr(m.CacheService, "delete", _delete)

    async def list_registered_names(d, namespaces):
        return list(namespaces)

    monkeypatch.setattr(
        m,
        "ai_mcp_namespace_repository",
        SimpleNamespace(list_registered_names=list_registered_names),
    )

    async def eval_soft_delete(d, ids):
        calls["eval_soft_delete"].append(ids)
        return len(ids)

    async def eval_sample_delete(d, ids):
        calls["eval_sample_delete"].append(ids)
        return len(ids)

    async def eval_run_delete(d, aid):
        calls["eval_run_delete"].append(aid)
        return 0

    monkeypatch.setattr(
        m,
        "ai_agent_eval_dataset_repository",
        SimpleNamespace(list_by_agent=empty_list, soft_delete_by_ids=eval_soft_delete),
    )
    monkeypatch.setattr(
        m,
        "ai_agent_eval_sample_repository",
        SimpleNamespace(delete_by_datasets=eval_sample_delete),
    )
    monkeypatch.setattr(
        m,
        "ai_agent_eval_run_repository",
        SimpleNamespace(delete_by_agent=eval_run_delete),
    )
    monkeypatch.setattr(
        m,
        "mongo_audit_log_repository",
        SimpleNamespace(create_audit_async=lambda **kw: calls["audit"].append(kw)),
    )

    # 测试替身：仅实现服务实际调用的仓储方法（多方法结构型桩 + 用例会重赋 get_by_id，
    # 改真子类会触发方法覆写/属性赋值告警）；
    # delete_agent 不触达 agent_version_service，用真实默认实例满足契约（不改变被测分支）
    svc = AgentService(
        ai_agent_repository=cast(AiAgentRepository, agent_repo),  # 替身：多方法结构型仓储桩
        ai_agent_version_repository=cast(AiAgentVersionRepository, version_repo),  # 替身：同上
        ai_skill_repository=cast(AiSkillRepository, skill_repo),  # 替身：仅实现 list_names_existing
        agent_version_service=AgentVersionService(),
    )
    return db, redis, calls, svc


class TestDeleteAgent:
    async def test_default_agent_not_deletable(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def get_by_id(d, aid):
            return _agent(code=DEFAULT_AGENT_CODE)

        svc.ai_agent_repository.get_by_id = get_by_id
        with pytest.raises(BusinessException) as exc:
            await svc.delete_agent(db, redis, 1)
        assert "默认 Agent" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_deleted_when_referenced_by_conversation(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def count_conv(d, code):
            return 3

        svc.ai_agent_repository.count_conversation_references = count_conv
        with pytest.raises(BusinessException) as exc:
            await svc.delete_agent(db, redis, 1)
        assert "会话" in str(exc.value)
        assert "3" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_deleted_when_used_as_subagent(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def count_sub(d, aid):
            return 2

        svc.ai_agent_repository.count_subagent_references = count_sub
        with pytest.raises(BusinessException) as exc:
            await svc.delete_agent(db, redis, 1)
        assert "子 Agent" in str(exc.value)
        assert calls["soft_delete"] == []

    async def test_delete_without_references_soft_deletes(self, env):
        db, redis, calls, svc = env
        await svc.delete_agent(db, redis, 1)
        assert calls["soft_delete"] == [1]

    async def test_delete_nonexistent_raises(self, env, monkeypatch):
        db, redis, calls, svc = env

        async def get_by_id(d, aid):
            return None

        svc.ai_agent_repository.get_by_id = get_by_id
        with pytest.raises(BusinessException):
            await svc.delete_agent(db, redis, 99)
        assert calls["soft_delete"] == []


class TestCacheInvalidationAfterCommit:
    """缓存失效必须登记在事务上、提交后才执行。

    提交前失效的竞态：并发请求 miss → 读到尚未提交的旧数据 → 回填旧快照，
    脏缓存存活至 TTL。
    """

    async def test_set_status_defers_cache_clear(self, env):
        db, redis, calls, svc = env
        await svc.set_status(db, redis, 1, 0)
        assert calls["delete_cache"] == []
        await run_after_commit_callbacks(db)
        assert "ai:agent:list:enabled" in calls["delete_cache"]

    async def test_delete_agent_defers_cache_clear(self, env):
        db, redis, calls, svc = env
        await svc.delete_agent(db, redis, 1)
        assert calls["delete_cache"] == []
        await run_after_commit_callbacks(db)
        assert "ai:agent:dehaze_helper" in calls["delete_cache"]
        assert "ai:agent:1:published" in calls["delete_cache"]

    async def test_rollback_discards_pending_cache_clear(self, env):
        """事务回滚（不执行提交后回调）→ 缓存不失效，避免提交前的空窗期"""
        db, redis, calls, svc = env
        await svc.set_status(db, redis, 1, 1)
        db.info.pop("after_commit_callbacks", None)
        await run_after_commit_callbacks(db)
        assert calls["delete_cache"] == []

    async def test_set_skills_clears_skill_cache_after_commit(self, env):
        db, redis, calls, svc = env
        await svc.set_skills(db, redis, 1, ["s1"])
        assert calls["delete_cache"] == []
        await run_after_commit_callbacks(db)
        assert "ai:agent:1:skills" in calls["delete_cache"]


def _subagents_form(*agent_ids):
    return AgentSubAgentsForm(subagents=[AgentSubAgentItem(agent_id=i) for i in agent_ids])


class TestDeleteAgentAudit:
    """删除 Agent 属危险操作：必须落审计，且审计随事务提交才发出。"""

    async def test_commit_callbacks_run_without_error(self, env, caplog):
        """defer_after_commit 会 await 回调：审计回调必须是协程，否则每次提交都抛
        TypeError 被 except 吞掉（审计看似写入、实则依赖回调先执行完的副作用）。"""
        import logging

        db, redis, calls, svc = env
        set_current_user_id(7)
        try:
            await svc.delete_agent(db, redis, 1)
            with caplog.at_level(logging.WARNING, logger="app.database"):
                await run_after_commit_callbacks(db)
        finally:
            set_current_user_id(None)

        assert "after_commit 回调执行失败" not in caplog.text
        assert calls["audit"]

    async def test_delete_writes_one_audit_after_commit(self, env):
        db, redis, calls, svc = env
        set_current_user_id(7)
        await svc.delete_agent(db, redis, 1)
        assert calls["audit"] == []
        await run_after_commit_callbacks(db)

        assert len(calls["audit"]) == 1
        audit = calls["audit"][0]
        assert audit["operator_id"] == 7
        assert audit["target_type"] == "ai_agent"
        assert audit["target_id"] == 1
        assert audit["action"] == "delete"
        assert audit["before_value"]["agent_code"] == "dehaze_helper"
        set_current_user_id(None)

    async def test_delete_cascades_eval_assets(self, env, monkeypatch):
        """评测资产随 Agent 删除一并清理：Agent 软删后失去清理入口，孤儿评测集/样本/
        执行记录会残留在评测中心聚合与评测集列表中。"""
        from app.service import ai_agent_service as m

        db, redis, calls, svc = env

        async def list_by_agent(d, agent_id):
            return [SimpleNamespace(id=11), SimpleNamespace(id=12)]

        async def delete_by_datasets(d, ids):
            calls["eval_sample_delete"].append(ids)
            return 7

        async def delete_by_agent(d, agent_id):
            calls["eval_run_delete"].append(agent_id)
            return 3

        monkeypatch.setattr(m.ai_agent_eval_dataset_repository, "list_by_agent", list_by_agent)
        monkeypatch.setattr(
            m.ai_agent_eval_sample_repository, "delete_by_datasets", delete_by_datasets
        )
        monkeypatch.setattr(m.ai_agent_eval_run_repository, "delete_by_agent", delete_by_agent)

        await svc.delete_agent(db, redis, 1)

        assert calls["eval_soft_delete"] == [[11, 12]]
        assert calls["eval_sample_delete"] == [[11, 12]]
        assert calls["eval_run_delete"] == [1]
        await run_after_commit_callbacks(db)
        assert calls["audit"][0]["after_value"] == {
            "eval_datasets_soft_deleted": 2,
            "eval_samples_deleted": 7,
            "eval_runs_deleted": 3,
        }

    async def test_delete_without_eval_assets_records_zero(self, env):
        db, redis, calls, svc = env
        await svc.delete_agent(db, redis, 1)
        assert calls["eval_soft_delete"] == []
        assert calls["eval_sample_delete"] == []
        # 无评测集的 Agent 其执行记录仍需清理（Run 不依赖评测集存在）
        assert calls["eval_run_delete"] == [1]
        await run_after_commit_callbacks(db)
        assert calls["audit"][0]["after_value"] == {
            "eval_datasets_soft_deleted": 0,
            "eval_samples_deleted": 0,
            "eval_runs_deleted": 0,
        }


class TestSetMcpNamespaceValidation:
    async def test_unregistered_namespace_rejected(self, env, monkeypatch):
        from app.service import ai_agent_service as m

        db, redis, _calls, svc = env

        async def list_registered_names(d, namespaces):
            return ["ns_a"]

        monkeypatch.setattr(
            m.ai_mcp_namespace_repository, "list_registered_names", list_registered_names
        )
        with pytest.raises(BusinessException) as exc:
            await svc.set_mcp(db, redis, 1, ["ns_a", "ns_ghost"])
        assert "ns_ghost" in str(exc.value)

    async def test_registered_namespace_accepted(self, env, monkeypatch):
        from app.service import ai_agent_service as m

        db, redis, _calls, svc = env

        async def list_registered_names(d, namespaces):
            return ["ns_a"]

        monkeypatch.setattr(
            m.ai_mcp_namespace_repository, "list_registered_names", list_registered_names
        )
        await svc.set_mcp(db, redis, 1, ["ns_a"])

    async def test_empty_namespaces_skips_lookup(self, env, monkeypatch):
        from app.service import ai_agent_service as m

        db, redis, _calls, svc = env
        seen = []

        async def list_registered_names(d, namespaces):
            seen.append(namespaces)
            return []

        monkeypatch.setattr(
            m.ai_mcp_namespace_repository, "list_registered_names", list_registered_names
        )
        await svc.set_mcp(db, redis, 1, [])
        assert seen == []


class TestSetSubagentsValidation:
    async def test_self_reference_rejected(self, env):
        db, redis, _calls, svc = env
        with pytest.raises(BusinessException) as exc:
            await svc.set_subagents(db, redis, 1, _subagents_form(1))
        assert "自身" in str(exc.value)

    async def test_missing_subagent_rejected(self, env, monkeypatch):
        db, redis, _calls, svc = env

        async def get_by_ids(d, ids):
            return [_agent(agent_id=2)]

        svc.ai_agent_repository.get_by_ids = get_by_ids
        with pytest.raises(BusinessException) as exc:
            await svc.set_subagents(db, redis, 1, _subagents_form(2, 9))
        assert "不存在" in str(exc.value)

    async def test_cycle_rejected(self, env, monkeypatch):
        """A→B→A：子 Agent 环会让推理期展开无限递归。"""
        db, redis, _calls, svc = env

        async def list_subagents(d, aid):
            if aid == 2:
                return [SimpleNamespace(subagent_agent_id=1)]
            return []

        svc.ai_agent_repository.list_subagents = list_subagents
        with pytest.raises(BusinessException) as exc:
            await svc.set_subagents(db, redis, 1, _subagents_form(2))
        assert "环" in str(exc.value)

    async def test_existing_indirect_cycle_rejected(self, env, monkeypatch):
        db, redis, _calls, svc = env

        async def list_subagents(d, aid):
            if aid == 2:
                return [SimpleNamespace(subagent_agent_id=3)]
            if aid == 3:
                return [SimpleNamespace(subagent_agent_id=1)]
            return []

        svc.ai_agent_repository.list_subagents = list_subagents
        with pytest.raises(BusinessException) as exc:
            await svc.set_subagents(db, redis, 1, _subagents_form(2))
        assert "环" in str(exc.value)

    async def test_acyclic_binding_accepted(self, env, monkeypatch):
        db, redis, _calls, svc = env
        replaced = []

        async def replace_subagents(d, aid, items):
            replaced.append(items)

        svc.ai_agent_repository.replace_subagents = replace_subagents
        await svc.set_subagents(db, redis, 1, _subagents_form(2, 3))
        assert [i["agent_id"] for i in replaced[0]] == [2, 3]
