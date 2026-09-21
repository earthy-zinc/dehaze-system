"""智能体版本仓储真实 MySQL 测试：版本号并发取号、版本列表分页下推 SQL。

版本号为 MAX+1，与插入非原子；并发发布由 (agent_id, version_no) 唯一键兜底，
冲突时 SAVEPOINT 回滚该次插入并按冲突号递增重试。
"""

import pytest
from sqlalchemy import inspect

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.entity.sys_ai_agent import SysAiAgent
from app.models.entity.sys_ai_agent_version import SysAiAgentVersion
from app.repository.ai_agent_version_repository import ai_agent_version_repository
from app.service import ai_agent_version_service as m
from app.service.ai_agent_version_service import agent_version_service


async def _seed_agent(db, code: str) -> SysAiAgent:
    agent = SysAiAgent(agent_code=code, name="版本测试", description="", model_id="m1")
    db.add(agent)
    await db.flush()
    return agent


def _seed_version(agent_id: int, version_no: int) -> SysAiAgentVersion:
    return SysAiAgentVersion(
        agent_id=agent_id, version_no=version_no, snapshot={"name": "v"}, status=1
    )


class TestVersionNoConflictRetry:
    async def test_retries_with_next_no_after_unique_conflict(self, db, mock_redis, monkeypatch):
        """取号撞已占用版本号：SAVEPOINT 回滚后按冲突号递增重试，不产生重复版本。"""
        agent = await _seed_agent(db, "ver_conflict_retry")
        db.add(_seed_version(agent.id, 1))
        await db.flush()

        # 固定取号返回已占用的 1，模拟并发发布下 MAX+1 读到同一版本号
        async def _next_no(d, aid):
            return 1

        monkeypatch.setattr(ai_agent_version_repository, "next_version_no", _next_no)

        async def _resolve(d, r, cfg, conv):
            return {}

        monkeypatch.setattr(m.agent_config_resolver, "resolve", _resolve)

        version = await agent_version_service._write_draft(
            db, mock_redis, agent, 1, "note", status=1
        )
        assert version.version_no == 2

    async def test_raises_when_conflicts_exhaust_retry_budget(self, db, mock_redis, monkeypatch):
        agent = await _seed_agent(db, "ver_conflict_exhaust")
        for version_no in range(1, 6):
            db.add(_seed_version(agent.id, version_no))
        await db.flush()

        async def _next_no(d, aid):
            return 1

        monkeypatch.setattr(ai_agent_version_repository, "next_version_no", _next_no)

        async def _resolve(d, r, cfg, conv):
            return {}

        monkeypatch.setattr(m.agent_config_resolver, "resolve", _resolve)

        with pytest.raises(BusinessException) as exc:
            await agent_version_service._write_draft(db, mock_redis, agent, 1, "note", status=1)
        assert exc.value.code == ResultCode.DATA_EXISTS


class TestListVersionsSql:
    async def test_pagination_pushed_to_sql_and_snapshot_deferred(self, db):
        agent = await _seed_agent(db, "ver_page_sql")
        for version_no in range(1, 6):
            db.add(_seed_version(agent.id, version_no))
        await db.flush()

        first_page, total = await ai_agent_version_repository.list_versions(db, agent.id, 0, 2)
        assert total == 5
        assert [v.version_no for v in first_page] == [5, 4]
        # 列表接口不加载快照大字段（详情页才读）
        assert "snapshot" in inspect(first_page[0]).unloaded

        last_page, _ = await ai_agent_version_repository.list_versions(db, agent.id, 4, 2)
        assert [v.version_no for v in last_page] == [1]

    async def test_returns_zero_total_without_querying_rows(self, db):
        agent = await _seed_agent(db, "ver_page_empty")
        rows, total = await ai_agent_version_repository.list_versions(db, agent.id, 0, 10)
        assert (rows, total) == ([], 0)
