"""Agent 契约字段增强单测：分类标签 tags / 类型筛选 type / 列表关联计数"""

import pytest

pytestmark = pytest.mark.requires_db

from app.models.entity.sys_ai_agent_mcp import SysAiAgentMcp
from app.models.entity.sys_ai_agent_skill import SysAiAgentSkill
from app.models.entity.sys_ai_agent_subagent import SysAiAgentSubagent
from app.models.schema.ai_agent import AgentCreate, AgentUpdate
from app.service.ai_agent_service import AgentService

_MODEL_ID = "qwen3-0.6b"


def _form(code, name, **kw):
    return AgentCreate(agent_code=code, name=name, model_id=_MODEL_ID, **kw)


async def _seed_agent(db, redis, code, name, **kw) -> int:
    detail = await AgentService().create_agent(db, redis, _form(code, name, **kw))
    return detail.id


class TestAgentTags:
    async def test_create_with_tags(self, db, mock_redis):
        detail = await AgentService().create_agent(
            db, mock_redis, _form("tagged", "标签Agent", tags=["客服", "去雾"])
        )
        assert detail.tags == ["客服", "去雾"]

    async def test_create_without_tags_defaults_empty(self, db, mock_redis):
        detail = await AgentService().create_agent(db, mock_redis, _form("untagged", "无标签"))
        assert detail.tags == []

    async def test_update_tags(self, db, mock_redis):
        agent_id = await _seed_agent(db, mock_redis, "tag-upd", "标签更新", tags=["旧"])
        detail = await AgentService().update_agent(
            db, mock_redis, agent_id, AgentUpdate(tags=["新1", "新2"])
        )
        assert detail.tags == ["新1", "新2"]

    async def test_copy_keeps_tags(self, db, mock_redis):
        agent_id = await _seed_agent(db, mock_redis, "tag-src", "标签复制", tags=["a"])
        detail = await AgentService().copy_agent(db, mock_redis, agent_id, "tag-dst")
        assert detail.tags == ["a"]


class TestTypeFilter:
    async def test_filter_agent(self, db, mock_redis):
        normal = await _seed_agent(db, mock_redis, "t-normal", "普通")
        await _seed_agent(db, mock_redis, "t-sub", "子Agent", is_subagent=True)
        await _seed_agent(db, mock_redis, "t-team", "团队", is_team=True)
        svc = AgentService()

        result = await svc.list_agents(db, mock_redis, 1, 10, agent_type="agent")
        assert [i.id for i in result.list] == [normal]

        result = await svc.list_agents(db, mock_redis, 1, 10, agent_type="subagent")
        assert [i.agent_code for i in result.list] == ["t-sub"]

        result = await svc.list_agents(db, mock_redis, 1, 10, agent_type="team")
        assert [i.agent_code for i in result.list] == ["t-team"]

    async def test_no_filter_returns_all(self, db, mock_redis):
        await _seed_agent(db, mock_redis, "n1", "普通")
        await _seed_agent(db, mock_redis, "n2", "子Agent", is_subagent=True)
        result = await AgentService().list_agents(db, mock_redis, 1, 10)
        assert result.total == 2


class TestListCounts:
    async def test_list_agents_aggregates_relations(self, db, mock_redis):
        svc = AgentService()
        agent_id = await _seed_agent(db, mock_redis, "cnt-a", "计数A")
        other_id = await _seed_agent(db, mock_redis, "cnt-b", "计数B")

        db.add_all(
            [
                SysAiAgentSkill(agent_id=agent_id, skill_name="skill1"),
                SysAiAgentSkill(agent_id=agent_id, skill_name="skill2"),
                SysAiAgentMcp(agent_id=agent_id, mcp_namespace="ns1"),
                SysAiAgentSubagent(parent_agent_id=agent_id, subagent_agent_id=other_id),
            ]
        )
        await db.flush()

        result = await svc.list_agents(db, mock_redis, 1, 10)
        by_code = {i.agent_code: i for i in result.list}
        assert by_code["cnt-a"].skill_count == 2
        assert by_code["cnt-a"].mcp_count == 1
        assert by_code["cnt-a"].sub_agent_count == 1
        assert by_code["cnt-b"].skill_count == 0
        assert by_code["cnt-b"].mcp_count == 0
        assert by_code["cnt-b"].sub_agent_count == 0
