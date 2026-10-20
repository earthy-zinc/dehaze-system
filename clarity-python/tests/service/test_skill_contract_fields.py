"""Skill 契约字段增强单测：适用场景 scene / 列表关联计数 agentCount"""

import pytest

from app.models.entity.sys_ai_agent_skill import SysAiAgentSkill
from app.models.schema.ai_skill import SkillCreate, SkillUpdate
from app.service.ai_skill_service import SkillManageService

pytestmark = pytest.mark.requires_db


async def _create_skill(db, name, **kw):
    """创建并启用：新建默认禁用（§2.6.11），普通用户可见性断言需启用态。"""
    created = await SkillManageService().create_skill(
        db, SkillCreate(name=name, description=f"{name}描述", instruction="指令", **kw)
    )
    assert created.status == 0
    return await SkillManageService().set_status(db, created.id, enabled=True)


class TestSkillScene:
    async def test_create_with_scene(self, db):
        result = await _create_skill(db, "scene-skill", scene="客服问答")
        assert result.scene == "客服问答"

    async def test_create_default_empty_scene(self, db):
        result = await _create_skill(db, "no-scene-skill")
        assert result.scene == ""

    async def test_update_scene(self, db):
        created = await _create_skill(db, "scene-upd", scene="旧场景")
        result = await SkillManageService().update_skill(
            db, created.id, SkillUpdate(scene="新场景")
        )
        assert result.scene == "新场景"

    async def test_list_returns_scene(self, db):
        await _create_skill(db, "scene-list", scene="去雾调度")
        result = await SkillManageService().list_skills(db, enabled_only=True)
        assert result.list[0].scene == "去雾调度"


class TestSkillAgentCount:
    async def test_list_aggregates_agent_count(self, db):
        await _create_skill(db, "ref-skill")
        db.add_all(
            [
                SysAiAgentSkill(agent_id=1, skill_name="ref-skill"),
                SysAiAgentSkill(agent_id=2, skill_name="ref-skill"),
            ]
        )
        await db.flush()

        result = await SkillManageService().list_skills(db, enabled_only=True)
        assert result.list[0].agentCount == 2

    async def test_list_zero_count_without_references(self, db):
        await _create_skill(db, "orphan-skill")
        result = await SkillManageService().list_skills(db, enabled_only=True)
        assert result.list[0].agentCount == 0
