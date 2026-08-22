import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_skill import SkillResult
from app.models.schema.common import PageResult
from app.router import ai_skill


class _FakeUser:
    def __init__(self, is_root=False, permissions=()):
        self.is_root = is_root
        self.permissions = list(permissions)


def _detail(**overrides) -> SkillResult:
    base = dict(
        id=1,
        name="去雾工作流",
        description="指导去雾",
        content="# 步骤",
        status=1,
        source="admin",
    )
    base.update(overrides)
    return SkillResult(**base)


@pytest.fixture
async def ai_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser()}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def test_skill_paths_registered(app):
    schema = app.openapi()
    paths = schema["paths"]
    assert "/api/v1/ai/skills" in paths
    assert "get" in paths["/api/v1/ai/skills"]
    assert "post" in paths["/api/v1/ai/skills"]
    assert "put" in paths["/api/v1/ai/skills/{skill_id}"]
    assert "patch" in paths["/api/v1/ai/skills/{skill_id}/status"]
    assert "delete" in paths["/api/v1/ai/skills/{skill_id}"]


async def test_list_admin_passes_enabled_only_false(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_list(db, *, enabled_only, page, size, keyword):
        captured["enabled_only"] = enabled_only
        return PageResult(list=[_detail()], total=1)

    monkeypatch.setattr(ai_skill.skill_manage_service, "list_skills", fake_list)
    resp = await client.get("/api/v1/ai/skills?pageNum=1&pageSize=10&keyword=去雾")
    assert resp.status_code == 200
    assert resp.json()["data"]["total"] == 1
    assert captured["enabled_only"] is False


async def test_list_normal_user_passes_enabled_only_true(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=False)
    captured = {}

    async def fake_list(db, *, enabled_only, page, size, keyword):
        captured["enabled_only"] = enabled_only
        return PageResult(list=[], total=0)

    monkeypatch.setattr(ai_skill.skill_manage_service, "list_skills", fake_list)
    resp = await client.get("/api/v1/ai/skills")
    assert resp.status_code == 200
    assert captured["enabled_only"] is True


async def test_create_admin_success(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_create(db, form):
        captured["name"] = form.name
        return _detail(name=form.name)

    monkeypatch.setattr(ai_skill.skill_manage_service, "create_skill", fake_create)
    resp = await client.post(
        "/api/v1/ai/skills",
        json={"name": "新技能", "description": "描述", "content": "# 步骤"},
    )
    assert resp.status_code == 200
    assert captured["name"] == "新技能"


async def test_create_normal_user_forbidden(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=False, permissions=[])
    resp = await client.post(
        "/api/v1/ai/skills",
        json={"name": "新技能", "description": "描述", "content": "# 步骤"},
    )
    assert resp.status_code == 403


async def test_update_passes_fields(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_update(db, skill_id, form):
        captured["skill_id"] = skill_id
        captured["content"] = form.content
        return _detail(content=form.content)

    monkeypatch.setattr(ai_skill.skill_manage_service, "update_skill", fake_update)
    resp = await client.put("/api/v1/ai/skills/7", json={"content": "# 新步骤"})
    assert resp.status_code == 200
    assert captured["skill_id"] == 7
    assert captured["content"] == "# 新步骤"


async def test_set_status(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_status(db, skill_id, enabled):
        captured["skill_id"] = skill_id
        captured["enabled"] = enabled

    monkeypatch.setattr(ai_skill.skill_manage_service, "set_status", fake_status)
    resp = await client.patch("/api/v1/ai/skills/7/status", json={"enabled": False})
    assert resp.status_code == 200
    assert captured["skill_id"] == 7
    assert captured["enabled"] is False


async def test_set_status_normal_user_forbidden(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=False, permissions=[])
    resp = await client.patch("/api/v1/ai/skills/7/status", json={"enabled": False})
    assert resp.status_code == 403


async def test_delete_passes_id(ai_client, monkeypatch):
    client, state = ai_client
    state["user"] = _FakeUser(is_root=True)
    captured = {}

    async def fake_delete(db, skill_id):
        captured["skill_id"] = skill_id

    monkeypatch.setattr(ai_skill.skill_manage_service, "delete_skill", fake_delete)
    resp = await client.delete("/api/v1/ai/skills/7")
    assert resp.status_code == 200
    assert captured["skill_id"] == 7
