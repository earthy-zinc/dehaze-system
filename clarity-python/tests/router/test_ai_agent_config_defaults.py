"""推理参数系统默认值只读端点测试（GET /api/v1/ai/agents/config-defaults）

覆盖重点：路由注册在 /{agent_id} 之前（否则被路径参数吞掉）、登录即可读、
响应与代码常量 REASONING_DEFAULTS 单源一致（防前端硬编码漂移）。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_agent import AgentConfigDefaults
from app.service.ai.strategies.agent_config_resolver import REASONING_DEFAULTS

pytestmark = pytest.mark.api


_PATH = "/api/v1/ai/agents/config-defaults"
_EXPECTED = AgentConfigDefaults(**REASONING_DEFAULTS).model_dump(by_alias=True)


class _FakeUser:
    def __init__(self, id=8, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


@pytest.fixture
async def defaults_client():
    async def _override_db():
        return object()

    async def _override_user():
        return _FakeUser()

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_path_registered(app):
    assert _PATH in app.openapi()["paths"]


async def test_defaults_readable_and_matches_code_constant(defaults_client):
    """无 ai:agent:manage 的普通用户也能读：纯静态常量，非管理敏感数据。"""
    resp = await defaults_client.get(_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert body["data"] == _EXPECTED


def test_contract_model_covers_every_constant():
    """契约模型字段必须与常量键一一对应（新增常量漏补契约即失败）。"""
    assert set(AgentConfigDefaults.model_fields) == set(REASONING_DEFAULTS)
