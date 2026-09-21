import json

import pytest
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.auth import get_current_user
from app.infrastructure.a2a.a2a_server import a2a_server

pytestmark = pytest.mark.api


@pytest.fixture
def a2a_client_env(monkeypatch):
    card = {"name": "去雾助手", "version": "1", "url": "https://x/a2a", "capabilities": {}}

    async def _fake_card(db, redis, agent_id, base_url):
        return card

    monkeypatch.setattr(a2a_server, "build_agent_card", _fake_card)

    return card


async def test_agent_card_reachable_via_m2m_context(a2a_client_env, mock_redis):
    from app.router.a2a import agent_card

    req = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
        }
    )
    # db 仅透传给被打桩的 build_agent_card，用无绑定真实会话实例满足契约
    resp = await agent_card(1, req, db=AsyncSession(), redis=mock_redis)
    body = json.loads(bytes(resp.body))
    assert body["name"] == "去雾助手"


async def test_get_current_user_prefers_m2m_context(mock_redis):
    req = Request(
        {"type": "http", "method": "GET", "path": "/", "query_string": b"", "headers": []}
    )
    req.state.user_context = {"id": 0, "username": "a2a", "is_m2m": True}

    user = await get_current_user(request=req, credentials=None, redis=mock_redis)
    assert user.is_m2m is True
    assert user.username == "a2a"
