import json

import pytest
from fastapi import Request

pytestmark = pytest.mark.api

from app.dependencies.auth import get_current_user
from app.infrastructure.a2a.a2a_server import a2a_server


@pytest.fixture
def a2a_client_env(monkeypatch):
    card = {
        "name": "去雾助手",
        "version": "1",
        "url": "https://x/a2a",
        "capabilities": {}
    }

    async def _fake_card(db, redis, agent_id, base_url):
        return card

    monkeypatch.setattr(a2a_server, "build_agent_card", _fake_card)

    yield card


async def test_agent_card_reachable_via_m2m_context(a2a_client_env):
    from app.router.a2a import agent_card

    req = Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("test", 80)
    })
    resp = await agent_card(1, req, db=None, redis=None)
    body = json.loads(resp.body)
    assert body["name"] == "去雾助手"


async def test_get_current_user_prefers_m2m_context(mock_redis):
    class _Req:
        state = type("S", (), {"user_context": {"id": 0, "username": "a2a", "is_m2m": True}})()

    user = await get_current_user(request=_Req(), credentials=None, redis=mock_redis)
    assert user.is_m2m is True
    assert user.username == "a2a"
