"""算法监控统计报表路由测试：days 下界校验。

days 曾为裸 int + service 静默回退默认值（`days <= 0 → 7`）：既掩盖调用方传参错误，
也与"参数非法即拒绝"的三端口径相悖。现改为路由层 `Query(ge=1)` → 负值/零值 400+A0400。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import algorithm as algorithm_module
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

_PATH = "/api/v1/algorithms/1/monitor/stats"


@pytest.fixture
async def client():
    async def _override_db():
        return object()

    async def _override_user():
        return make_user_context(1, username="u")

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as c:
        yield c
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


class TestMonitorStatsDays:
    @pytest.mark.parametrize("days", [0, -1])
    async def test_non_positive_days_rejected(self, client, days):
        resp = await client.get(_PATH, params={"days": days})

        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_default_days_reaches_service(self, client, monkeypatch):
        captured = {}

        async def _fake_report(db, algorithm_id, days=7):
            captured.update(algorithm_id=algorithm_id, days=days)
            return []

        monkeypatch.setattr(
            algorithm_module.algorithm_service, "get_monitor_stats_report", _fake_report
        )
        resp = await client.get(_PATH)

        assert resp.status_code == 200
        assert resp.json()["code"] == "00000"
        assert captured == {"algorithm_id": 1, "days": 7}
