"""促销活动分页路由测试：pageSize 上下界。

原本只有 `ge=1`（漏 `le=100`），与 `BasePageQuery` 主口径及 go `parsePagination`
（`>100 → 400`）不一致；补齐后三端统一 1..100。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import promotion as promotion_module
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

_PATH = "/api/v1/packages/promotions/page"


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


def _patch_page(monkeypatch, captured):
    async def _fake_page(db, page_num, page_size, **kwargs):
        captured.update(page_num=page_num, page_size=page_size)
        return {"list": [], "total": 0}

    monkeypatch.setattr(promotion_module.promotion_service, "get_page", _fake_page)


class TestPromotionPageSize:
    @pytest.mark.parametrize("page_size", [0, -1, 101])
    async def test_out_of_range_rejected(self, client, page_size):
        resp = await client.get(_PATH, params={"pageSize": page_size})

        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_upper_bound_accepted(self, client, monkeypatch):
        captured = {}
        _patch_page(monkeypatch, captured)

        resp = await client.get(_PATH, params={"pageSize": 100})

        assert resp.status_code == 200
        assert captured["page_size"] == 100

    async def test_default_page_size_is_ten(self, client, monkeypatch):
        captured = {}
        _patch_page(monkeypatch, captured)

        resp = await client.get(_PATH)

        assert resp.status_code == 200
        assert captured == {"page_num": 1, "page_size": 10}
