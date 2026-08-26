import pytest
from fastapi import HTTPException

from app.dependencies.auth import UserContext

pytestmark = pytest.mark.api
from app.router.order import get_order_page, get_order_stats, list_refunds


def _user(roles=None, permissions=None):
    return UserContext(
        id=1,
        username="u",
        roles=roles or [],
        permissions=permissions or [],
    )


def _fake_service(called):
    async def _fake(db, query, current_user=None):
        called["hit"] = True
        return {"list": [], "total": 0}

    return _fake


@pytest.mark.parametrize(
    "endpoint,permission,service",
    [
        (get_order_page, "order:list", "order_service.list_paged"),
        (list_refunds, "order:refund:list", "refund_service.list_refunds"),
        (get_order_stats, "order:stats", "order_service.get_stats"),
    ],
)
async def test_permission_required_without_perm_403(monkeypatch, endpoint, permission, service):
    user = _user(permissions=[])

    async def _boom(*a, **k):
        raise AssertionError(f"未拦截 {permission}，直接触达业务逻辑")

    monkeypatch.setattr(f"app.router.order.{service}", _boom)

    with pytest.raises(HTTPException) as ei:
        await endpoint(user=user, db=None)
    assert ei.value.status_code == 403


async def test_order_page_with_permission_reaches_service(monkeypatch):
    user = _user(permissions=["order:list"])
    called = {"hit": False}

    monkeypatch.setattr("app.router.order.order_service.list_paged", _fake_service(called))
    await get_order_page(user=user, db=None)
    assert called["hit"] is True


async def test_root_bypasses_permission(monkeypatch):
    user = _user(roles=["ROOT"], permissions=[])
    called = {"hit": False}

    monkeypatch.setattr("app.router.order.order_service.list_paged", _fake_service(called))
    await get_order_page(user=user, db=None)
    assert called["hit"] is True
