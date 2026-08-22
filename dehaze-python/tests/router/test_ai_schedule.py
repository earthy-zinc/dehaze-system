import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.schema.ai_schedule import (
    NextTimesPreview,
    RunHistoryItem,
    ScheduleDetail,
    ScheduleListItem,
)
from app.models.schema.common import PageResult
from app.router import ai_schedule
from tests.stubs import make_user_context


def _detail(**overrides) -> ScheduleDetail:
    base = dict(
        id=1,
        userId=42,
        name="每日去雾",
        cron="0 9 * * *",
        timezone="Asia/Shanghai",
        enabled=1,
        status=1,
        circuitStreak=0,
    )
    base.update(overrides)
    return ScheduleDetail(**base)


@pytest.fixture
async def ai_client():
    async def _override_db():
        return object()

    async def _override_user():
        return make_user_context(42)

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_next_times_path_registered_before_path_param(app):
    schema = app.openapi()
    paths = schema["paths"]
    assert "/api/v1/ai/scheduled-tasks/next-times" in paths
    assert "get" in paths["/api/v1/ai/scheduled-tasks/next-times"]
    assert "/api/v1/ai/scheduled-tasks/{schedule_id}" in paths


async def test_create_passes_form_and_returns_detail(ai_client, monkeypatch):
    captured = {}

    async def fake_create(db, user_id, form):
        captured["user_id"] = user_id
        captured["name"] = form.name
        captured["input_type"] = form.input["type"]
        return _detail()

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "create", fake_create)

    resp = await ai_client.post(
        "/api/v1/ai/scheduled-tasks",
        json={
            "name": "每日去雾",
            "cron": "0 9 * * *",
            "input": {"type": "fixed", "images": [1, 2]},
            "output": {"type": "message"},
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert body["data"]["id"] == 1
    assert captured["user_id"] == 42
    assert captured["name"] == "每日去雾"
    assert captured["input_type"] == "fixed"


async def test_list_returns_page_result(ai_client, monkeypatch):
    item = ScheduleListItem(**_detail().model_dump(), lastRun=None)

    async def fake_list_page(db, user_id, query):
        return PageResult(list=[item], total=1)

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "list_page", fake_list_page)

    resp = await ai_client.get("/api/v1/ai/scheduled-tasks?pageNum=1&pageSize=10")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["total"] == 1
    assert data["list"][0]["name"] == "每日去雾"


async def test_get_detail_returns_detail(ai_client, monkeypatch):
    async def fake_get_detail(db, user_id, schedule_id):
        assert user_id == 42
        assert schedule_id == 9
        return _detail(id=9)

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "get_detail", fake_get_detail)

    resp = await ai_client.get("/api/v1/ai/scheduled-tasks/9")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == 9


async def test_update_passes_fields(ai_client, monkeypatch):
    captured = {}

    async def fake_update(db, user_id, schedule_id, form):
        captured["schedule_id"] = schedule_id
        captured["new_name"] = form.name
        return _detail(name=form.name)

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "update", fake_update)

    resp = await ai_client.put("/api/v1/ai/scheduled-tasks/9", json={"name": "新名称"})

    assert resp.status_code == 200
    assert captured["schedule_id"] == 9
    assert captured["new_name"] == "新名称"
    assert resp.json()["data"]["name"] == "新名称"


async def test_set_status_passes_enabled(ai_client, monkeypatch):
    captured = {}

    async def fake_set_enabled(db, user_id, schedule_id, enabled):
        captured["schedule_id"] = schedule_id
        captured["enabled"] = enabled

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "set_enabled", fake_set_enabled)

    resp = await ai_client.patch("/api/v1/ai/scheduled-tasks/9/status", json={"enabled": False})

    assert resp.status_code == 200
    assert captured["schedule_id"] == 9
    assert captured["enabled"] == 0


async def test_delete_calls_service(ai_client, monkeypatch):
    captured = {}

    async def fake_delete(db, user_id, schedule_id):
        captured["schedule_id"] = schedule_id

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "delete", fake_delete)

    resp = await ai_client.delete("/api/v1/ai/scheduled-tasks/9")
    assert resp.status_code == 200
    assert captured["schedule_id"] == 9


async def test_next_times_preview(ai_client, monkeypatch):
    async def fake_preview(cron, count):
        return NextTimesPreview(description="每天 09点00分", nextTimes=[])

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "preview_next_times", fake_preview)

    resp = await ai_client.get("/api/v1/ai/scheduled-tasks/next-times?cron=0%209%20*%20*%20*")
    assert resp.status_code == 200
    assert resp.json()["data"]["description"] == "每天 09点00分"


async def test_history_returns_page(ai_client, monkeypatch):
    captured = {}
    run = RunHistoryItem(
        id=5,
        scheduleId=9,
        status=1,
        credits=1.5,
        durationMs=120,
        conversationId=100,
        windowStart=None,
        createTime=None,
    )

    async def fake_list_history(db, user_id, schedule_id, page, size):
        captured["user_id"] = user_id
        captured["schedule_id"] = schedule_id
        return PageResult(list=[run], total=1)

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "list_history", fake_list_history)

    resp = await ai_client.get("/api/v1/ai/scheduled-tasks/9/history?pageNum=1&pageSize=10")

    assert resp.status_code == 200
    assert captured["user_id"] == 42
    assert captured["schedule_id"] == 9
    data = resp.json()["data"]
    assert data["total"] == 1
    assert data["list"][0]["scheduleId"] == 9
    assert data["list"][0]["credits"] == 1.5


async def test_run_returns_accepted_and_triggers_in_background(ai_client, monkeypatch):
    triggered = asyncio.Event()

    async def fake_get_detail(db, user_id, schedule_id):
        return _detail(id=9)

    async def fake_trigger_once(db, redis, schedule_id, user_id, *, manual):
        triggered.set()

    @asynccontextmanager
    async def fake_db_session():
        yield object()

    monkeypatch.setattr(ai_schedule.scheduled_task_service, "get_detail", fake_get_detail)
    monkeypatch.setattr(ai_schedule.schedule_executor, "trigger_once", fake_trigger_once)
    monkeypatch.setattr(ai_schedule, "get_db_session", fake_db_session)
    monkeypatch.setattr(
        "app.dependencies.redis.get_redis_client",
        lambda: asyncio.sleep(0),
    )

    resp = await ai_client.post("/api/v1/ai/scheduled-tasks/9/run")

    assert resp.status_code == 200
    assert resp.json()["data"] == {"accepted": True}

    await asyncio.wait_for(triggered.wait(), timeout=2)
    assert triggered.is_set()
