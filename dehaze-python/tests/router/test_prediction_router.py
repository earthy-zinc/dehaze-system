"""预测路由层越权测试（F-REC-002 审查修复：任务查询/取消/日志列表归属校验）。

遵循 05-python-test-rules：marker=api、dependency_overrides + 真实测试库 db。
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

OWNER_ID = 101
INTRUDER_ID = 202


def _ctx(user_id: int):
    return make_user_context(user_id, username=f"u{user_id}", roles=[], permissions=[])


@pytest.fixture
async def pred_client(db: AsyncSession):
    async def _override_db():
        return db

    async def _override_user():
        return _ctx(OWNER_ID)

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


async def _create_processing_log(db: AsyncSession, create_by: int) -> SysPredLog:
    log = SysPredLog(
        algorithm_id=1,
        origin_md5="0" * 32,
        origin_url="http://example.com/a.png",
        pred_md5="",
        pred_url="",
        time=0,
        status=LogStatus.PROCESSING.value,
        create_by=create_by,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/v1/prediction/{task_id}", "get"),
        ("/api/v1/prediction/{task_id}/cancel", "post"),
    ],
)
async def test_read_and_cancel_other_users_task_not_found(db, pred_client, path, method):
    """非任务归属者访问（读/取消）→ A0401 防枚举（与不存在任务同口径）"""
    log = await _create_processing_log(db, create_by=OWNER_ID)

    async def _override_user():
        return _ctx(INTRUDER_ID)

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp = await getattr(pred_client, method)(path.format(task_id=log.id))
    assert resp.json()["code"] == "A0401"


async def test_read_own_task_ok(pred_client, db):
    log = await _create_processing_log(db, create_by=OWNER_ID)
    resp = await pred_client.get(f"/api/v1/prediction/{log.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert body["data"]["logId"] == log.id
    assert body["data"]["status"] == LogStatus.PROCESSING.value


async def test_cancel_other_users_task_not_found(pred_client, db):
    log = await _create_processing_log(db, create_by=INTRUDER_ID)
    resp = await pred_client.post(f"/api/v1/prediction/{log.id}/cancel")
    assert resp.json()["code"] == "A0401"
    # 状态未被篡改
    await db.refresh(log)
    assert log.status == LogStatus.PROCESSING.value


async def test_logs_scoped_to_current_user(pred_client, db):
    await _create_processing_log(db, create_by=OWNER_ID)
    await _create_processing_log(db, create_by=INTRUDER_ID)

    resp = await pred_client.get("/api/v1/prediction/logs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert body["data"]["total"] == 1
    assert body["data"]["list"][0]["id"] is not None

    # 入侵者视角：只能看到自己的那条
    async def _override_user():
        return _ctx(INTRUDER_ID)

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp2 = await pred_client.get("/api/v1/prediction/logs")
    assert resp2.json()["data"]["total"] == 1
