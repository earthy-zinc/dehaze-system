"""图像任务路由层测试（/api/v1/images/* 契约：提交/轮询/取消/历史 + 归属校验）。

遵循 05-python-test-rules：marker=api、dependency_overrides + 真实测试库 db。
状态值域：1=处理中；2=已完成；3=失败；4=已取消。
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.entity.sys_log import SysPredLog
from app.models.enum.log_status import LogStatus
from app.service.prediction.prediction_service import prediction_service
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

OWNER_ID = 101
INTRUDER_ID = 202
MODEL_ID = "aecrnet"
PRED_MD5 = "d" * 32
ORIGIN_MD5 = "c" * 32


def _ctx(user_id: int):
    return make_user_context(user_id, username=f"u{user_id}", roles=[], permissions=[])


@pytest.fixture
async def image_client(db: AsyncSession):
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


def _log(
    create_by: int,
    status: int = LogStatus.PROCESSING.value,
    pred_md5: str = "",
    pred_url: str = "",
) -> SysPredLog:
    return SysPredLog(
        model_id=MODEL_ID,
        origin_md5="0" * 32,
        origin_url="http://example.com/a.png",
        pred_md5=pred_md5,
        pred_url=pred_url,
        time=0,
        status=status,
        create_by=create_by,
    )


async def _create_log(db: AsyncSession, create_by: int, **kwargs) -> SysPredLog:
    log = _log(create_by, **kwargs)
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


# ===== 提交任务参数校验 =====


async def test_submit_rejects_missing_image_source(image_client):
    """fileId 与 imageUrl 均缺省 → A0400（图片来源不能为空）"""
    resp = await image_client.post("/api/v1/images/edits", json={"model": MODEL_ID})
    assert resp.status_code == 400
    assert resp.json()["code"] == ResultCode.PARAM_IS_NULL.code


async def test_submit_rejects_invalid_params_json(image_client):
    """params 非合法 JSON → A0400，不进入编排层"""
    resp = await image_client.post(
        "/api/v1/images/edits",
        json={"model": MODEL_ID, "imageUrl": "http://cdn/a.png", "params": "not-a-json{[["},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == ResultCode.PARAM_ERROR.code


async def test_submit_cache_hit_carries_result_and_origin_md5(image_client, monkeypatch):
    """提交命中缓存/拦截器（直接 completed）响应必须带 resultMd5 与 originMd5：
    调用方以两个内容 MD5 落库各自 sys_pred_log，缺失会落 null 或被 URL 的 MD5 冒充。
    """

    async def _hit(**kwargs):
        assert kwargs["model"] == MODEL_ID
        return {
            "taskId": 555,
            "status": LogStatus.COMPLETED.value,
            "resultUrl": "http://cdn.example.com/result.png",
            "resultMd5": PRED_MD5,
            "originMd5": ORIGIN_MD5,
            "resultThumbnailUrl": None,
            "time": 12,
        }

    monkeypatch.setattr(prediction_service, "predict", _hit)
    resp = await image_client.post(
        "/api/v1/images/edits",
        json={"model": MODEL_ID, "imageUrl": "http://cdn.example.com/a.png"},
    )

    data = resp.json()["data"]
    assert data["taskId"] == 555
    assert data["status"] == LogStatus.COMPLETED.value
    assert data["resultMd5"] == PRED_MD5
    assert data["originMd5"] == ORIGIN_MD5


# ===== 任务查询 / 取消：归属校验 =====


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/v1/images/tasks/{task_id}", "get"),
        ("/api/v1/images/tasks/{task_id}/cancel", "post"),
    ],
)
async def test_read_and_cancel_other_users_task_not_found(db, image_client, path, method):
    """非任务归属者访问（读/取消）→ A0401 防枚举（与不存在任务同口径）"""
    log = await _create_log(db, create_by=OWNER_ID)

    async def _override_user():
        return _ctx(INTRUDER_ID)

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp = await getattr(image_client, method)(path.format(task_id=log.id))
    assert resp.json()["code"] == "A0401"


async def test_get_task_rejects_non_integer_id(image_client):
    resp = await image_client.get("/api/v1/images/tasks/not-an-int")
    assert resp.status_code == 400
    assert resp.json()["code"] == ResultCode.PARAM_ERROR.code


async def test_read_own_processing_task_ok(image_client, db):
    log = await _create_log(db, create_by=OWNER_ID)
    resp = await image_client.get(f"/api/v1/images/tasks/{log.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert body["data"]["taskId"] == log.id
    assert body["data"]["status"] == LogStatus.PROCESSING.value


async def test_get_completed_task_returns_result_and_origin_md5(image_client, db):
    """已完成任务返回 resultUrl/resultMd5/originMd5，与提交命中口径一致。"""
    log = await _create_log(
        db,
        create_by=OWNER_ID,
        status=LogStatus.COMPLETED.value,
        pred_md5=PRED_MD5,
        pred_url="http://cdn.example.com/result.png",
    )

    resp = await image_client.get(f"/api/v1/images/tasks/{log.id}")

    data = resp.json()["data"]
    assert data["status"] == LogStatus.COMPLETED.value
    assert data["resultUrl"] == log.pred_url
    assert data["resultMd5"] == PRED_MD5
    assert data["originMd5"] == log.origin_md5


async def test_get_failed_task_returns_error_message(image_client, db):
    log = await _create_log(db, create_by=OWNER_ID, status=LogStatus.FAILED.value)
    log.error_message = "图像推理服务响应超时"
    await db.commit()

    resp = await image_client.get(f"/api/v1/images/tasks/{log.id}")
    assert resp.json()["data"]["errorMessage"] == "图像推理服务响应超时"


async def test_get_cancelled_task_returns_no_result_fields(image_client, db):
    """已取消（4）：仅返回 taskId + status，无结果字段"""
    log = await _create_log(db, create_by=OWNER_ID, status=LogStatus.CANCELLED.value)

    resp = await image_client.get(f"/api/v1/images/tasks/{log.id}")
    data = resp.json()["data"]
    assert data["taskId"] == log.id
    assert data["status"] == LogStatus.CANCELLED.value
    assert data.get("resultUrl") is None


async def test_cancel_other_users_task_not_found(image_client, db):
    log = await _create_log(db, create_by=INTRUDER_ID)
    resp = await image_client.post(f"/api/v1/images/tasks/{log.id}/cancel")
    assert resp.json()["code"] == "A0401"
    await db.refresh(log)
    assert log.status == LogStatus.PROCESSING.value


async def test_cancel_own_task(image_client, db):
    log = await _create_log(db, create_by=OWNER_ID)

    resp = await image_client.post(f"/api/v1/images/tasks/{log.id}/cancel")
    assert resp.json()["data"]["status"] == LogStatus.CANCELLED.value


# ===== 历史列表：用户隔离 + 模型筛选 =====


async def test_list_tasks_scoped_to_current_user(image_client, db):
    await _create_log(db, create_by=OWNER_ID)
    await _create_log(db, create_by=INTRUDER_ID)

    resp = await image_client.get("/api/v1/images/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "00000"
    assert body["data"]["total"] == 1
    assert body["data"]["list"][0]["model"] == MODEL_ID

    async def _override_user():
        return _ctx(INTRUDER_ID)

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    resp2 = await image_client.get("/api/v1/images/tasks")
    assert resp2.json()["data"]["total"] == 1


async def test_list_tasks_filters_by_model(image_client, db):
    await _create_log(db, create_by=OWNER_ID)
    other = _log(OWNER_ID)
    other.model_id = "aodnet"
    db.add(other)
    await db.commit()

    resp = await image_client.get("/api/v1/images/tasks", params={"model": "aodnet"})
    body = resp.json()["data"]
    assert body["total"] == 1
    assert body["list"][0]["model"] == "aodnet"


@pytest.mark.parametrize("params", [{"pageSize": 0}, {"pageSize": 101}, {"pageNum": 0}])
async def test_list_tasks_rejects_out_of_range_paging(image_client, params):
    resp = await image_client.get("/api/v1/images/tasks", params=params)
    assert resp.status_code == 400
