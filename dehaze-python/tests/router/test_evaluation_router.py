"""评估路由层测试（归属校验 / 用户隔离传参 / 状态字段口径）。

遵循 05-python-test-rules：marker=api、dependency_overrides + monkeypatch service、
只断言业务结果。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.models.enum.log_status import LogStatus
from app.router import evaluation
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.api

OWNER_ID = 3001001
OTHER_ID = 3001002


def _ctx(user_id: int):
    return make_user_context(user_id, username=f"u{user_id}")


def _log(user_id: int, status: int = LogStatus.COMPLETED.value, result=None):
    return SimpleNamespace(
        id=1,
        algorithm_id=1,
        pred_md5="a" * 32,
        pred_url="http://storage/pred.jpg",
        gt_md5="b" * 32,
        gt_url="http://storage/gt.jpg",
        status=status,
        error_message=None,
        time=3,
        result=result,
        create_by=user_id,
        create_time=None,
    )


@pytest.fixture
async def client():
    async def _override_user():
        return _ctx(OWNER_ID)

    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        yield client
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def _override_log(monkeypatch, log):
    monkeypatch.setattr(evaluation.evaluation_service, "get_log", AsyncMock(return_value=log))


async def test_get_task_other_users_log_rejected(client, monkeypatch):
    """查询他人评估任务 → A0401（按不存在处理，不泄露资源存在性）"""
    _override_log(monkeypatch, _log(OTHER_ID))

    resp = await client.get("/api/v1/evaluation/1")

    assert resp.status_code == 400
    assert resp.json()["code"] == "A0401"


async def test_get_task_own_completed_returns_metrics(client, monkeypatch):
    """本人已完成任务 → 返回指标与耗时"""
    _override_log(monkeypatch, _log(OWNER_ID, result={"PSNR": 30.1, "SSIM": 0.9}))

    resp = await client.get("/api/v1/evaluation/1")

    body = resp.json()
    assert resp.status_code == 200
    assert body["data"]["status"] == LogStatus.COMPLETED.value
    assert body["data"]["metrics"] == {"PSNR": 30.1, "SSIM": 0.9}
    assert body["data"]["time"] == 3


async def test_get_task_own_failed_returns_error_message(client, monkeypatch):
    """本人失败任务 → 返回 errorMessage"""
    log = _log(OWNER_ID, status=LogStatus.FAILED.value)
    log.error_message = "图片下载失败"
    _override_log(monkeypatch, log)

    resp = await client.get("/api/v1/evaluation/1")

    body = resp.json()
    assert resp.status_code == 200
    assert body["data"]["status"] == LogStatus.FAILED.value
    assert body["data"]["errorMessage"] == "图片下载失败"


async def test_get_task_missing_log_rejected(client, monkeypatch):
    """不存在的评估任务 → A0401"""
    from app.core.code import ResultCode
    from app.core.exceptions import BusinessException

    async def raise_missing(db, log_id):
        raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评估任务不存在")

    monkeypatch.setattr(evaluation.evaluation_service, "get_log", raise_missing)

    resp = await client.get("/api/v1/evaluation/99999")

    assert resp.status_code == 400
    assert resp.json()["code"] == "A0401"


async def test_list_logs_scoped_to_current_user(client, monkeypatch):
    """评估日志列表以当前用户身份查询（服务层按 user_id 过滤）"""
    captured = {}

    async def fake_list_logs(db, user_id, algorithm_id=None, page=1, size=10):
        captured["user_id"] = user_id
        return [], 0

    monkeypatch.setattr(evaluation.evaluation_service, "list_logs", fake_list_logs)

    resp = await client.get("/api/v1/evaluation/logs", params={"pageNum": 1, "pageSize": 10})

    assert resp.status_code == 200
    assert captured["user_id"] == OWNER_ID


async def test_list_metrics_scoped_to_current_user(client, monkeypatch):
    """评估指标历史以当前用户身份查询"""
    captured = {}

    async def fake_list_completed(db, user_id, algorithm_id=None, page=1, size=10):
        captured["user_id"] = user_id
        return [], 0

    monkeypatch.setattr(
        evaluation.evaluation_service, "list_completed_metrics", fake_list_completed
    )

    resp = await client.get("/api/v1/evaluation/metrics", params={"pageNum": 1, "pageSize": 10})

    assert resp.status_code == 200
    assert captured["user_id"] == OWNER_ID
