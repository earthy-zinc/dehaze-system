"""智能体评测路由测试（/api/v1/ai/agents/{agent_id}/eval）

覆盖重点：路由注册、ai:agent:manage 权限拦截（A0301）、参数校验（A0400）、
评测集/样本 CRUD、手动触发评测（trigger_type=manual）、执行记录分页与 camelCase 序列化。
"""
from datetime import datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.dependencies.redis import get_redis_client as _ORIGINAL_GET_REDIS_CLIENT
from app.main import app as fastapi_app
from app.service.ai_eval_service import eval_service

MANAGE_PERM = "ai:agent:manage"


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=(MANAGE_PERM,)):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _dataset(**overrides):
    base = {
        "id": 5,
        "agent_id": 2,
        "name": "回归集",
        "description": "发布门禁用",
        "dataset_type": "regression",
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _sample(**overrides):
    base = {
        "id": 11,
        "dataset_id": 5,
        "task_goal": "为用户推荐去雾算法",
        "allowed_input": "一张雾图",
        "tools": ["algorithm_recommend"],
        "expected_process": None,
        "expected_result": "给出算法建议",
        "forbidden_behavior": None,
        "risk_level": "low",
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(**overrides):
    base = {
        "id": 30,
        "agent_id": 2,
        "dataset_id": 5,
        "trigger_type": "manual",
        "status": 2,
        "score_summary": {"result_quality": 88.0},
        "results": [],
        "create_by": 8,
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
async def eval_client():
    async def _override_db():
        return object()

    async def _override_redis():
        return _REDIS_STUB

    current_user = {"user": _FakeUser(id=8)}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    fastapi_app.dependency_overrides[_ORIGINAL_GET_REDIS_CLIENT] = _override_redis
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)
    fastapi_app.dependency_overrides.pop(_ORIGINAL_GET_REDIS_CLIENT, None)


_REDIS_STUB = object()
# OpenAPI 中的路径模板（路径参数占位），实际请求用 _EVAL
_EVAL_TPL = "/api/v1/ai/agents/{agent_id}/eval"
_EVAL = "/api/v1/ai/agents/2/eval"


def test_eval_paths_registered(app):
    schema = app.openapi()
    for path in (
        f"{_EVAL_TPL}/datasets",
        f"{_EVAL_TPL}/datasets/{{dataset_id}}",
        f"{_EVAL_TPL}/datasets/{{dataset_id}}/samples",
        f"{_EVAL_TPL}/samples/{{sample_id}}",
        f"{_EVAL_TPL}/runs",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


_PERMISSION_CASES = [
    ("post", f"{_EVAL}/datasets", {"name": "d", "dataset_type": "dev"}),
    ("get", f"{_EVAL}/datasets", None),
    ("patch", f"{_EVAL}/datasets/5", {"name": "d"}),
    ("delete", f"{_EVAL}/datasets/5", None),
    ("post", f"{_EVAL}/datasets/5/samples", {"dataset_id": 5, "task_goal": "t"}),
    ("get", f"{_EVAL}/datasets/5/samples", None),
    ("patch", f"{_EVAL}/samples/11", {"task_goal": "t"}),
    ("delete", f"{_EVAL}/samples/11", None),
    ("post", f"{_EVAL}/runs", None),
    ("get", f"{_EVAL}/runs", None),
]


@pytest.mark.parametrize(("method", "path", "body"), _PERMISSION_CASES)
async def test_endpoints_forbidden_without_permission(method, path, body, eval_client):
    client, state = eval_client
    state["user"] = _FakeUser(id=8, permissions=[])
    resp = await client.request(method, path, json=body)
    assert resp.status_code == 403
    assert resp.json()["code"] == "A0301"


class TestDatasets:
    async def test_create_dataset(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_create(db, agent_id, form):
            captured.update(agent_id=agent_id, name=form.name, dataset_type=form.dataset_type)
            return _dataset()

        monkeypatch.setattr(eval_service, "create_dataset", _fake_create)
        resp = await client.post(
            f"{_EVAL}/datasets",
            json={"name": "回归集", "description": "门禁", "dataset_type": "regression"},
        )
        assert resp.status_code == 200
        assert captured == {"agent_id": 2, "name": "回归集", "dataset_type": "regression"}
        data = resp.json()["data"]
        assert data["agentId"] == 2
        assert data["datasetType"] == "regression"

    async def test_create_dataset_invalid_type_rejected(self, eval_client):
        client, _ = eval_client
        resp = await client.post(
            f"{_EVAL}/datasets", json={"name": "回归集", "dataset_type": "unknown"}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_dataset_name_too_long_rejected(self, eval_client):
        client, _ = eval_client
        resp = await client.post(
            f"{_EVAL}/datasets", json={"name": "x" * 129, "dataset_type": "dev"}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_dataset_duplicate_maps_a0501(self, eval_client, monkeypatch):
        client, _ = eval_client

        async def _fake_create(db, agent_id, form):
            raise BusinessException(ResultCode.DATA_EXISTS, "该 Agent 已存在同类型评测集")

        monkeypatch.setattr(eval_service, "create_dataset", _fake_create)
        resp = await client.post(
            f"{_EVAL}/datasets", json={"name": "回归集", "dataset_type": "regression"}
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0501"

    async def test_list_datasets(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_list(db, agent_id):
            captured["agent_id"] = agent_id
            return [_dataset(), _dataset(id=6, name="dev集", dataset_type="dev")]

        monkeypatch.setattr(eval_service, "list_datasets", _fake_list)
        resp = await client.get(f"{_EVAL}/datasets")
        assert resp.status_code == 200
        assert captured["agent_id"] == 2
        assert [item["id"] for item in resp.json()["data"]] == [5, 6]

    async def test_update_dataset(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_update(db, dataset_id, form):
            captured.update(dataset_id=dataset_id, name=form.name)
            return _dataset(name=form.name)

        monkeypatch.setattr(eval_service, "update_dataset", _fake_update)
        resp = await client.patch(f"{_EVAL}/datasets/5", json={"name": "改后名称"})
        assert resp.status_code == 200
        assert captured == {"dataset_id": 5, "name": "改后名称"}
        assert resp.json()["data"]["name"] == "改后名称"

    async def test_delete_dataset(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_delete(db, dataset_id):
            captured["dataset_id"] = dataset_id

        monkeypatch.setattr(eval_service, "delete_dataset", _fake_delete)
        resp = await client.delete(f"{_EVAL}/datasets/5")
        assert resp.status_code == 200
        assert captured == {"dataset_id": 5}

    async def test_delete_dataset_missing_maps_a0401(self, eval_client, monkeypatch):
        client, _ = eval_client

        async def _fake_delete(db, dataset_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测集不存在")

        monkeypatch.setattr(eval_service, "delete_dataset", _fake_delete)
        resp = await client.delete(f"{_EVAL}/datasets/404")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"


class TestSamples:
    async def test_create_sample(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_create(db, dataset_id, form):
            captured.update(dataset_id=dataset_id, task_goal=form.task_goal)
            return _sample()

        monkeypatch.setattr(eval_service, "create_sample", _fake_create)
        resp = await client.post(
            f"{_EVAL}/datasets/5/samples",
            json={
                "dataset_id": 5,
                "task_goal": "为用户推荐去雾算法",
                "tools": ["algorithm_recommend"],
                "risk_level": "high",
            },
        )
        assert resp.status_code == 200
        assert captured == {"dataset_id": 5, "task_goal": "为用户推荐去雾算法"}
        data = resp.json()["data"]
        assert data["datasetId"] == 5
        assert data["taskGoal"] == "为用户推荐去雾算法"

    async def test_create_sample_invalid_risk_level_rejected(self, eval_client):
        client, _ = eval_client
        resp = await client.post(
            f"{_EVAL}/datasets/5/samples",
            json={"dataset_id": 5, "task_goal": "t", "risk_level": "critical"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_create_sample_requires_task_goal(self, eval_client):
        client, _ = eval_client
        resp = await client.post(f"{_EVAL}/datasets/5/samples", json={"dataset_id": 5})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_list_samples(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_list(db, dataset_id):
            captured["dataset_id"] = dataset_id
            return [_sample()]

        monkeypatch.setattr(eval_service, "list_samples", _fake_list)
        resp = await client.get(f"{_EVAL}/datasets/5/samples")
        assert resp.status_code == 200
        assert captured == {"dataset_id": 5}
        assert resp.json()["data"][0]["riskLevel"] == "low"

    async def test_update_sample(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_update(db, sample_id, form):
            captured.update(sample_id=sample_id, risk_level=form.risk_level)
            return _sample(risk_level=form.risk_level)

        monkeypatch.setattr(eval_service, "update_sample", _fake_update)
        resp = await client.patch(f"{_EVAL}/samples/11", json={"risk_level": "medium"})
        assert resp.status_code == 200
        assert captured == {"sample_id": 11, "risk_level": "medium"}
        assert resp.json()["data"]["riskLevel"] == "medium"

    async def test_delete_sample(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_delete(db, sample_id):
            captured["sample_id"] = sample_id

        monkeypatch.setattr(eval_service, "delete_sample", _fake_delete)
        resp = await client.delete(f"{_EVAL}/samples/11")
        assert resp.status_code == 200
        assert captured == {"sample_id": 11}


class TestRuns:
    async def test_manual_trigger_forwards_manual_and_redis(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_run(db, redis, agent_id, trigger_type="publish"):
            captured.update(
                agent_id=agent_id, trigger_type=trigger_type, redis_is_stub=redis is _REDIS_STUB
            )
            return {"run_id": 30, "passed": True, "score_summary": None, "failed_samples": []}

        monkeypatch.setattr(eval_service, "run_regression", _fake_run)
        resp = await client.post(f"{_EVAL}/runs")
        assert resp.status_code == 200
        assert captured == {"agent_id": 2, "trigger_type": "manual", "redis_is_stub": True}
        assert resp.json()["data"] == {"runId": 30, "passed": True, "failedSamples": []}

    async def test_list_runs_forwards_dataset_filter(self, eval_client, monkeypatch):
        client, _ = eval_client
        captured: dict = {}

        async def _fake_list(db, agent_id, page, size, dataset_id=None):
            captured.update(agent_id=agent_id, page=page, size=size, dataset_id=dataset_id)
            return [_run()], 1

        monkeypatch.setattr(eval_service, "list_runs", _fake_list)
        resp = await client.get(
            f"{_EVAL}/runs", params={"datasetId": 5, "pageNum": 2, "pageSize": 5}
        )
        assert resp.status_code == 200
        assert captured == {"agent_id": 2, "page": 2, "size": 5, "dataset_id": 5}
        data = resp.json()["data"]
        assert data["total"] == 1
        assert data["list"][0]["triggerType"] == "manual"
        assert data["list"][0]["scoreSummary"] == {"result_quality": 88.0}

    async def test_list_runs_page_size_too_large_rejected(self, eval_client):
        client, _ = eval_client
        resp = await client.get(f"{_EVAL}/runs", params={"pageSize": 101})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"
