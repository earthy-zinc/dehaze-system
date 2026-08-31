"""评测中心路由测试：路径注册 / 权限校验 / 参数校验 / 响应 camelCase 序列化。"""
import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.api

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.main import app as fastapi_app
from app.router import ai_agent_eval as eval_module
from app.service.ai_eval_center_service import eval_center_service


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


@pytest.fixture
async def eval_center_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser(permissions=["ai:agent:manage"])}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_eval_center_paths_registered(app):
    schema = app.openapi()
    for path in (
        "/api/v1/ai/eval-center/overview",
        "/api/v1/ai/eval-center/trends",
        "/api/v1/ai/eval-center/runs/{run_id}/compare",
        "/api/v1/ai/eval-center/judge-status",
        "/api/v1/ai/eval-center/reviews",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


class TestPermissions:
    async def test_overview_forbidden_without_permission(self, eval_center_client):
        client, state = eval_center_client
        state["user"] = _FakeUser(permissions=[])
        resp = await client.get("/api/v1/ai/eval-center/overview")
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"

    async def test_reviews_submit_forbidden_without_permission(self, eval_center_client):
        client, state = eval_center_client
        state["user"] = _FakeUser(permissions=[])
        resp = await client.post(
            "/api/v1/ai/eval-center/reviews/1", json={"agree": True}
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"


class TestEndpoints:
    async def test_overview_wire_and_camel_case(self, eval_center_client, monkeypatch):
        client, _ = eval_center_client

        async def _fake_overview(db):
            return [
                {
                    "agent_id": 1,
                    "agent_code": "a1",
                    "agent_name": "A1",
                    "run_id": 10,
                    "run_time": "2026-08-29T10:00:00",
                    "trigger_type": "manual",
                    "gate_status": "passed",
                    "total_score": 85.0,
                    "dimensions": {"result_quality": 80},
                    "degraded": False,
                    "high_risk_failed": True,
                }
            ]

        monkeypatch.setattr(eval_center_service, "overview", _fake_overview)
        resp = await client.get("/api/v1/ai/eval-center/overview")
        assert resp.status_code == 200
        item = resp.json()["data"][0]
        assert item["agentId"] == 1
        assert item["agentName"] == "A1"
        assert item["gateStatus"] == "passed"
        assert item["totalScore"] == 85.0
        assert item["highRiskFailed"] is True

    async def test_judge_status_wire(self, eval_center_client, monkeypatch):
        client, _ = eval_center_client

        async def _fake_judge_status(db):
            return {
                "consistency_state": "drifted",
                "drift_paused": True,
                "consistency_threshold": 90,
                "review_stats": {
                    "total": 2,
                    "pending": 0,
                    "reviewed": 2,
                    "agree_count": 1,
                    "disagree_count": 1,
                    "agreement_rate": 50.0,
                },
            }

        monkeypatch.setattr(eval_center_service, "judge_status", _fake_judge_status)
        resp = await client.get("/api/v1/ai/eval-center/judge-status")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["consistencyState"] == "drifted"
        assert data["driftPaused"] is True
        assert data["reviewStats"]["agreementRate"] == 50.0

    async def test_trends_filters_forwarded(self, eval_center_client, monkeypatch):
        client, _ = eval_center_client
        captured: dict = {}

        async def _fake_trends(db, agent_id=None, start_time=None, end_time=None, limit=100):
            captured.update(
                agent_id=agent_id, start_time=start_time, end_time=end_time, limit=limit
            )
            return []

        monkeypatch.setattr(eval_center_service, "trends", _fake_trends)
        resp = await client.get(
            "/api/v1/ai/eval-center/trends",
            params={"agentId": 3, "startTime": "2026-08-01T00:00:00", "limit": 50},
        )
        assert resp.status_code == 200
        assert captured["agent_id"] == 3
        assert captured["limit"] == 50
        assert captured["start_time"] is not None

    async def test_compare_missing_base_run_id_rejected(self, eval_center_client):
        client, _ = eval_center_client
        resp = await client.get("/api/v1/ai/eval-center/runs/1/compare")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_compare_wire(self, eval_center_client, monkeypatch):
        client, _ = eval_center_client
        captured: dict = {}

        async def _fake_compare(db, run_id, base_run_id):
            captured.update(run_id=run_id, base_run_id=base_run_id)
            return {
                "run_id": 2,
                "base_run_id": 1,
                "agent_id": 5,
                "current": {
                    "run_id": 2,
                    "total_score": 80.0,
                    "dimensions": {},
                    "sample_count": 2,
                    "pass_rate": 0.5,
                    "create_time": None,
                },
                "base": {
                    "run_id": 1,
                    "total_score": 90.0,
                    "dimensions": {},
                    "sample_count": 2,
                    "pass_rate": 1.0,
                    "create_time": None,
                },
                "dimension_diff": {"result_quality": -10.0},
                "sample_diff": {"added": [], "removed": [], "changed": [], "unchanged_count": 1},
            }

        monkeypatch.setattr(eval_center_service, "compare_runs", _fake_compare)
        resp = await client.get(
            "/api/v1/ai/eval-center/runs/2/compare", params={"baseRunId": 1}
        )
        assert resp.status_code == 200
        assert captured == {"run_id": 2, "base_run_id": 1}
        data = resp.json()["data"]
        assert data["baseRunId"] == 1
        assert data["dimensionDiff"]["result_quality"] == -10.0
        assert data["sampleDiff"]["unchangedCount"] == 1

    async def test_reviews_wire(self, eval_center_client, monkeypatch):
        client, _ = eval_center_client

        async def _fake_list_reviews(db, status=None):
            return {
                "items": [
                    {
                        "id": 1,
                        "run_id": 10,
                        "sample_id": 20,
                        "agent_id": 5,
                        "agent_name": "A5",
                        "judge_passed": False,
                        "risk_level": "high",
                        "status": 1,
                        "agree": None,
                        "remark": None,
                        "create_time": "2026-08-29T10:00:00",
                    }
                ],
                "pending": 1,
                "reviewed": 0,
            }

        monkeypatch.setattr(eval_center_service, "list_reviews", _fake_list_reviews)
        resp = await client.get("/api/v1/ai/eval-center/reviews", params={"status": 1})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["pending"] == 1
        assert data["items"][0]["judgePassed"] is False
        assert data["items"][0]["agentName"] == "A5"

    async def test_review_submit_wire(self, eval_center_client, monkeypatch):
        client, _ = eval_center_client
        captured: dict = {}

        async def _fake_submit(db, review_id, agree, remark, reviewer_id):
            captured.update(
                review_id=review_id, agree=agree, remark=remark, reviewer_id=reviewer_id
            )
            return {
                "id": review_id,
                "run_id": 10,
                "sample_id": 20,
                "agent_id": 5,
                "judge_passed": False,
                "risk_level": "high",
                "status": 2,
                "agree": agree,
                "remark": remark,
            }

        monkeypatch.setattr(eval_center_service, "submit_review", _fake_submit)
        resp = await client.post(
            "/api/v1/ai/eval-center/reviews/1",
            json={"agree": False, "remark": "判分有误"},
        )
        assert resp.status_code == 200
        assert captured == {
            "review_id": 1,
            "agree": False,
            "remark": "判分有误",
            "reviewer_id": 1,
        }
        assert resp.json()["data"]["status"] == 2

    async def test_review_submit_remark_too_long_rejected(self, eval_center_client):
        client, _ = eval_center_client
        resp = await client.post(
            "/api/v1/ai/eval-center/reviews/1",
            json={"agree": True, "remark": "x" * 501},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"
