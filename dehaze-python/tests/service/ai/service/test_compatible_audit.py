from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.repository.mongo_ai_call_log_repository import MongoAiCallLogRepository
from app.service.ai.service import compatible_audit
from app.service.ai.service.compatible_audit import record_call

INSERT_DEFAULTS = {
    "user_id": 1,
    "key_id": 2,
    "key_prefix": "dhak_ab3x",
    "conversation_id": None,
    "model": "gpt-4",
    "endpoint": "chat/completions",
    "protocol": "openai",
    "is_stream": True,
    "status_code": 200,
}


def _capture_task(monkeypatch):
    created = []

    def _fake_create_task(coro):
        created.append(coro)
        return MagicMock()

    monkeypatch.setattr("asyncio.create_task", _fake_create_task)
    return created


async def _run_insert_async(monkeypatch, repo, **kwargs):
    created = _capture_task(monkeypatch)
    repo.insert_async(**kwargs)
    assert len(created) == 1
    await created[0]


class TestInsert:
    async def test_insert_defaults_applied(self, mongo_db, monkeypatch):
        repo = MongoAiCallLogRepository()
        await _run_insert_async(monkeypatch, repo, **INSERT_DEFAULTS, request_id="req-1")
        doc = await mongo_db["ai_api_call_log"].find_one({"request_id": "req-1"})
        assert doc is not None
        assert doc["input_tokens"] == 0
        assert doc["output_tokens"] == 0
        assert doc["credits"] is None
        assert doc["error_msg"] is None
        assert doc["conversation_id"] is None
        assert doc["key_prefix"] == "dhak_ab3x"
        # create_time 由 insert_async 自动补全（BSON 读取为 naive UTC）
        now_naive = datetime.now(UTC).replace(tzinfo=None)
        assert 0 <= (now_naive - doc["create_time"]).total_seconds() < 60

    async def test_insert_async_write_failure_not_raised(self, monkeypatch):
        collection = MagicMock()
        collection.insert_one = AsyncMock(side_effect=RuntimeError("mongo down"))
        client = MagicMock()
        client.__getitem__.return_value = {"ai_api_call_log": collection}
        monkeypatch.setattr("app.dependencies.mongo.get_mongo_client", lambda: client)
        repo = MongoAiCallLogRepository()
        await _run_insert_async(monkeypatch, repo, **INSERT_DEFAULTS, request_id="req-x")
        collection.insert_one.assert_awaited_once()


class TestQuery:
    @staticmethod
    async def _seed(db, docs):
        await db["ai_api_call_log"].insert_many(docs)

    async def test_query_pagination_and_user_filter(self, mongo_db):
        await self._seed(
            mongo_db,
            [
                {"user_id": 1, "model": "gpt-4", "create_time": datetime(2026, 1, 1, tzinfo=UTC)},
                {"user_id": 1, "model": "gpt-4", "create_time": datetime(2026, 1, 2, tzinfo=UTC)},
                {"user_id": 2, "model": "gpt-4", "create_time": datetime(2026, 1, 3, tzinfo=UTC)},
            ],
        )
        repo = MongoAiCallLogRepository()
        records, total = await repo.query(user_id=1, page=1, size=20)
        assert total == 2
        assert len(records) == 2
        assert all(r["user_id"] == 1 for r in records)
        # 按创建时间倒序
        assert records[0]["create_time"] > records[1]["create_time"]

    async def test_query_with_key_model_time_filters_and_pagination(self, mongo_db):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
        # 12 条命中（page=2/size=10 应返回第 11-12 条），外加各 1 条 key/model/时间不命中的干扰记录
        docs = [
            {
                "user_id": 1, "key_id": 5, "model": "gpt-4",
                "create_time": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            }
            for _ in range(12)
        ]
        docs += [
            {"user_id": 1, "key_id": 5, "model": "gpt-3.5",
             "create_time": datetime(2026, 1, 1, 12, 0, tzinfo=UTC)},
            {"user_id": 1, "key_id": 6, "model": "gpt-4",
             "create_time": datetime(2026, 1, 1, 12, 0, tzinfo=UTC)},
            {"user_id": 1, "key_id": 5, "model": "gpt-4",
             "create_time": datetime(2026, 1, 5, 12, 0, tzinfo=UTC)},
        ]
        await self._seed(mongo_db, docs)
        repo = MongoAiCallLogRepository()
        records, total = await repo.query(
            user_id=1, key_id=5, model="gpt-4", start_time=start, end_time=end, page=2, size=10
        )
        assert total == 12
        assert len(records) == 2
        assert all(r["key_id"] == 5 and r["model"] == "gpt-4" for r in records)

    async def test_query_no_time_filters(self, mongo_db):
        await self._seed(
            mongo_db,
            [
                {"user_id": 1, "model": "gpt-4", "create_time": datetime(2026, 1, 1, tzinfo=UTC)},
                {"user_id": 1, "model": "gpt-4", "create_time": datetime(2027, 6, 1, tzinfo=UTC)},
            ],
        )
        repo = MongoAiCallLogRepository()
        records, total = await repo.query(user_id=1)
        assert total == 2
        assert len(records) == 2


class TestRecordCall:
    @staticmethod
    def _stub_repository(monkeypatch):
        captured = {}
        mock_repo = MagicMock()
        mock_repo.insert_async.side_effect = lambda **kwargs: captured.update(kwargs)
        monkeypatch.setattr(compatible_audit, "mongo_ai_call_log_repository", mock_repo)
        return captured

    def test_request_id_auto_generated(self, monkeypatch):
        captured = self._stub_repository(monkeypatch)
        record_call(**INSERT_DEFAULTS)
        assert isinstance(captured["request_id"], str) and captured["request_id"]

    def test_request_id_passed_through(self, monkeypatch):
        captured = self._stub_repository(monkeypatch)
        record_call(**INSERT_DEFAULTS, request_id="fixed-req")
        assert captured["request_id"] == "fixed-req"

    def test_401_record_masked_prefix(self, monkeypatch):
        captured = self._stub_repository(monkeypatch)
        record_call(
            user_id=None,
            key_id=None,
            key_prefix="dhak_ab3x",
            conversation_id=None,
            model=None,
            endpoint="chat/completions",
            protocol="openai",
            is_stream=False,
            status_code=401,
            error_msg="未授权",
        )
        assert captured["key_prefix"] == "dhak_ab3x"
        assert captured["user_id"] is None
        assert captured["key_id"] is None
        assert captured["status_code"] == 401
        assert captured["error_msg"] == "未授权"

    def test_record_call_returns_none_and_no_side_effect(self, monkeypatch):
        captured = self._stub_repository(monkeypatch)
        assert record_call(**INSERT_DEFAULTS) is None
        assert "request_id" in captured
