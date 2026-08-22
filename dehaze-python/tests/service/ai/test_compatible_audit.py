from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.repository.mongo_ai_call_log_repository import MongoAiCallLogRepository
from app.service.ai import compatible_audit
from app.service.ai.compatible_audit import record_call

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


def _mock_mongo_client(insert_result=None):
    if insert_result is None:
        insert_result = MagicMock(inserted_id="507f1f77bcf86cd799439011")
    collection = MagicMock()
    collection.insert_one = AsyncMock(return_value=insert_result)
    client = MagicMock()
    client.__getitem__.return_value = {"ai_api_call_log": collection}
    return client, collection


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
    async def test_insert_returns_id_string(self, monkeypatch):
        client, collection = _mock_mongo_client()
        monkeypatch.setattr(
            "app.repository.mongo_ai_call_log_repository.get_mongo_client", lambda: client
        )
        repo = MongoAiCallLogRepository()
        _id = await repo.insert({**INSERT_DEFAULTS, "create_time": datetime.now(UTC)})
        assert _id == "507f1f77bcf86cd799439011"
        collection.insert_one.assert_awaited_once()

    async def test_insert_async_field_completeness_and_defaults(self, monkeypatch):
        client, collection = _mock_mongo_client()
        monkeypatch.setattr(
            "app.repository.mongo_ai_call_log_repository.get_mongo_client", lambda: client
        )
        repo = MongoAiCallLogRepository()
        await _run_insert_async(
            monkeypatch,
            repo,
            user_id=1,
            key_id=2,
            key_prefix="dhak_ab3x",
            endpoint="chat/completions",
            protocol="openai",
            is_stream=True,
            status_code=200,
            error_msg=None,
            input_tokens=None,
            output_tokens=None,
            credits=None,
            conversation_id=None,
            request_id="req-1",
        )
        inserted = collection.insert_one.await_args.args[0]
        assert inserted["input_tokens"] == 0
        assert inserted["output_tokens"] == 0
        assert inserted["credits"] is None
        assert inserted["error_msg"] is None
        assert inserted["conversation_id"] is None
        assert inserted["request_id"] == "req-1"
        assert inserted["key_prefix"] == "dhak_ab3x"
        assert inserted["create_time"].tzinfo == UTC

    async def test_insert_async_write_failure_not_raised(self, monkeypatch):
        client, collection = _mock_mongo_client()
        collection.insert_one = AsyncMock(side_effect=RuntimeError("mongo down"))
        monkeypatch.setattr(
            "app.repository.mongo_ai_call_log_repository.get_mongo_client", lambda: client
        )
        repo = MongoAiCallLogRepository()
        await _run_insert_async(monkeypatch, repo, **INSERT_DEFAULTS, request_id="req-x")
        collection.insert_one.assert_awaited_once()


class TestQuery:
    def _build_repo(self, monkeypatch, records, total):
        def _iter(*_args):
            async def _agen():
                for r in records:
                    yield r

            return _agen()

        collection = MagicMock()
        collection.find = MagicMock(return_value=collection)
        collection.sort = MagicMock(return_value=collection)
        collection.skip = MagicMock(return_value=collection)
        collection.limit = MagicMock(return_value=collection)
        collection.__aiter__ = _iter
        collection.count_documents = AsyncMock(return_value=total)
        client = MagicMock()
        client.__getitem__.return_value = {"ai_api_call_log": collection}
        monkeypatch.setattr(
            "app.repository.mongo_ai_call_log_repository.get_mongo_client", lambda: client
        )
        return MongoAiCallLogRepository(), collection

    async def test_query_pagination_and_user_filter(self, monkeypatch):
        repo, collection = self._build_repo(
            monkeypatch,
            records=[{"_id": "a", "user_id": 1}, {"_id": "b", "user_id": 1}],
            total=2,
        )
        records, total = await repo.query(user_id=1, page=1, size=20)
        assert total == 2
        assert len(records) == 2
        assert records[0]["_id"] == "a"
        filter_cond = collection.find.call_args.args[0]
        assert filter_cond == {"user_id": 1}
        collection.sort.assert_called_once_with("create_time", -1)
        collection.skip.assert_called_once_with(0)
        collection.limit.assert_called_once_with(20)

    async def test_query_with_key_model_time_filters(self, monkeypatch):
        repo, collection = self._build_repo(
            monkeypatch, records=[{"_id": "a", "user_id": 1}], total=1
        )
        start = datetime(2026, 1, 1, tzinfo=UTC)
        end = datetime(2026, 1, 2, tzinfo=UTC)
        await repo.query(
            user_id=1, key_id=5, model="gpt-4", start_time=start, end_time=end, page=2, size=10
        )
        filter_cond = collection.find.call_args.args[0]
        assert filter_cond == {
            "user_id": 1,
            "key_id": 5,
            "model": "gpt-4",
            "create_time": {"$gte": start, "$lte": end},
        }
        collection.skip.assert_called_once_with(10)

    async def test_query_no_time_filters(self, monkeypatch):
        repo, collection = self._build_repo(monkeypatch, records=[], total=0)
        await repo.query(user_id=1)
        filter_cond = collection.find.call_args.args[0]
        assert filter_cond == {"user_id": 1}
        assert "create_time" not in filter_cond


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
