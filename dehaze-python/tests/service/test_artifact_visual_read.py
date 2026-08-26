import asyncio
from types import SimpleNamespace

import pytest

import app.service.ai_artifact_service as mod
import app.service.file_service as fs
from app.core.exceptions import BusinessException
from app.repository.ai_artifact_repository import ai_artifact_repository
from app.service.ai_artifact_service import ai_artifact_service
from tests.stubs.fakes import LLMChunk, MemberBenefitRepo
from tests.stubs.factories import make_benefit, make_member


def _make_artifact(**overrides):
    fields = {
        "id": 1,
        "conversation_id": 1,
        "message_id": 1,
        "type": "image_result",
        "ref_type": "sys_file",
        "ref_id": 10,
        "summary": {"algorithm": "RIDCP"},
        "is_invalid": 0,
        "create_time": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _quota_repo(level_code="level_0", limit=0, member_exists=True):
    member = make_member(level_code) if member_exists else None
    benefit = make_benefit(multimodal_limit=limit)
    repo = MemberBenefitRepo(member, benefit)
    return repo


def _make_svc(artifact_repo=None, conv_repo=None, quota=None):
    kw = {}
    if artifact_repo is not None:
        kw["ai_artifact_repository"] = artifact_repo
    if conv_repo is not None:
        kw["ai_conversation_repository"] = conv_repo
    if quota is not None:
        kw["member_repository"] = quota
        kw["member_benefit_repository"] = quota
    return mod.AiArtifactService(**kw)


def _visual_svc(
    artifact,
    level_code="level_1",
    limit=10,
    conv_exists=True,
    image_url="http://cdn.example.com/pred.jpg",
    model_id="vlm-1",
):
    class _ArtifactRepo:
        async def get_by_id(self, db, artifact_id):
            return artifact

    class _ConvRepo:
        async def get_by_id_and_user(self, db, conv_id, user_id):
            return object() if conv_exists else None

    svc = _make_svc(
        artifact_repo=_ArtifactRepo(),
        conv_repo=_ConvRepo(),
        quota=_quota_repo(level_code=level_code, limit=limit),
    )

    async def _resolve(db, artifact):
        return image_url

    async def _pick(db, model_id):
        return model_id or "vlm-1"

    svc._resolve_image_url = staticmethod(_resolve)
    svc._pick_multimodal_model = staticmethod(_pick)
    return svc


def _freeze_quota_key(svc, user_id=1):
    key = f"ai:multimodal:{user_id}:20260101"
    svc._visual_quota_key = staticmethod(lambda uid: key)
    return key


async def test_quota_limit_per_level(mock_redis):
    benefits = {"level_0": 5, "level_1": 10, "level_2": 20, "level_3": 50}
    for level, limit in benefits.items():
        svc = _make_svc(quota=_quota_repo(level_code=level, limit=limit))
        used, actual = await svc.check_visual_quota(None, mock_redis, 1)
        assert actual == limit
        assert used == 0


async def test_quota_consumption_rejects_at_limit(mock_redis):
    key = _freeze_quota_key(ai_artifact_service)
    await mock_redis.set(key, "5")
    ok = await ai_artifact_service._consume_visual_quota(mock_redis, 1, limit=5)
    assert ok is False
    assert int(await mock_redis.get(key)) == 5

    await mock_redis.set(key, "4")
    ok = await ai_artifact_service._consume_visual_quota(mock_redis, 1, limit=5)
    assert ok is True
    assert int(await mock_redis.get(key)) == 5


async def test_quota_count_accumulates_across_calls(mock_redis):
    svc = _make_svc(quota=_quota_repo(level_code="level_3", limit=50))
    key = _freeze_quota_key(svc)
    oks = [await svc._consume_visual_quota(mock_redis, 1, limit=50) for _ in range(3)]
    assert oks == [True, True, True]
    used, limit = await svc.check_visual_quota(None, mock_redis, 1)
    assert used == 3
    assert limit == 50


async def test_quota_consumption_sets_midnight_ttl(mock_redis):
    key = ai_artifact_service._visual_quota_key(7)
    ok = await ai_artifact_service._consume_visual_quota(mock_redis, 7, limit=50)
    assert ok is True
    assert int(await mock_redis.get(key)) == 1
    assert await mock_redis.ttl(key) > 0


async def test_quota_concurrent_cannot_bypass(mock_redis):
    svc = _make_svc()
    key = _freeze_quota_key(svc)

    async def _consume():
        return await svc._consume_visual_quota(mock_redis, 1, limit=2)

    results = await asyncio.gather(*[_consume() for _ in range(5)])
    assert results.count(True) == 2
    assert results.count(False) == 3
    assert int(await mock_redis.get(key)) == 2


async def test_quota_missing_member_defaults_to_level0(mock_redis):
    svc = _make_svc(quota=_quota_repo(level_code="level_0", limit=5, member_exists=False))
    used, limit = await svc.check_visual_quota(None, mock_redis, 99)
    assert used == 0
    assert limit == 5


async def test_visual_read_success(monkeypatch, mock_redis):
    svc = _visual_svc(_make_artifact(id=5))

    captured_usage = {"input_tokens": 1200}

    async def _fake_stream_chat(db, model_id, messages, **kwargs):
        assert model_id == "vlm-1"
        assert messages[0]["content"][1]["type"] == "image_url"
        assert messages[0]["content"][1]["image_url"]["url"] == "http://cdn.example.com/pred.jpg"
        yield LLMChunk(type="text_delta", content="图像")
        yield LLMChunk(type="text_delta", content="清晰")
        yield LLMChunk(type="done", content="", usage=captured_usage)

    monkeypatch.setattr(mod.llm_client, "stream_chat", _fake_stream_chat)

    text, input_tokens = await svc.visual_read(None, mock_redis, 1, artifact_id=5)
    assert text == "图像清晰"
    assert input_tokens == 1200
    keys = [k async for k in mock_redis.scan_iter(match="ai:multimodal:1:*")]
    assert keys and int(await mock_redis.get(keys[0])) == 1


async def test_visual_read_quota_exceeded_returns_degraded(monkeypatch, mock_redis):
    svc = _visual_svc(_make_artifact(id=5, summary={"psnr": 31.2}), level_code="level_0", limit=5)
    key = _freeze_quota_key(svc)
    await mock_redis.set(key, "5")

    called = {}

    async def _fake_stream_chat(*args, **kwargs):
        called["called"] = True

    monkeypatch.setattr(mod.llm_client, "stream_chat", _fake_stream_chat)

    text, input_tokens = await svc.visual_read(None, mock_redis, 1, artifact_id=5)
    assert text.startswith("视觉读取已达今日上限，基于指标判断：")
    assert "31.2" in text
    assert input_tokens == 0
    assert "called" not in called


async def test_visual_read_invalid_artifact_raises(mock_redis):
    class _ArtifactRepo:
        async def get_by_id(self, db, artifact_id):
            return None

    svc = _make_svc(artifact_repo=_ArtifactRepo())

    with pytest.raises(BusinessException, match="产物不存在或已失效"):
        await svc.visual_read(None, mock_redis, 1, artifact_id=99)


async def test_mark_invalid_for_file_direct_and_indirect(monkeypatch):
    marked = []

    class _ArtifactRepo:
        async def mark_invalid(self, db, ref_type, ref_id):
            marked.append((ref_type, ref_id))

    class _PredRepo:
        async def list_ids_by_file(self, db, file_id):
            return [11, 12]

    class _EvalRepo:
        async def list_ids_by_file(self, db, file_id):
            return [21]

    svc = _make_svc(artifact_repo=_ArtifactRepo())
    monkeypatch.setattr(mod, "pred_log_repository", _PredRepo())
    monkeypatch.setattr(mod, "eval_log_repository", _EvalRepo())

    await svc.mark_invalid_for_file(None, file_id=10)
    assert marked == [
        ("sys_file", 10),
        ("sys_pred_log", 11),
        ("sys_pred_log", 12),
        ("sys_eval_log", 21),
    ]


async def test_get_message_artifact_refs_grouped(monkeypatch):
    artifacts = [
        _make_artifact(id=1, message_id=10, type="image_result"),
        _make_artifact(id=2, message_id=10, type="metric_report", summary={"psnr": 30}),
        _make_artifact(id=3, message_id=11, type="file_ref", summary=None),
    ]

    class _ArtifactRepo:
        async def list_by_message_ids(self, db, message_ids):
            return [a for a in artifacts if a.message_id in message_ids]

    svc = _make_svc(artifact_repo=_ArtifactRepo())

    result = await svc.get_message_artifact_refs(None, [10, 11, 12])
    assert set(result.keys()) == {10, 11}
    assert result[10] == [
        {"id": 1, "type": "image_result", "summary": {"algorithm": "RIDCP"}},
        {"id": 2, "type": "metric_report", "summary": {"psnr": 30}},
    ]
    assert result[11] == [{"id": 3, "type": "file_ref", "summary": None}]


async def test_list_by_message_ids_sql_filters_invalid():
    captured = {}

    class _Rows:
        def scalars(self):
            return self

        def all(self):
            return []

    class _DB:
        async def execute(self, stmt):
            captured["stmt"] = stmt
            return _Rows()

    await ai_artifact_repository.list_by_message_ids(_DB(), [10, 11, 12])
    sql = str(captured["stmt"].compile(compile_kwargs={"literal_binds": True}))
    assert "is_invalid = 0" in sql
    assert "message_id IN (10, 11, 12)" in sql


async def test_get_detail_ownership_check(monkeypatch):
    class _ArtifactRepo:
        async def get_by_id(self, db, artifact_id):
            return _make_artifact(id=artifact_id, conversation_id=1)

    class _ConvRepo:
        async def get_by_id_and_user(self, db, conv_id, user_id):
            return None

    svc = _make_svc(artifact_repo=_ArtifactRepo(), conv_repo=_ConvRepo())

    with pytest.raises(BusinessException, match="产物所属会话不存在"):
        await svc.get_detail(None, artifact_id=1, user_id=999)


async def test_list_by_ref_filters_owned(monkeypatch):
    artifacts = [
        _make_artifact(id=1, conversation_id=1, ref_type="sys_file", ref_id=10),
        _make_artifact(id=2, conversation_id=2, ref_type="sys_file", ref_id=10),
    ]

    class _ArtifactRepo:
        async def list_by_ref(self, db, ref_type, ref_id):
            return artifacts

    class _ConvRepo:
        async def get_by_id_and_user(self, db, conv_id, user_id):
            return object() if conv_id == 1 else None

    svc = _make_svc(artifact_repo=_ArtifactRepo(), conv_repo=_ConvRepo())

    result = await svc.list_by_ref(None, "sys_file", 10, user_id=5)
    assert len(result) == 1
    assert result[0].id == 1


async def test_metric_report_registration(monkeypatch):
    created = {}

    class _ArtifactRepo:
        async def create(self, db, entity):
            entity.is_invalid = 0
            entity.id = 42
            created["entity"] = entity
            return entity

    svc = _make_svc(artifact_repo=_ArtifactRepo())

    metrics = {"psnr": 32.1, "ssim": 0.91}
    result = await svc.register_artifact(
        None,
        conv_id=1,
        msg_id=2,
        artifact_type="metric_report",
        ref_type="sys_eval_log",
        ref_id=7,
        summary=metrics,
    )
    assert result.id == 42
    entity = created["entity"]
    assert entity.type == "metric_report"
    assert entity.ref_type == "sys_eval_log"
    assert entity.ref_id == 7
    assert entity.summary == metrics


def test_artifact_query_routes_registered(app):
    paths = app.openapi()["paths"]

    by_ref = paths.get("/api/v1/ai/artifacts/by-ref")
    assert by_ref is not None
    assert "get" in by_ref
    params = {p["name"]: p for p in by_ref["get"]["parameters"]}
    assert params["refType"]["in"] == "query"
    assert params["refId"]["in"] == "query"

    detail = paths.get("/api/v1/ai/artifacts/{artifact_id}/detail")
    assert detail is not None
    assert "get" in detail


async def test_file_service_delete_hooks_invalid_for_file(monkeypatch, tmp_path):
    from app.service.storage.local_storage import LocalStorageService

    called = {}

    class _FileRepo:
        async def get_by_id(self, db, file_id):
            return SimpleNamespace(object_name="pred/20260101/x.jpg", storage="local")

        async def soft_delete_by_ids(self, db, ids):
            return len(ids)

    monkeypatch.setattr(fs, "file_repository", _FileRepo())

    async def _fake_invalidate(db, file_id):
        called["file_id"] = file_id

    monkeypatch.setattr(ai_artifact_service, "mark_invalid_for_file", staticmethod(_fake_invalidate))

    # 注入真实本地存储（tmp_path 真实文件读写），断言物理文件被删除
    storage = LocalStorageService(base_dir=str(tmp_path))
    storage.upload("dehaze", "pred/20260101/x.jpg", b"file-content", "image/jpeg")
    assert storage.exists("dehaze", "pred/20260101/x.jpg") is True
    monkeypatch.setattr(fs, "get_storage_by_name", lambda name: storage)

    await fs.file_service.delete_file_with_storage(None, file_id=10)
    assert called.get("file_id") == 10
    assert storage.exists("dehaze", "pred/20260101/x.jpg") is False
