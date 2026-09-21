"""批量处理图片地址归属校验：LLM 传入的 URL 不得被服务端任意下载。"""

from types import SimpleNamespace

from app.service.ai.service import batch_process_service as bps
from tests.stubs.fakes import NullDBSession

_OWNER = 10
_OTHER = 11


def _patch(monkeypatch, owner_id=_OWNER, file_exists=True):
    base = bps.settings.FILE_STORAGE_BASE_URLS["minio"]
    calls = []
    events = []

    async def _get_by_object_name(db, object_name):
        if not file_exists:
            return None
        return SimpleNamespace(object_name=object_name, create_by=owner_id, deleted=0)

    async def _predict(**kwargs):
        calls.append(kwargs["image_url"])
        return {"logId": 7}

    async def _send(stream_session_id, event_type, data):
        events.append((event_type, data))

    async def _register(db, conv_id, msg_id, **kwargs):
        return SimpleNamespace(id=99)

    monkeypatch.setattr(bps, "get_db_session", lambda: NullDBSession())
    monkeypatch.setattr(bps.file_repository, "get_by_object_name", _get_by_object_name)
    monkeypatch.setattr(bps.prediction_service, "predict", _predict)
    monkeypatch.setattr(bps.sse_emitter_manager, "send_event", _send)
    monkeypatch.setattr(bps.ai_artifact_service, "register_artifact", _register)
    return base, calls, events


class TestValidateOwnedImageUrl:
    async def test_external_domain_rejected(self, monkeypatch):
        _base, calls, _events = _patch(monkeypatch)
        summary = await bps.process_batch(1, 2, _OWNER, ["https://evil.com/a.png"], 1, "s1")
        assert calls == []
        assert (summary["total"], summary["success"], summary["failed"]) == (1, 0, 1)
        assert "本系统产物地址" in summary["failures"][0]["reason"]

    async def test_internal_ip_rejected(self, monkeypatch):
        _base, calls, _events = _patch(monkeypatch)
        summary = await bps.process_batch(
            1, 2, _OWNER, ["http://169.254.169.254/latest/meta-data"], 1, "s1"
        )
        assert calls == []
        assert summary["failed"] == 1

    async def test_other_users_file_rejected(self, monkeypatch):
        base, calls, _events = _patch(monkeypatch, owner_id=_OTHER)
        summary = await bps.process_batch(1, 2, _OWNER, [f"{base}/own.png"], 1, "s1")
        assert calls == []
        assert summary["failed"] == 1
        assert "不属于当前用户" in summary["failures"][0]["reason"]

    async def test_missing_file_rejected(self, monkeypatch):
        base, calls, _events = _patch(monkeypatch, file_exists=False)
        summary = await bps.process_batch(1, 2, _OWNER, [f"{base}/ghost.png"], 1, "s1")
        assert calls == []
        assert summary["failed"] == 1

    async def test_owned_file_passes(self, monkeypatch):
        base, calls, _events = _patch(monkeypatch)
        summary = await bps.process_batch(1, 2, _OWNER, [f"{base}/own.png"], 1, "s1")
        assert calls == [f"{base}/own.png"]
        assert summary["success"] == 1
        assert summary["failed"] == 0
