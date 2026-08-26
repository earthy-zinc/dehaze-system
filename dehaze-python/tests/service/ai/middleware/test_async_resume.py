from app.service.ai.middleware import async_resume


class _Emitter:
    def __init__(self, acquire=True):
        self.acquire_result = acquire
        self.released = False

    async def acquire_lock(self, conv_id):
        return self.acquire_result

    async def release_lock(self, conv_id):
        self.released = True


class _RS:
    def __init__(self):
        self.calls = []

    async def resume(self, **kw):
        self.calls.append(kw)


class _FailingRS:
    async def resume(self, **kw):
        raise RuntimeError("boom")


def _mapping():
    return {
        "thread_id": "1:2",
        "conv_id": 1,
        "msg_id": 2,
        "user_id": 10,
        "stream_session_id": "s1",
    }


async def test_task_mapping_save_and_load(mock_redis):
    task_id = "batch:1:2:123"
    mapping = _mapping()
    await async_resume._save_task_mapping(task_id, mapping)
    assert await async_resume._load_task_mapping(task_id) == mapping
    assert await mock_redis.exists("ai:async_task:batch:1:2:123")


async def test_notify_completed_resumes(mock_redis, monkeypatch):
    task_id = "batch:1:2:999"
    await async_resume._save_task_mapping(task_id, _mapping())

    rs = _RS()
    emitter = _Emitter(acquire=True)
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)

    await async_resume.notify_task_completed(task_id, {"total": 4, "success": 4, "failed": 0})

    assert len(rs.calls) == 1
    call = rs.calls[0]
    assert call["conv_id"] == 1
    assert call["msg_id"] == 2
    assert call["user_id"] == 10
    assert call["resume_data"] == {"async_task": {"total": 4, "success": 4, "failed": 0}}
    assert emitter.released
    assert not await mock_redis.exists("ai:async_task:batch:1:2:999")


async def test_notify_skipped_when_lock_held(mock_redis, monkeypatch):
    task_id = "batch:1:2:888"
    await async_resume._save_task_mapping(task_id, _mapping())

    rs = _RS()
    emitter = _Emitter(acquire=False)
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)

    await async_resume.notify_task_completed(task_id, {"total": 4})

    assert rs.calls == []
    assert not emitter.released


async def test_notify_releases_lock_on_failure(mock_redis, monkeypatch):
    task_id = "batch:1:2:777"
    await async_resume._save_task_mapping(task_id, _mapping())

    emitter = _Emitter(acquire=True)
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", _FailingRS())
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)

    await async_resume.notify_task_completed(task_id, {"total": 4})

    assert emitter.released


async def test_notify_ignored_without_mapping(mock_redis, monkeypatch):
    rs = _RS()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    await async_resume.notify_task_completed("batch:missing:1", {"total": 1})
    assert rs.calls == []
