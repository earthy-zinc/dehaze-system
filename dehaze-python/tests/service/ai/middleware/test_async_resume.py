import asyncio

from app.service.ai.middleware import async_resume
from app.service.ai.middleware.interrupt_handler import interrupt_handler


class _Emitter:
    def __init__(self, acquire=(True,), fallback=True):
        self.results = list(acquire)
        # 预设结果用尽后的返回值（默认放行）
        self.fallback = fallback
        self.released = False
        self.errors = []

    async def acquire_lock(self, conv_id):
        return self.results.pop(0) if self.results else self.fallback

    async def release_lock(self, conv_id):
        self.released = True

    async def send_error(self, stream_session_id, error):
        self.errors.append((stream_session_id, error))


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
    # 自动恢复前置：中断点须存在（已停止/已恢复的消息不得被回调续流）
    await interrupt_handler.save_interrupt("1:2", "async_wait", {"taskId": task_id})

    rs = _RS()
    emitter = _Emitter()
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


async def test_notify_skips_when_interrupt_cleared(mock_redis, monkeypatch):
    """中断点已清理（用户停止/手动恢复过）：静默跳过自动恢复，不报错不续流"""
    task_id = "batch:1:2:555"
    await async_resume._save_task_mapping(task_id, _mapping())

    rs = _RS()
    emitter = _Emitter()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)

    await async_resume.notify_task_completed(task_id, {"total": 4})

    assert rs.calls == []
    assert emitter.released  # finally 仍释放锁


async def test_notify_retries_when_lock_held(mock_redis, monkeypatch):
    """锁被占用：回滚反查键后延迟重试，不得丢键挂起到 TTL 过期。"""
    task_id = "batch:1:2:888"
    await async_resume._save_task_mapping(task_id, _mapping())
    await interrupt_handler.save_interrupt("1:2", "async_wait", {"taskId": task_id})

    rs = _RS()
    emitter = _Emitter(acquire=(False, True))
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)
    monkeypatch.setattr(async_resume, "_RESUME_RETRY_DELAY", 0.01)

    await async_resume.notify_task_completed(task_id, {"total": 4})

    # 抢锁失败时反查键已回滚（重试计数写入），等待重试完成
    restored = await async_resume._load_task_mapping(task_id)
    assert restored is not None
    assert restored["resume_retry"] == 1
    await asyncio.sleep(0.05)

    assert len(rs.calls) == 1
    assert rs.calls[0]["resume_data"] == {"async_task": {"total": 4}}
    assert emitter.released
    assert not await mock_redis.exists("ai:async_task:batch:1:2:888")


async def test_notify_gives_up_after_max_retries(mock_redis, monkeypatch):
    """重试耗尽：交回用户侧恢复，且不再残留反查键。"""
    task_id = "batch:1:2:666"
    await async_resume._save_task_mapping(task_id, _mapping())

    rs = _RS()
    emitter = _Emitter(acquire=(False, False, False, False), fallback=False)
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)
    monkeypatch.setattr(async_resume, "_RESUME_RETRY_DELAY", 0.01)

    await async_resume.notify_task_completed(task_id, {"total": 4})
    await asyncio.sleep(0.2)

    assert rs.calls == []
    assert not emitter.released
    assert await async_resume._load_task_mapping(task_id) is None


async def test_notify_releases_lock_on_failure(mock_redis, monkeypatch):
    task_id = "batch:1:2:777"
    await async_resume._save_task_mapping(task_id, _mapping())
    await interrupt_handler.save_interrupt("1:2", "async_wait", {"taskId": task_id})

    emitter = _Emitter()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", _FailingRS())
    monkeypatch.setattr("app.service.ai.middleware.async_resume.sse_emitter_manager", emitter)

    await async_resume.notify_task_completed(task_id, {"total": 4})

    assert emitter.released
    # 失败收尾走 send_error 单一出口（error + message.end），不自行拼装载荷
    assert len(emitter.errors) == 1
    assert emitter.errors[0][0] == "s1"
    assert isinstance(emitter.errors[0][1], RuntimeError)


async def test_notify_ignored_without_mapping(mock_redis, monkeypatch):
    rs = _RS()
    monkeypatch.setattr("app.service.ai.service.reasoning_service.reasoning_service", rs)
    await async_resume.notify_task_completed("batch:missing:1", {"total": 1})
    assert rs.calls == []
