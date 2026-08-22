from app.service.ai.dehaze_tools_builder import build_business_tools
from tests.stubs import NullDBSession


async def _invoke_batch(image_urls, algorithm_id=1):
    ctx = {
        "conversation_id": 1,
        "message_id": 2,
        "user_id": 10,
        "stream_session_id": "s1",
        "model_id": "gpt-4o-mini",
        "token_budget": 1000,
        "max_steps": 20,
        "task_type": "",
        "task_algorithm": "",
        "task_params": {},
        "task_status": "",
        "task_id": "",
        "task_artifacts": [],
    }
    tools = build_business_tools(ctx)
    batch_tool = next(t for t in tools if t.name == "batch_process")
    result = await batch_tool.func(image_urls, "去雾", algorithm_id)
    return result, ctx


def _patch_async_batch(monkeypatch, task_id="batch:1:2:555", resume_payload=None):
    captured = {}

    def _fake_submit(**kwargs):
        captured["submit_kwargs"] = kwargs
        return task_id

    monkeypatch.setattr("app.service.ai.dehaze_tools_builder.submit_batch_task", _fake_submit)

    saved = {}

    class _IH:
        async def save_interrupt(self, thread_id, itype, data):
            saved.update({"t": thread_id, "type": itype, "data": data})

    monkeypatch.setattr("app.service.ai.dehaze_tools_builder.interrupt_handler", _IH())
    monkeypatch.setattr("app.service.ai.dehaze_tools_builder.get_db_session", NullDBSession)

    class _Repo:
        async def update_task_id(self, db, mid, tid):
            captured["task_id_written"] = tid

    monkeypatch.setattr(
        "app.repository.ai_message_repository.ai_message_repository", _Repo()
    )

    if resume_payload is None:
        resume_payload = {"async_task": {"total": 4, "success": 4, "failed": 0, "results": []}}
    monkeypatch.setattr(
        "app.service.ai.dehaze_tools_builder.interrupt",
        lambda data: (captured.update({"interrupt_data": data}) or resume_payload),
    )
    return captured, saved, resume_payload


async def test_small_batch_stays_sync(monkeypatch):
    summary = {"total": 2, "success": 2, "failed": 0, "results": []}

    async def _fake_process(*a, **k):
        return summary

    monkeypatch.setattr("app.service.ai.dehaze_tools_builder.process_batch", _fake_process)

    called = {"submit": False, "interrupt": False}
    monkeypatch.setattr(
        "app.service.ai.dehaze_tools_builder.submit_batch_task",
        lambda **kwargs: (called.update({"submit": True}) or "batch:1:2:123"),
    )
    monkeypatch.setattr(
        "app.service.ai.dehaze_tools_builder.interrupt",
        lambda data: (called.update({"interrupt": True}) or {}),
    )

    result, ctx = await _invoke_batch(["a", "b"])
    assert "2 张" in result
    assert not called["submit"]
    assert not called["interrupt"]
    assert ctx["task_status"] == "completed"


async def test_large_batch_submits_async_and_interrupts(monkeypatch):
    captured, saved, _resume = _patch_async_batch(monkeypatch)

    result, ctx = await _invoke_batch(["a", "b", "c", "d"])

    data = captured["interrupt_data"]
    assert data["type"] == "async_wait"
    assert data["stream_session_id"] == "s1"
    assert data["data"]["task_id"] == "batch:1:2:555"
    assert data["data"]["task_type"] == "batch_process"
    assert data["data"]["image_count"] == 4
    assert data["data"]["est_duration"]
    assert saved["type"] == "async_wait"
    assert saved["t"] == "1:2"
    assert saved["data"]["data"]["task_id"] == "batch:1:2:555"
    assert captured["submit_kwargs"]["thread_id"] == "1:2"
    assert captured["task_id_written"] == "batch:1:2:555"
    assert "4 张" in result
    assert ctx["task_status"] == "completed"
    assert ctx["task_id"] == "batch:1:2:555"


async def test_large_batch_failure_result(monkeypatch):
    captured, saved, resume = _patch_async_batch(
        monkeypatch,
        task_id="batch:1:2:666",
        resume_payload={"async_task": {"total": 4, "success": 1, "failed": 3, "results": []}},
    )

    result, ctx = await _invoke_batch(["a", "b", "c", "d"])
    assert ctx["task_status"] == "failed"
    assert ctx["task_id"] == "batch:1:2:666"
    assert "失败 3 张" in result
    assert saved["type"] == "async_wait"
    assert captured["task_id_written"] == "batch:1:2:666"
