from tests.stubs import install_reasoning_chain_mocks


async def test_async_wait_suspend_skips_finalize_but_pushes_end(monkeypatch):
    interrupt_data = {
        "type": "async_wait",
        "data": {"task_id": "batch:1:2:1", "stream_session_id": "s1"},
    }
    service, recorder = install_reasoning_chain_mocks(
        monkeypatch,
        interrupt=interrupt_data,
        snapshot={"reasoning_mode": "planning", "max_steps": 20},
        resolve_mode=("planning", 20),
    )

    result = await service.run(1, 10, 2, "gpt-4o-mini", "s1")

    assert recorder["finalized"] == []
    assert recorder["suggested"] == 0
    assert recorder["step_summaries"] == 0
    assert recorder["pushed_end"] == [0]
    assert recorder["released"] == [1]
    assert recorder["sync"] == [1]
    assert result["stop_reason"] == "stop"


async def test_confirm_suspend_still_finalizes(monkeypatch):
    interrupt_data = {"type": "confirm", "data": {"stream_session_id": "s1"}}
    service, recorder = install_reasoning_chain_mocks(
        monkeypatch,
        interrupt=interrupt_data,
        snapshot={"reasoning_mode": "planning", "max_steps": 20},
        resolve_mode=("planning", 20),
    )

    await service.run(1, 10, 2, "gpt-4o-mini", "s1")

    assert len(recorder["finalized"]) == 1
    assert recorder["suggested"] == 1
    assert recorder["step_summaries"] == 1
    assert recorder["pushed_end"] == [0]
    assert recorder["released"] == [1]
    assert recorder["sync"] == [1]
