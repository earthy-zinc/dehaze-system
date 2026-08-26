from app.service.ai.strategies.quota_recall import quota_recall
from tests.stubs.fakes import RecorderEmitter


def _make_ctx(token_used=0, token_budget=0, remaining_budget=None):
    ctx = {
        "token_used": token_used,
        "token_budget": token_budget,
        "stream_session_id": "s1",
        "model_id": "gpt-4o-mini",
        "messages": [],
    }
    if remaining_budget is not None:
        ctx["billing_context"] = {"user_id": 10, "remaining_budget": remaining_budget}
    return ctx


def _patch_precharge(monkeypatch, estimate=10, prededuct=True):
    async def _estimate(db, model_id, messages):
        return estimate

    async def _prededuct(db, uid, credits):
        return prededuct

    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.estimate_service.estimate_step_credits", _estimate
    )
    monkeypatch.setattr("app.service.ai.strategies.quota_recall.quota_service.pre_deduct", _prededuct)


async def test_precharge_batch_deducts_budget(monkeypatch):
    ctx = _make_ctx(remaining_budget=100)
    _patch_precharge(monkeypatch)

    assert await quota_recall.precharge_batch(ctx, batch_size=3) is True
    assert ctx["billing_context"]["remaining_budget"] == 70
    assert ctx["billing_context"]["precharged_batch"] == 30


async def test_precharge_batch_insufficient_returns_false(monkeypatch):
    ctx = _make_ctx(remaining_budget=20)
    _patch_precharge(monkeypatch)

    assert await quota_recall.precharge_batch(ctx, batch_size=3) is False
    assert ctx["billing_context"]["remaining_budget"] == 20


async def test_precharge_batch_no_billing_ctx_passes():
    ctx = _make_ctx()
    assert await quota_recall.precharge_batch(ctx, batch_size=5) is True


async def test_check_and_recall_recalls_pending_when_exhausted(monkeypatch):
    ctx = _make_ctx(token_used=900, token_budget=800)
    emitter = RecorderEmitter()

    monkeypatch.setattr("app.service.ai.strategies.quota_recall.sse_emitter_manager", emitter)

    running = [{"task_id": "r1"}, {"task_id": "r2"}]
    pending = [{"task_id": "p1"}, {"task_id": "p2"}, {"task_id": "p3"}]
    recalled = await quota_recall.check_and_recall(ctx, running, pending)

    assert [t["task_id"] for t in recalled] == ["p1", "p2", "p3"]
    assert emitter.events and emitter.events[0][0] == "thought"
    assert emitter.events[0][1]["status"] == 3
    assert "配额不足" in emitter.events[0][1]["thought"]


async def test_check_and_recall_no_pending_when_ok(monkeypatch):
    ctx = _make_ctx(token_used=100, token_budget=800)
    emitter = RecorderEmitter()

    monkeypatch.setattr("app.service.ai.strategies.quota_recall.sse_emitter_manager", emitter)

    recalled = await quota_recall.check_and_recall(ctx, [{"task_id": "r1"}], [{"task_id": "p1"}])
    assert recalled == []
    assert emitter.events == []
