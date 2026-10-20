import pytest

from app.service.ai.strategies.quota_recall import quota_recall
from tests.stubs.fakes import NullDBSession, RecorderEmitter


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """预扣/结算只需事务语义，DB 交互全部打桩，避免依赖真实测试库。"""
    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.get_db_session", lambda: NullDBSession()
    )


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


def _tasks(*descriptions):
    return [{"id": f"T{i}", "description": d} for i, d in enumerate(descriptions)]


def _patch_precharge(monkeypatch, prededuct=True, credits=None):
    """预估算子：按子任务描述长度计费（1 字符 1 积分），便于断言预估口径。"""

    async def _estimate(db, model_id, messages):
        return sum(len(m.get("content") or "") for m in messages)

    async def _prededuct(db, uid, amount):
        return prededuct

    async def _calculate(db, model_id, provider_id, i, o, c):
        return {"credits": credits if credits is not None else i + o}

    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.estimate_service.estimate_step_credits", _estimate
    )
    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.quota_service.pre_deduct", _prededuct
    )
    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.rate_provider.calculate", _calculate
    )


def _patch_quota_ops(monkeypatch):
    prededucted: list[int] = []
    refunded: list[int] = []

    async def _prededuct(db, uid, amount):
        prededucted.append(amount)
        return True

    async def _refund(uid, amount):
        refunded.append(amount)

    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.quota_service.pre_deduct", _prededuct
    )
    monkeypatch.setattr("app.service.ai.strategies.quota_recall.quota_service.refund", _refund)
    return prededucted, refunded


async def test_precharge_batch_estimates_by_task_description(monkeypatch):
    ctx = _make_ctx(remaining_budget=100)
    _patch_precharge(monkeypatch)
    prededucted, _ = _patch_quota_ops(monkeypatch)

    # 预估按子任务描述长度逐条累加（空上下文不得再导致低估）
    reserved = await quota_recall.precharge_batch(ctx, _tasks("abc", "de"))

    assert reserved == 5
    assert prededucted == [5]
    assert ctx["billing_context"]["remaining_budget"] == 95
    assert ctx["billing_context"]["precharged_batch"] == 5


async def test_precharge_batch_no_tasks_is_noop(monkeypatch):
    ctx = _make_ctx(remaining_budget=1)
    _patch_precharge(monkeypatch)
    _, refunded = _patch_quota_ops(monkeypatch)

    assert await quota_recall.precharge_batch(ctx, []) == 0
    assert refunded == []


async def test_precharge_batch_insufficient_returns_none(monkeypatch):
    ctx = _make_ctx(remaining_budget=2)
    _patch_precharge(monkeypatch)
    prededucted, _ = _patch_quota_ops(monkeypatch)

    assert await quota_recall.precharge_batch(ctx, _tasks("abc", "de")) is None
    assert prededucted == []
    assert ctx["billing_context"]["remaining_budget"] == 2


async def test_precharge_batch_quota_rejected_returns_none(monkeypatch):
    ctx = _make_ctx(remaining_budget=100)
    _patch_precharge(monkeypatch, prededuct=False)

    assert await quota_recall.precharge_batch(ctx, _tasks("abc")) is None
    assert "precharged_batch" not in ctx["billing_context"]


async def test_precharge_batch_no_billing_ctx_passes():
    ctx = _make_ctx()
    assert await quota_recall.precharge_batch(ctx, _tasks("abc")) == 0


async def test_settle_batch_refunds_reservation_and_fixes_budget(monkeypatch):
    """预留全额退回（实际消耗由主结算统一扣减），预算池按实际消耗修正。"""
    ctx = _make_ctx(remaining_budget=100)
    _patch_precharge(monkeypatch)
    prededucted, refunded = _patch_quota_ops(monkeypatch)
    reserved = await quota_recall.precharge_batch(ctx, _tasks("abc", "de"))
    assert reserved is not None

    await quota_recall.settle_batch(
        ctx, reserved, {"input_tokens": 2, "output_tokens": 1, "cached_input_tokens": 0}
    )

    assert prededucted == [5]
    assert refunded == [5]
    # 预扣 5、实际 3 → 预算池回补 2
    assert ctx["billing_context"]["remaining_budget"] == 97
    assert ctx["billing_context"]["precharged_batch"] == 0


async def test_settle_batch_failed_tasks_refund_full(monkeypatch):
    """失败/跳过任务无实际消耗：预留全额退回，预算池同步回补。"""
    ctx = _make_ctx(remaining_budget=100)
    _patch_precharge(monkeypatch)
    _, refunded = _patch_quota_ops(monkeypatch)
    reserved = await quota_recall.precharge_batch(ctx, _tasks("abc"))
    assert reserved is not None

    await quota_recall.settle_batch(ctx, reserved, {})

    assert refunded == [3]
    assert ctx["billing_context"]["remaining_budget"] == 100


async def test_settle_batch_keeps_budget_when_rate_unavailable(monkeypatch):
    ctx = _make_ctx(remaining_budget=100)
    _patch_precharge(monkeypatch)
    prededucted, refunded = _patch_quota_ops(monkeypatch)
    reserved = await quota_recall.precharge_batch(ctx, _tasks("abc"))
    assert reserved is not None

    async def _boom(db, model_id, provider_id, i, o, c):
        raise RuntimeError("未配置售价")

    monkeypatch.setattr("app.service.ai.strategies.quota_recall.rate_provider.calculate", _boom)
    await quota_recall.settle_batch(ctx, reserved, {"input_tokens": 9})

    assert refunded == prededucted == [3]
    assert ctx["billing_context"]["remaining_budget"] == 97


async def test_reservation_release_never_drops_quota_below_precharge_baseline(monkeypatch):
    """预留释放只回到主预扣基线，不产生低于基线的可用窗口（防双花）。

    主预扣（before_agent 的 pre_charge）在整个 run 内持续占用配额，批次预留是叠加
    在其之上的临时占用；释放后配额回到基线而非清零，故"批结束→主结算"之间不会有
    比无批次场景更多的可用额度。
    """
    ledger = {"used": 0}

    async def _prededuct(db, uid, amount):
        ledger["used"] += amount
        return True

    async def _refund(uid, amount):
        ledger["used"] -= amount

    _patch_precharge(monkeypatch)
    # 配额账本须覆盖预扣/退回（_patch_precharge 的桩不记账）
    monkeypatch.setattr(
        "app.service.ai.strategies.quota_recall.quota_service.pre_deduct", _prededuct
    )
    monkeypatch.setattr("app.service.ai.strategies.quota_recall.quota_service.refund", _refund)

    # 主预扣基线（pre_charge 的 estimated）
    baseline = 1000
    ledger["used"] = baseline
    ctx = _make_ctx(remaining_budget=200)

    reserved = await quota_recall.precharge_batch(ctx, _tasks("abc", "de"))
    assert reserved is not None
    assert ledger["used"] == baseline + reserved > baseline
    await quota_recall.settle_batch(ctx, reserved, {"input_tokens": 2, "output_tokens": 1})

    assert ledger["used"] == baseline
    assert ctx["billing_context"]["precharged_batch"] == 0

    # 多批次串行：每批释放后都回到基线，占用不累积
    for _ in range(3):
        reserved = await quota_recall.precharge_batch(ctx, _tasks("abc"))
        assert reserved is not None
        assert ledger["used"] == baseline + reserved
        await quota_recall.settle_batch(ctx, reserved, {})
        assert ledger["used"] == baseline


async def test_check_and_recall_recalls_pending_when_exhausted(monkeypatch):
    ctx = _make_ctx(token_used=900, token_budget=800)

    pending = ["p1", "p2", "p3"]
    assert quota_recall.check_and_recall(ctx, pending) == ["p1", "p2", "p3"]

    emitter = RecorderEmitter()
    monkeypatch.setattr("app.service.ai.strategies.quota_recall.sse_emitter_manager", emitter)
    await quota_recall.notify_partial_skipped(ctx, 3)
    assert emitter.events[0][0] == "thought"
    assert emitter.events[0][1]["status"] == 3
    assert "配额不足" in emitter.events[0][1]["thought"]


async def test_check_and_recall_no_pending_when_ok():
    ctx = _make_ctx(token_used=100, token_budget=800)
    assert quota_recall.check_and_recall(ctx, ["p1"]) == []


async def test_check_and_recall_budget_exhausted_by_billing():
    ctx = _make_ctx(remaining_budget=0)
    assert quota_recall.check_and_recall(ctx, ["p1"]) == ["p1"]
