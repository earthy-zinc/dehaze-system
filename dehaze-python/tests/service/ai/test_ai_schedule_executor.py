import asyncio
from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service.ai import ai_schedule_executor as ex
from app.service.ai.ai_schedule_executor import ScheduleExecutor
from tests.stubs import FakeInternalResponse, MinimalExecutorDB


def _schedule(**overrides):
    base = dict(
        id=1,
        user_id=7,
        name="每日去雾",
        cron="0 9 * * *",
        timezone="Asia/Shanghai",
        input={"type": "fixed", "content": "处理这批图片"},
        output={"type": "message"},
        enabled=1,
        status=1,
        circuit_streak=0,
        next_trigger_time=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(**overrides):
    base = dict(
        id=100,
        schedule_id=1,
        user_id=7,
        status=1,
        skip_reason=None,
        credits=None,
        duration_ms=None,
        error_msg=None,
        conversation_id=None,
        request_id=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _conv(conv_id=10):
    return SimpleNamespace(id=conv_id)


def _patch_create_conv(monkeypatch, conv_id=10):
    async def _create(db, user_id, form):
        return _conv(conv_id)

    monkeypatch.setattr(ex.ai_conversation_service, "create_conversation", _create)


@pytest.fixture
def mock_db():
    return MinimalExecutorDB()


@pytest.fixture
def executor():
    return ScheduleExecutor()


class _ScheduleRepo:
    def __init__(self, sched=None):
        self.sched = sched

    async def get_by_id(self, db, schedule_id):
        return self.sched

    async def mark_circuit(self, db, schedule_id):
        self.sched.status = 2

    async def update_next_trigger(self, db, schedule_id, nxt):
        self.sched.next_trigger_time = nxt


class _RunRepo:
    def __init__(self, existing=None):
        self.existing = existing
        self.run = None

    async def create_with_window(self, db, entity):
        if self.existing is not None:
            return self.existing
        self.run = entity
        return entity


def _patch_repos(monkeypatch, schedule, run_repo=None):
    sr = _ScheduleRepo(schedule)
    rr = run_repo or _RunRepo()
    monkeypatch.setattr(ex.ai_schedule_repository, "get_by_id", sr.get_by_id)
    monkeypatch.setattr(ex.ai_schedule_repository, "mark_circuit", sr.mark_circuit)
    monkeypatch.setattr(ex.ai_schedule_repository, "update_next_trigger", sr.update_next_trigger)
    monkeypatch.setattr(ex.ai_schedule_run_repository, "create_with_window", rr.create_with_window)
    return sr, rr


def _patch_notify(monkeypatch):
    async def _noop(db, schedule, run):
        return [1]

    monkeypatch.setattr(ex, "notify_run_result", _noop)


def _patch_quota(monkeypatch, used=(0, 0)):
    async def _get_limits(db, user_id):
        return (100, 1000)

    async def _get_used(user_id):
        return used

    monkeypatch.setattr(ex.quota_service, "get_limits", _get_limits)
    monkeypatch.setattr(ex.quota_service, "get_used", _get_used)


def _patch_inference_success(monkeypatch, credits=3, conv_id=10):
    _patch_create_conv(monkeypatch, conv_id)

    async def _send(db, conv_id, user_id, form, idem):
        return FakeInternalResponse(b"")

    async def _list(db, conv_id, page, size):
        return ([SimpleNamespace(role="assistant", status=2, credits=credits, error=None)], 1)

    monkeypatch.setattr(ex.ai_message_service, "send_message", _send)
    monkeypatch.setattr(ex.ai_message_repository, "list_by_conversation", _list)


async def test_idempotent_duplicate_window_skipped(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule, run_repo=_RunRepo(existing=_run()))
    called = {"send": False}

    async def _fail(**kw):
        called["send"] = True
        raise AssertionError("不应发起推理")

    monkeypatch.setattr(ex.ai_message_service, "send_message", _fail)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["skipped"] is True
    assert result["skip_reason"] == "idempotent"
    assert called["send"] is False


async def test_overlap_running_mark_skipped(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)

    await mock_redis.set("ai:schedule:1:running", "1")

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["skipped"] is True
    assert result["skip_reason"] == "overlap"


async def test_execute_success_clears_running_mark(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch)
    _patch_inference_success(monkeypatch)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["ok"] is True
    assert not await mock_redis.exists("ai:schedule:1:running")


async def test_circuit_breaker_disables_after_threshold(monkeypatch, mock_redis, mock_db, executor):
    monkeypatch.setattr(ex, "RETRY_MAX", 0)
    schedule = _schedule(circuit_streak=4)
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch)
    _patch_create_conv(monkeypatch)

    async def _fail_send(db, conv_id, user_id, form, idem):
        raise BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "模型不可用")

    monkeypatch.setattr(ex.ai_message_service, "send_message", _fail_send)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["ok"] is False
    assert schedule.circuit_streak == 5
    assert schedule.status == 2
    assert result["circuited"] is True


async def test_success_resets_circuit_streak(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule(circuit_streak=3)
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch)
    _patch_inference_success(monkeypatch)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["ok"] is True
    assert schedule.circuit_streak == 0
    assert schedule.status == 1


async def test_circuit_disabled_task_skipped(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule(status=2)
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    called = {"send": False}

    async def _fail(**kw):
        called["send"] = True

    monkeypatch.setattr(ex.ai_message_service, "send_message", _fail)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["skipped"] is True
    assert result["skip_reason"] == "circuit"
    assert called["send"] is False


async def test_retryable_temporary_error_retries_then_success(
    monkeypatch, mock_redis, mock_db, executor
):
    monkeypatch.setattr(ex, "RETRY_MAX", 3)
    monkeypatch.setattr(ex, "RETRY_BACKOFF", (0, 0, 0))
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch)
    _patch_inference_success(monkeypatch)

    attempts = {"n": 0}

    async def _send(db, conv_id, user_id, form, idem):
        attempts["n"] += 1
        if attempts["n"] <= 3:
            raise ex._RetryableError("网络超时")
        return FakeInternalResponse(b"")

    monkeypatch.setattr(ex.ai_message_service, "send_message", _send)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["ok"] is True
    assert attempts["n"] == 4


async def test_non_retryable_business_error_fails_immediately(
    monkeypatch, mock_redis, mock_db, executor
):
    monkeypatch.setattr(ex, "RETRY_MAX", 3)
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch)
    _patch_create_conv(monkeypatch)

    attempts = {"n": 0}

    async def _fail_send(db, conv_id, user_id, form, idem):
        attempts["n"] += 1
        raise BusinessException(ResultCode.PARAM_ERROR, "参数错误")

    monkeypatch.setattr(ex.ai_message_service, "send_message", _fail_send)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["ok"] is False
    assert attempts["n"] == 1


async def test_quota_insufficient_skipped_and_notified(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch, used=(100, 0))

    called = {"send": False}

    async def _fail(**kw):
        called["send"] = True

    monkeypatch.setattr(ex.ai_message_service, "send_message", _fail)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["skipped"] is True
    assert result["skip_reason"] == "quota"
    assert called["send"] is False


async def test_quota_sufficient_allows_execution(monkeypatch, mock_redis, mock_db, executor):
    schedule = _schedule()
    _patch_repos(monkeypatch, schedule)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch, used=(99, 0))
    _patch_inference_success(monkeypatch, credits=2.5, conv_id=66)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7)
    assert result["ok"] is True
    assert result["conversation_id"] == 66
    assert result["credits"] == 2.5


async def test_global_concurrency_capped_by_semaphore(monkeypatch, mock_redis):
    limit = 3
    total = 10
    executor = ex.ScheduleExecutor.__new__(ex.ScheduleExecutor)
    executor._semaphore = asyncio.Semaphore(limit)

    monkeypatch.setattr(
        ex.ai_schedule_run_repository, "create_with_window", _RunRepo().create_with_window
    )

    active = {"n": 0, "peak": 0}

    async def _run_with_guards(self, db, redis, schedule, run, manual):
        active["n"] += 1
        active["peak"] = max(active["peak"], active["n"])
        try:
            await asyncio.sleep(0.05)
            return {"ok": True}
        finally:
            active["n"] -= 1

    monkeypatch.setattr(ex.ScheduleExecutor, "_run_with_guards", _run_with_guards)

    async def _get_by_id(db, sid):
        return _schedule(id=sid, name=f"任务{sid}")

    monkeypatch.setattr(ex.ai_schedule_repository, "get_by_id", _get_by_id)

    db = MinimalExecutorDB()
    results = await asyncio.gather(
        *[executor.trigger_once(db, mock_redis, sid, 7) for sid in range(1, total + 1)]
    )
    assert all(r["ok"] for r in results)
    assert active["peak"] <= limit


async def test_global_concurrency_lower_than_limit_allows_parallel(monkeypatch, mock_redis):
    limit = 5
    executor = ex.ScheduleExecutor.__new__(ex.ScheduleExecutor)
    executor._semaphore = asyncio.Semaphore(limit)

    monkeypatch.setattr(
        ex.ai_schedule_run_repository, "create_with_window", _RunRepo().create_with_window
    )

    gate = asyncio.Event()
    entered = {"n": 0}

    async def _run_with_guards(self, db, redis, schedule, run, manual):
        entered["n"] += 1
        if entered["n"] == 3:
            gate.set()
        await gate.wait()
        return {"ok": True}

    monkeypatch.setattr(ex.ScheduleExecutor, "_run_with_guards", _run_with_guards)

    async def _get_by_id(db, sid):
        return _schedule(id=sid, name=f"任务{sid}")

    monkeypatch.setattr(ex.ai_schedule_repository, "get_by_id", _get_by_id)

    results = await asyncio.gather(
        *[executor.trigger_once(MinimalExecutorDB(), mock_redis, sid, 7) for sid in (1, 2, 3)]
    )
    assert entered["n"] == 3
    assert all(r["ok"] for r in results)


async def test_run_history_persists_conversation_credits_duration(
    monkeypatch, mock_redis, mock_db, executor
):
    sched = _schedule(id=1, name="任务1")
    run_repo = _RunRepo()
    _patch_repos(monkeypatch, sched, run_repo=run_repo)
    _patch_notify(monkeypatch)
    _patch_quota(monkeypatch)
    _patch_inference_success(monkeypatch, credits=7.5, conv_id=88)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7, manual=False)

    assert result["ok"] is True
    assert result["conversation_id"] == 88
    assert result["credits"] == 7.5
    assert run_repo.run is not None
    assert run_repo.run.status == 1
    assert run_repo.run.conversation_id == 88
    assert run_repo.run.credits == 7.5
    assert run_repo.run.duration_ms is not None


async def test_disabled_task_skipped_with_reason(monkeypatch, mock_redis, mock_db, executor):
    sched = _schedule(id=1, name="任务1")
    sched.enabled = 0
    run_repo = _RunRepo()
    _patch_repos(monkeypatch, sched, run_repo=run_repo)
    _patch_notify(monkeypatch)
    called = {"send": False}

    async def _fail(**kw):
        called["send"] = True

    monkeypatch.setattr(ex.ai_message_service, "send_message", _fail)

    result = await executor.trigger_once(mock_db, mock_redis, 1, 7, manual=False)

    assert result["skipped"] is True
    assert result["skip_reason"] == "disabled"
    assert called["send"] is False
    assert run_repo.run.status == 3
    assert run_repo.run.skip_reason == "disabled"
