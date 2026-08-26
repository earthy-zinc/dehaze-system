from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.ai_schedule import ScheduleCreate, SchedulePageQuery, ScheduleUpdate
from app.service.ai.service import ai_schedule_service as m
from app.service.ai.service.ai_schedule_service import (
    DEFAULT_TIMEZONE,
    MAX_SCHEDULES_PER_USER,
    scheduled_task_service,
)
from tests.stubs.fakes import StubAsyncSession
from tests.stubs.factories import make_member


def _task(
    task_id=1,
    user_id=1,
    status=1,
    enabled=1,
    cron="0 9 * * *",
    timezone=DEFAULT_TIMEZONE,
    deleted=0,
    next_trigger_time=None,
):
    return SimpleNamespace(
        id=task_id,
        user_id=user_id,
        name="去雾任务",
        cron=cron,
        timezone=timezone,
        input=None,
        output=None,
        enabled=enabled,
        status=status,
        circuit_streak=0,
        next_trigger_time=next_trigger_time,
        create_time=datetime.now(),
        deleted=deleted,
    )


class _ScheduleRepo:
    def __init__(self, task=None):
        self.task = task
        self.count = 0
        self.items = []
        self.total = 0
        self.calls = {
            "set_enabled": [],
            "reset_circuit": [],
            "soft_delete": [],
            "update_next_trigger": [],
            "create": [],
        }

    async def get_by_id(self, db, schedule_id, **kw):
        return self.task

    async def count_by_user(self, db, user_id):
        return self.count

    async def create(self, db, entity):
        entity.id = 1
        self.calls["create"].append(entity)
        return entity

    async def paginate(self, db, stmt, page, size):
        return self.items, self.total

    async def set_enabled(self, db, schedule_id, enabled):
        self.calls["set_enabled"].append((schedule_id, enabled))

    async def soft_delete(self, db, schedule_id):
        self.calls["soft_delete"].append(schedule_id)

    async def reset_circuit(self, db, schedule_id):
        self.calls["reset_circuit"].append(schedule_id)

    async def update_next_trigger(self, db, schedule_id, next_trigger_time):
        self.calls["update_next_trigger"].append((schedule_id, next_trigger_time))


class _RunRepo:
    def __init__(self):
        self.latest = {}
        self.history = []
        self.total = 0

    async def get_latest_by_schedule_ids(self, db, schedule_ids):
        return self.latest

    async def page_by_schedule(self, db, schedule_id, page, size):
        return self.history, self.total


def _member_repo(members):
    class _MemberRepo:
        async def get_by_user_id(self, db, user_id):
            return members.get(user_id)

    return _MemberRepo()


@pytest.fixture
def env():
    sched = _ScheduleRepo()
    run = _RunRepo()
    members = {}
    svc = m.ScheduledTaskService(
        ai_schedule_repository=sched,
        ai_schedule_run_repository=run,
        member_repository=_member_repo(members),
    )
    return SimpleNamespace(db=StubAsyncSession(), svc=svc, sched=sched, run=run, members=members)


def _form(**kw):
    return ScheduleCreate(**{"name": "去雾任务", "cron": "0 9 * * *", **kw})


class TestCreate:
    async def test_reject_non_vip2(self, env):
        env.members[1] = make_member("level_1")
        with pytest.raises(BusinessException) as exc:
            await env.svc.create(env.db, 1, _form())
        assert "VIP2" in str(exc.value)

    async def test_reject_no_member(self, env):
        with pytest.raises(BusinessException) as exc:
            await env.svc.create(env.db, 1, _form())
        assert "VIP2" in str(exc.value)

    async def test_vip2_ok_and_compute_next_trigger(self, env):
        env.members[1] = make_member("level_2")
        task = await env.svc.create(env.db, 1, _form())
        assert task.nextTriggerTime is not None
        assert task.id == 1

    async def test_reject_over_limit(self, env):
        env.members[1] = make_member("level_2")
        env.sched.count = MAX_SCHEDULES_PER_USER
        with pytest.raises(BusinessException) as exc:
            await env.svc.create(env.db, 1, _form())
        assert "上限" in str(exc.value)

    async def test_reject_invalid_cron(self, env):
        env.members[1] = make_member("level_2")
        with pytest.raises(BusinessException) as exc:
            await env.svc.create(env.db, 1, _form(cron="not a cron"))
        assert "Cron" in str(exc.value)


class TestSetEnabled:
    async def test_enable_resets_circuit_when_broken(self, env):
        env.sched.task = _task(status=2)
        await env.svc.set_enabled(env.db, 1, 1, 1)
        assert env.sched.calls["reset_circuit"] == [1]
        assert env.sched.calls["set_enabled"] == [(1, 1)]
        assert len(env.sched.calls["update_next_trigger"]) == 1

    async def test_enable_normal_does_not_reset_circuit(self, env):
        env.sched.task = _task(status=1)
        await env.svc.set_enabled(env.db, 1, 1, 1)
        assert env.sched.calls["reset_circuit"] == []

    async def test_disable_only_sets_enabled_zero(self, env):
        env.sched.task = _task(status=2)
        await env.svc.set_enabled(env.db, 1, 1, 0)
        assert env.sched.calls["set_enabled"] == [(1, 0)]
        assert env.sched.calls["reset_circuit"] == []
        assert env.sched.calls["update_next_trigger"] == []

    async def test_ownership_denied(self, env):
        env.sched.task = _task(user_id=999)
        with pytest.raises(BusinessException):
            await env.svc.set_enabled(env.db, 1, 1, 1)


class TestUpdate:
    async def test_update_recomputes_next_trigger_on_cron_change(self, env):
        env.sched.task = _task(cron="0 8 * * *")
        await env.svc.update(env.db, 1, 1, ScheduleUpdate(cron="0 7 * * *"))
        assert env.sched.task.cron == "0 7 * * *"
        assert env.sched.task.next_trigger_time is not None
        assert env.db.refreshed

    async def test_update_keep_next_trigger_without_cron_change(self, env):
        t = _task(cron="0 8 * * *")
        env.sched.task = t
        await env.svc.update(env.db, 1, 1, ScheduleUpdate(name="新名称"))
        assert t.name == "新名称"
        assert t.next_trigger_time is None

    async def test_update_reject_invalid_cron(self, env):
        env.sched.task = _task()
        with pytest.raises(BusinessException):
            await env.svc.update(env.db, 1, 1, ScheduleUpdate(cron="bad"))


class TestPreviewNextTimes:
    async def test_preview_description_and_times(self):
        result = await scheduled_task_service.preview_next_times("0 9 * * *")
        assert "每天 09点00分" in result.description
        assert len(result.nextTimes) == 5

    async def test_preview_invalid_cron(self):
        with pytest.raises(BusinessException):
            await scheduled_task_service.preview_next_times("bad cron")


class TestListPage:
    def _run_summary(self, **kw):
        return SimpleNamespace(
            status=kw.get("status", 1),
            skip_reason=kw.get("skip_reason"),
            credits=kw.get("credits"),
            duration_ms=kw.get("duration_ms"),
            error_msg=kw.get("error_msg"),
            conversation_id=kw.get("conversation_id"),
            create_time=datetime.now(),
        )

    async def test_aggregate_last_run_summary(self, env):
        env.sched.items = [_task()]
        env.sched.total = 1
        run = self._run_summary(status=1, credits=12.5, duration_ms=300, conversation_id=9)
        env.run.latest = {1: run}
        result = await env.svc.list_page(env.db, 1, SchedulePageQuery())
        assert result.total == 1
        assert result.list[0].lastRun is not None
        assert result.list[0].lastRun.status == 1
        assert result.list[0].lastRun.credits == 12.5

    async def test_last_run_none_when_no_history(self, env):
        env.sched.items = [_task()]
        env.sched.total = 1
        env.run.latest = {}
        result = await env.svc.list_page(env.db, 1, SchedulePageQuery())
        assert result.list[0].lastRun is None


class TestDelete:
    async def test_soft_delete_owned(self, env):
        env.sched.task = _task()
        await env.svc.delete(env.db, 1, 1)
        assert env.sched.calls["soft_delete"] == [1]

    async def test_delete_denied(self, env):
        env.sched.task = _task(user_id=999)
        with pytest.raises(BusinessException):
            await env.svc.delete(env.db, 1, 1)


class TestListHistory:
    async def test_page_history_with_ownership(self, env):
        env.sched.task = _task()
        run = SimpleNamespace(
            id=3,
            schedule_id=1,
            status=2,
            skip_reason="overlap",
            credits=None,
            duration_ms=100,
            error_msg="超时",
            conversation_id=None,
            request_id="req-1",
            window_start=datetime.now(),
            create_time=datetime.now(),
        )
        env.run.history = [run]
        env.run.total = 1
        result = await env.svc.list_history(env.db, 1, 1, 1, 10)
        assert result.total == 1
        assert result.list[0].skipReason == "overlap"
        assert result.list[0].errorMsg == "超时"

    async def test_history_ownership_denied(self, env):
        env.sched.task = _task(user_id=999)
        with pytest.raises(BusinessException):
            await env.svc.list_history(env.db, 1, 1, 1, 10)


class TestIdempotentInsert:
    async def test_insert_success(self):
        from app.repository.ai_schedule_run_repository import ai_schedule_run_repository

        entity = SimpleNamespace(schedule_id=1, window_start=datetime.now())
        db = StubAsyncSession()
        result = await ai_schedule_run_repository.create_with_window(db, entity)
        assert result is entity
        assert db.flushed == 1

    async def test_integrity_error_returns_existing(self, monkeypatch):
        from app.repository.ai_schedule_run_repository import (
            ai_schedule_run_repository as run_repo,
        )

        existing = SimpleNamespace(schedule_id=1, window_start=datetime.now())
        entity = SimpleNamespace(schedule_id=1, window_start=existing.window_start)
        db = StubAsyncSession()

        async def flush():
            raise IntegrityError("stmt", "params", Exception("dup"))

        async def get_by_window(d, schedule_id, window_start):
            assert window_start == existing.window_start
            return existing

        db.flush = flush
        monkeypatch.setattr(run_repo, "get_by_window", get_by_window)
        result = await run_repo.create_with_window(db, entity)
        assert result is existing
        assert db.savepoint_released == 1


@pytest.fixture
def freq_env():
    sched = _ScheduleRepo()
    svc = m.ScheduledTaskService(
        ai_schedule_repository=sched,
        member_repository=_member_repo({1: make_member("level_2")}),
    )
    return SimpleNamespace(db=StubAsyncSession(), svc=svc, sched=sched)


def _freq_form(cron):
    return ScheduleCreate(
        **{
            "name": "去雾任务",
            "cron": cron,
            "input": {"type": "fixed", "content": "x"},
            "output": {"type": "message"},
        }
    )


async def test_frequency_daily_identifier_normalized(freq_env):
    task = await freq_env.svc.create(freq_env.db, 1, _freq_form("daily@09:00"))
    stored = freq_env.sched.calls["create"][0]
    assert stored.cron == "0 9 * * *"
    assert stored.next_trigger_time.hour == 9
    assert stored.next_trigger_time.minute == 0
    assert task.nextTriggerTime is not None


async def test_frequency_weekly_weekday_alias_normalized(freq_env):
    await freq_env.svc.create(freq_env.db, 1, _freq_form("weekly@mon@09:30"))
    assert freq_env.sched.calls["create"][0].cron == "30 9 * * 1"


async def test_frequency_weekly_weekday_number_normalized(freq_env):
    await freq_env.svc.create(freq_env.db, 1, _freq_form("weekly@0@09:30"))
    assert freq_env.sched.calls["create"][0].cron == "30 9 * * 0"


async def test_frequency_monthly_normalized(freq_env):
    await freq_env.svc.create(freq_env.db, 1, _freq_form("monthly@15@23:59"))
    assert freq_env.sched.calls["create"][0].cron == "59 23 15 * *"


async def test_frequency_invalid_time_raises_param_error(freq_env):
    with pytest.raises(BusinessException) as exc:
        await freq_env.svc.create(freq_env.db, 1, _freq_form("daily@25:00"))
    assert exc.value.code is ResultCode.PARAM_ERROR


async def test_frequency_invalid_date_raises_param_error(freq_env):
    with pytest.raises(BusinessException) as exc:
        await freq_env.svc.create(freq_env.db, 1, _freq_form("monthly@32@10:00"))
    assert exc.value.code is ResultCode.PARAM_ERROR


async def test_frequency_unrecognized_format_passthrough(freq_env):
    with pytest.raises(BusinessException) as exc:
        await freq_env.svc.create(freq_env.db, 1, _freq_form("custom@09:00"))
    assert "Cron" in str(exc.value)


async def test_preview_frequency_identifier_usable():
    result = await scheduled_task_service.preview_next_times("daily@09:00")
    assert "每天 09点00分" in result.description
    assert len(result.nextTimes) == 5
    for t in result.nextTimes:
        assert t.hour == 9 and t.minute == 0


def test_next_times_computed_in_task_timezone():
    tz = "America/New_York"
    times = scheduled_task_service._compute_next_times("0 9 * * *", tz, 3)
    assert all(t.hour == 9 and t.minute == 0 for t in times)
    utcoffset_hours = {t.utcoffset().total_seconds() / 3600 for t in times}
    assert utcoffset_hours.issubset({-4.0, -5.0})


def test_next_trigger_computed_in_task_timezone():
    tz = "America/New_York"
    nxt = scheduled_task_service._compute_next_trigger("30 9 * * *", tz)
    assert nxt.hour == 9 and nxt.minute == 30


def test_next_times_rejects_invalid_timezone():
    with pytest.raises(BusinessException):
        scheduled_task_service._compute_next_times("0 9 * * *", "Bad/Zone", 3)
