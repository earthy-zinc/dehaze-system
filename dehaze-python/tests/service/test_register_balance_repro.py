from decimal import Decimal

from app.models.entity.sys_user import SysUser
from app.repository.user_repository import user_repository


class RecordingSession:

    def __init__(self):
        self.last_stmt = None

    async def execute(self, stmt, **kw):
        self.last_stmt = stmt
        return _Result()


class _Result:

    @property
    def rowcount(self):
        return 1


def _exec_opts(stmt) -> dict:
    return getattr(stmt, "_execution_options", {}) or {}


async def test_increase_balance_cas_disables_evaluate_sync():
    session = RecordingSession()
    assert await user_repository.increase_balance_cas(session, 1, Decimal("100"), 0) is True
    assert _exec_opts(session.last_stmt).get("synchronize_session") is False


async def test_deduct_balance_cas_disables_evaluate_sync():
    session = RecordingSession()
    assert await user_repository.deduct_balance_cas(session, 1, Decimal("50"), 0) is True
    assert _exec_opts(session.last_stmt).get("synchronize_session") is False


def test_model_credit_default_is_decimal():
    col = SysUser.__table__.c.credits_balance
    assert col.default is not None
    default = col.default.arg
    assert isinstance(default, Decimal), type(default)
    assert default == Decimal("0.00")
