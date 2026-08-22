from app.service.auth_service import auth_service


class _User:
    def __init__(self, admin: bool):
        self._admin = admin
        self.id = 1 if admin else 2

    @property
    def is_admin(self) -> bool:
        return self._admin


class _Repo:
    def __init__(self, docs=None, total=0):
        self.docs = docs or []
        self.total = total
        self.calls = []

    async def page_logs(self, page_num, page_size, **kwargs):
        self.calls.append((page_num, page_size, kwargs))
        return self.docs, self.total


def _make_doc(oid="id-1", user_id=1, username="admin", status=1, ip="1.2.3.4"):
    from datetime import UTC, datetime

    return {
        "_id": oid,
        "user_id": user_id,
        "username": username,
        "ip": ip,
        "location": "",
        "browser": "Chrome",
        "os": "Windows",
        "status": status,
        "message": "登录成功",
        "create_time": datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
    }


def _patch(monkeypatch, docs=None, total=0):
    from app.repository.login_log_repository import login_log_repository

    repo = _Repo(docs=docs, total=total)
    monkeypatch.setattr(login_log_repository, "page_logs", repo.page_logs)
    return repo


async def test_admin_sees_all_and_no_user_filter(monkeypatch):
    repo = _patch(monkeypatch, docs=[_make_doc()], total=1)
    result = await auth_service.list_login_logs(1, 10, user=_User(admin=True))
    assert result["total"] == 1
    assert result["list"][0]["username"] == "admin"
    assert result["list"][0]["loginTime"] == "2025-01-01 12:00:00"
    page_num, page_size, kwargs = repo.calls[0]
    assert kwargs["user_ids"] is None


async def test_normal_user_forced_to_own_logs(monkeypatch):
    repo = _patch(monkeypatch, docs=[_make_doc(username="normal")], total=1)
    result = await auth_service.list_login_logs(
        1, 10, username="admin", user=_User(admin=False)
    )
    assert result["total"] == 1
    _, _, kwargs = repo.calls[0]
    assert kwargs["user_ids"] == [2]
    assert kwargs["username"] == "admin"


async def test_empty_result(monkeypatch):
    repo = _patch(monkeypatch)
    result = await auth_service.list_login_logs(
        1, 10, username="nobody", user=_User(admin=True)
    )
    assert result["total"] == 0
    assert result["list"] == []


async def test_time_parse_iso_and_space_formats(monkeypatch):
    repo = _patch(monkeypatch)
    await auth_service.list_login_logs(
        1,
        10,
        start_time="2025-01-01 00:00:00",
        end_time="2025-01-01T23:59:59",
        user=_User(admin=True),
    )
    _, _, kwargs = repo.calls[0]
    assert kwargs["start_time"] is not None
    assert kwargs["end_time"] is not None
