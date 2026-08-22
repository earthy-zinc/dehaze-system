from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.user import UserForm
from app.service import user_service as m
from app.service.user_service import user_service
from tests.stubs import StubAsyncSession


class _User:
    def __init__(self, username: str, nickname: str | None = None, status: int = 1):
        self.id = 1
        self.username = username
        self.nickname = nickname or username
        self.gender = None
        self.dept_id = None
        self.mobile = None
        self.email = None
        self.status = status


def _stub_user_repo(**returns) -> AsyncMock:
    repo = AsyncMock()
    for name, value in returns.items():
        getattr(repo, name).return_value = value
    return repo


class TestUserFormValidation:
    def test_email_valid(self):
        form = UserForm(
            username="alice", nickname="Alice", deptId=1, roleIds=[1], email="alice@example.com"
        )
        assert form.email == "alice@example.com"

    def test_email_invalid(self):
        with pytest.raises(ValidationError):
            UserForm(username="alice", nickname="Alice", deptId=1, roleIds=[1], email="invalid-email")

    def test_email_empty_allowed(self):
        UserForm(username="alice", nickname="Alice", deptId=1, roleIds=[1], email="")

    def test_dept_required(self):
        with pytest.raises(ValidationError):
            UserForm(username="alice", nickname="Alice", roleIds=[1])


class TestUserDeleteProtection:
    async def test_self_delete(self):
        current = _User("alice")
        current.id = 6
        with pytest.raises(BusinessException) as ei:
            await user_service.delete_users(None, "5,6", current)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert ei.value.message == "不可删除自己"

    async def test_root_protected(self, monkeypatch):
        monkeypatch.setattr(
            m, "user_repository", _stub_user_repo(get_protected_user_ids=[7])
        )
        current = _User("alice")
        current.id = 6
        with pytest.raises(BusinessException) as ei:
            await user_service.delete_users(None, "7", current)
        assert ei.value.code == ResultCode.ROOT_USER_PROTECTED
        assert ei.value.message == "超级管理员不可删除"

    async def test_delete_normal_user_ok(self, monkeypatch):
        repo = _stub_user_repo(get_protected_user_ids=[], soft_delete_by_ids=None)
        monkeypatch.setattr(m, "user_repository", repo)
        audit = Mock()
        monkeypatch.setattr(m, "mongo_audit_log_repository", audit)
        current = _User("alice")
        current.id = 6
        result = await user_service.delete_users(None, "8", current)
        assert result == {"deleted_count": 1, "protected_count": 0}
        repo.soft_delete_by_ids.assert_awaited_once_with(None, [8])
        audit.create_audit_async.assert_called_once()


class TestUsernameReadonly:
    async def test_username_change_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "user_repository", _stub_user_repo(get_by_id=_User("old")))
        with pytest.raises(BusinessException) as ei:
            await user_service.update_user_with_roles(None, 1, {"username": "new"})
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert ei.value.message == "用户名不可修改"

    async def test_username_unchanged_ok(self, monkeypatch):
        user = _User("old", nickname="旧昵称")
        repo = _stub_user_repo(get_by_id=user, replace_user_roles=None)
        monkeypatch.setattr(m, "user_repository", repo)
        db = StubAsyncSession()
        await user_service.update_user_with_roles(
            db, 1, {"username": "old", "nickname": "新昵称", "roleIds": [1, 2]}
        )
        assert user.nickname == "新昵称"
        repo.replace_user_roles.assert_awaited_once_with(db, 1, [1, 2])


class TestUpdateUserStatus:
    async def test_disable_root_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "user_repository", _stub_user_repo(get_by_id=_User("root")))
        with pytest.raises(BusinessException) as ei:
            await user_service.update_user_status(None, 1, 0)
        assert ei.value.code == ResultCode.ROOT_USER_PROTECTED
        assert ei.value.message == "超级管理员不可禁用"

    async def test_disable_normal_user_ok(self, monkeypatch):
        user = _User("normal")
        monkeypatch.setattr(m, "user_repository", _stub_user_repo(get_by_id=user))
        await user_service.update_user_status(None, 1, 0)
        assert user.status == 0


class TestAuditAsyncWrite:
    async def test_create_audit_async_holds_task_reference_and_writes(self, monkeypatch):
        import asyncio

        from app.repository import mongo_audit_log_repository as repo_mod

        written = []

        async def fake_create_audit(self, **kwargs):
            written.append(kwargs)

        monkeypatch.setattr(repo_mod.MongoAuditLogRepository, "create_audit", fake_create_audit)
        repo_mod.mongo_audit_log_repository.create_audit_async(
            operator_id=1, target_type="user", target_id="8", action="delete", module="user"
        )
        assert len(repo_mod._BACKGROUND_AUDIT_TASKS) == 1
        await asyncio.sleep(0)
        assert written[0]["action"] == "delete"
        await asyncio.sleep(0)
        assert len(repo_mod._BACKGROUND_AUDIT_TASKS) == 0

    async def test_create_audit_async_swallows_write_failure(self, monkeypatch, caplog):
        import asyncio
        import logging

        from app.repository import mongo_audit_log_repository as repo_mod

        async def boom_create_audit(self, **kwargs):
            raise RuntimeError("mongo down")

        monkeypatch.setattr(repo_mod.MongoAuditLogRepository, "create_audit", boom_create_audit)
        with caplog.at_level(logging.WARNING):
            repo_mod.mongo_audit_log_repository.create_audit_async(
                operator_id=1, target_type="user", target_id="8", action="delete", module="user"
            )
            await asyncio.sleep(0)
        assert any("审计日志写入失败" in r.message for r in caplog.records)
