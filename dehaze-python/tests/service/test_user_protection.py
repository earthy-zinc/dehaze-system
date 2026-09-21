from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.models.schema.user import UserForm
from app.service.user_service import UserService
from tests.stubs.factories import make_user_context

pytestmark = pytest.mark.requires_db

# 测试替身：仓储层已 mock，db 仅传参占位
_DB: AsyncSession = AsyncMock(spec=AsyncSession)


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
    # 会话踢出联动会经 repo 查角色编码，缺省置空避免 AsyncMock 泄漏到迭代
    returns.setdefault("get_user_role_codes", [])
    for name, value in returns.items():
        getattr(repo, name).return_value = value
    return repo


def _service(**returns):
    return UserService(repo=_stub_user_repo(**returns), audit_repo=Mock())


class TestUserFormValidation:
    def test_email_valid(self):
        form = UserForm(
            username="alice", nickname="Alice", deptId=1, roleIds=[1], email="alice@example.com"
        )
        assert form.email == "alice@example.com"

    def test_email_invalid(self):
        with pytest.raises(ValidationError):
            UserForm(
                username="alice", nickname="Alice", deptId=1, roleIds=[1], email="invalid-email"
            )

    def test_email_empty_allowed(self):
        UserForm(username="alice", nickname="Alice", deptId=1, roleIds=[1], email="")


class TestUserDeleteProtection:
    async def test_self_delete(self, mock_redis):
        current = make_user_context(6, username="alice")
        with pytest.raises(BusinessException) as ei:
            await _service().delete_users(_DB, mock_redis, "5,6", current)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert ei.value.message == "不可删除自己"

    async def test_root_protected(self, mock_redis):
        svc = UserService(repo=_stub_user_repo(get_protected_user_ids=[7]), audit_repo=Mock())
        current = make_user_context(6, username="alice")
        with pytest.raises(BusinessException) as ei:
            await svc.delete_users(_DB, mock_redis, "7", current)
        assert ei.value.code == ResultCode.ROOT_USER_PROTECTED
        assert ei.value.message == "超级管理员不可删除"

    async def test_delete_normal_user_ok(self, mock_redis):
        repo = _stub_user_repo(get_protected_user_ids=[], soft_delete_by_ids=None)
        audit = Mock()
        svc = UserService(repo=repo, audit_repo=audit)
        current = make_user_context(6, username="alice")
        result = await svc.delete_users(_DB, mock_redis, "8", current)
        assert result == {"deleted_count": 1}
        repo.soft_delete_by_ids.assert_awaited_once_with(_DB, [8])
        audit.create_audit_async.assert_called_once()


class TestUsernameReadonly:
    async def test_username_change_rejected(self):
        svc = _service(get_by_id=_User("old"))
        with pytest.raises(BusinessException) as ei:
            await svc.update_user_with_roles(_DB, 1, {"username": "new"})
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert ei.value.message == "用户名不可修改"

    async def test_username_unchanged_ok(self, db):
        user = _User("old", nickname="旧昵称")
        repo = _stub_user_repo(get_by_id=user, replace_user_roles=None)
        svc = UserService(repo=repo, audit_repo=Mock())
        await svc.update_user_with_roles(
            db, 1, {"username": "old", "nickname": "新昵称", "roleIds": [1, 2]}
        )
        assert user.nickname == "新昵称"
        repo.replace_user_roles.assert_awaited_once_with(db, 1, [1, 2])


class TestUpdateUserStatus:
    async def test_disable_root_rejected(self, mock_redis):
        svc = _service(get_by_id=_User("root"))
        with pytest.raises(BusinessException) as ei:
            await svc.update_user_status(_DB, mock_redis, 1, 0)
        assert ei.value.code == ResultCode.ROOT_USER_PROTECTED
        assert ei.value.message == "超级管理员不可禁用"

    async def test_enable_root_allowed(self, mock_redis):
        user = _User("root")
        svc = _service(get_by_id=user)
        await svc.update_user_status(_DB, mock_redis, 1, 1)
        assert user.status == 1

    async def test_disable_self_rejected(self, mock_redis):
        svc = _service(get_by_id=_User("alice"))
        current = make_user_context(6, username="alice")
        with pytest.raises(BusinessException) as ei:
            await svc.update_user_status(_DB, mock_redis, 6, 0, current_user=current)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert ei.value.message == "不可禁用自己"

    async def test_disable_normal_user_ok(self, mock_redis):
        user = _User("normal")
        svc = _service(get_by_id=user)
        await svc.update_user_status(_DB, mock_redis, 1, 0)
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
