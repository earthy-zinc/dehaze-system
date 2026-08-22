import pytest
from types import SimpleNamespace

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.service import role_service as rs
from app.service.role_service import role_service


@pytest.fixture
def stub_repos(monkeypatch):
    class Stub:
        def __init__(self):
            self.get_by_id = None
            self.count_by_ids = 0
            self.replaced = []

    stub = Stub()

    async def _check_name(db, name, exclude_id=None):
        return False

    async def _check_code(db, code, exclude_id=None):
        return False

    async def _create(db, role):
        return role

    async def _replace(db, role_id, menu_ids):
        stub.replaced.append((role_id, menu_ids))

    async def _get_role(db, role_id):
        if not stub.get_by_id:
            return None
        return SimpleNamespace(code="TEST")

    async def _count(db, menu_ids):
        return stub.count_by_ids

    monkeypatch.setattr(rs.role_repository, "check_name_exists", _check_name)
    monkeypatch.setattr(rs.role_repository, "check_code_exists", _check_code)
    monkeypatch.setattr(rs.role_repository, "create", _create)
    monkeypatch.setattr(rs.role_repository, "replace_role_menus", _replace)
    monkeypatch.setattr(rs.role_repository, "get_by_id", _get_role)
    monkeypatch.setattr(rs.menu_repository, "count_by_ids", _count)
    monkeypatch.setattr(mongo_audit_log_repository, "create_audit_async", lambda **kw: None)
    return stub


async def test_create_role_without_data_scope_rejected(stub_repos, mock_redis):
    data = {"name": "测试角色", "code": "TEST_ROLE", "sort": 1, "status": 1}
    with pytest.raises(BusinessException) as ei:
        await role_service.create_role(None, mock_redis, data)
    assert ei.value.code == ResultCode.PARAM_ERROR
    assert "数据权限不能为空" in ei.value.message


async def test_create_role_with_data_scope_ok(stub_repos, mock_redis):
    data = {"name": "测试角色", "code": "TEST_ROLE", "dataScope": 0, "sort": 1, "status": 1}
    created = await role_service.create_role(None, mock_redis, data)
    assert created.code == "TEST_ROLE"
    assert created.data_scope == 0


async def test_assign_menus_rejects_missing_menu(stub_repos, mock_redis):
    stub_repos.get_by_id = True
    stub_repos.count_by_ids = 1
    with pytest.raises(BusinessException) as ei:
        await role_service.assign_menus_to_role(None, mock_redis, 1, [97, 98])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND
    assert "菜单不存在" in ei.value.message


async def test_assign_menus_all_exist_ok(stub_repos, mock_redis):
    stub_repos.get_by_id = True
    stub_repos.count_by_ids = 2
    cache_key = f"{role_service.ROLE_PERMS_PREFIX}TEST"
    await mock_redis.set(cache_key, "1,2")
    await role_service.assign_menus_to_role(None, mock_redis, 1, [97, 98])
    assert stub_repos.replaced == [(1, [97, 98])]
    assert await mock_redis.get(cache_key) is None
