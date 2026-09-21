import pytest
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service import dept_service as ds
from app.service.dept_service import dept_service


def _dept(**overrides):
    base = {"id": 1, "name": "部门", "parent_id": 0, "tree_path": "0", "sort": 1, "status": 1}
    base.update(overrides)
    return type("D", (), base)()


class _FakeDB(AsyncSession):
    """AsyncSession 替身：add 分配自增 id、flush/refresh 为 no-op（不发真实 SQL）。"""

    def add(self, obj, *args, **kwargs):
        self._obj = obj
        obj.id = 1

    async def flush(self, *args, **kwargs):
        return None

    async def refresh(self, obj, *args, **kwargs):
        return None


@pytest.fixture
def dept_env(monkeypatch):
    class Stub:
        name_exists = False
        get_by_id = None
        tree_path = None
        child_counts = {}
        user_counts = {}
        dept_list = []
        soft_delete_rows = 0

    stub = Stub()

    async def _check_name(db, name, *a, **k):
        return stub.name_exists

    async def _get_by_id(db, dept_id, *a, **k):
        if isinstance(stub.get_by_id, dict):
            return stub.get_by_id.get(dept_id)
        return stub.get_by_id

    async def _tree_path(db, parent_id, *a, **k):
        return stub.tree_path

    async def _children(db, parent_ids, *a, **k):
        return stub.child_counts

    async def _get_ids(db, dept_ids, *a, **k):
        return stub.dept_list

    async def _soft(db, dept_ids, *a, **k):
        return stub.soft_delete_rows

    async def _users(db, dept_ids, *a, **k):
        return stub.user_counts

    async def _noop_clear(redis):
        return None

    monkeypatch.setattr(ds.dept_repository, "check_name_exists", _check_name)
    monkeypatch.setattr(ds.dept_repository, "get_by_id", _get_by_id)
    monkeypatch.setattr(ds.dept_repository, "generate_tree_path", _tree_path)
    monkeypatch.setattr(ds.dept_repository, "count_children_by_parents", _children)
    monkeypatch.setattr(ds.dept_repository, "get_by_ids", _get_ids)
    monkeypatch.setattr(ds.dept_repository, "soft_delete_by_ids", _soft)
    monkeypatch.setattr(ds.user_repository, "count_users_by_depts", _users)
    monkeypatch.setattr(dept_service, "_clear_cache", _noop_clear)
    return stub


async def test_create_dept_exceeds_5_levels_rejected(dept_env):
    dept_env.get_by_id = _dept(id=9, name="父", tree_path="0,1,2,3,4")
    dept_env.tree_path = "0,1,2,3,4,9"
    with pytest.raises(BusinessException) as ei:
        await dept_service.create_dept(_FakeDB(), Redis(), {"name": "子", "parentId": 9})
    assert ei.value.code == ResultCode.DATA_BIND_EXISTS
    assert "部门层级不能超过5级" in ei.value.message


async def test_create_dept_name_duplicated_rejected(dept_env):
    dept_env.get_by_id = _dept(id=9, name="父", tree_path="0,1")
    dept_env.name_exists = True
    with pytest.raises(BusinessException) as ei:
        await dept_service.create_dept(_FakeDB(), Redis(), {"name": "重名", "parentId": 9})
    assert ei.value.code == ResultCode.DATA_EXISTS
    assert "部门名称已存在" in ei.value.message


async def test_create_dept_parent_not_found_rejected(dept_env):
    dept_env.get_by_id = None
    with pytest.raises(BusinessException) as ei:
        await dept_service.create_dept(_FakeDB(), Redis(), {"name": "子", "parentId": 999})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_create_dept_at_5_level_ok(dept_env):
    dept_env.get_by_id = _dept(id=9, name="父", tree_path="0,1,2,3")
    dept_env.tree_path = "0,1,2,3,9"
    created_id = await dept_service.create_dept(_FakeDB(), Redis(), {"name": "子", "parentId": 9})
    assert created_id is not None


async def test_update_root_dept_parent_rejected(dept_env):
    dept_env.get_by_id = _dept(id=1, name="根部门", tree_path="0")
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(AsyncSession(), Redis(), 1, {"parentId": 20})
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert "根部门不可修改上级" in ei.value.message


async def test_update_dept_exceeds_5_levels_rejected(dept_env):
    dept_env.get_by_id = {
        7: _dept(id=7, name="部门", tree_path="0,7"),
        20: _dept(id=20, name="目标", parent_id=5, tree_path="0,2,3,4,5"),
    }
    dept_env.tree_path = "0,2,3,4,5,20"
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(AsyncSession(), Redis(), 7, {"parentId": 20})
    assert ei.value.code == ResultCode.DATA_BIND_EXISTS
    assert "部门层级不能超过5级" in ei.value.message


async def test_update_dept_not_found_rejected(dept_env):
    dept_env.get_by_id = None
    with pytest.raises(BusinessException) as ei:
        await dept_service.update_dept(AsyncSession(), Redis(), 999, {"name": "x"})
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_delete_dept_with_children_rejected(dept_env):
    dept_env.dept_list = [_dept(id=2, name="部门B")]
    dept_env.child_counts = {2: 1}
    dept_env.soft_delete_rows = 0
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(AsyncSession(), Redis(), [2])
    assert ei.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    assert "该部门下存在子部门，请先删除子部门" in ei.value.message


async def test_delete_dept_with_users_rejected(dept_env):
    dept_env.dept_list = [_dept(id=2, name="部门B")]
    dept_env.child_counts = {}
    dept_env.user_counts = {2: 3}
    dept_env.soft_delete_rows = 0
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(AsyncSession(), Redis(), [2])
    assert ei.value.code == ResultCode.DATA_STATE_NOT_ALLOW
    assert "该部门下存在用户，无法删除" in ei.value.message


async def test_delete_dept_no_children_no_users_ok(dept_env):
    dept_env.dept_list = [_dept(id=2, name="部门B")]
    dept_env.child_counts = {}
    dept_env.user_counts = {}
    dept_env.soft_delete_rows = 1
    result = await dept_service.delete_depts(AsyncSession(), Redis(), [2])
    assert result is None
    assert dept_env.soft_delete_rows == 1


async def test_delete_dept_not_found_rejected(dept_env):
    dept_env.dept_list = []
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(AsyncSession(), Redis(), [999])
    assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND


async def test_delete_root_dept_rejected(dept_env):
    dept_env.dept_list = [_dept(id=1, name="根部门", tree_path="0")]
    with pytest.raises(BusinessException) as ei:
        await dept_service.delete_depts(AsyncSession(), Redis(), [1])
    assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
    assert "根部门不可删除" in ei.value.message
