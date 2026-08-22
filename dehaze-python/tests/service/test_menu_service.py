from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.service import menu_service as m
from app.service.menu_service import menu_service


def _menu(**overrides):
    base = {
        "id": 1,
        "parent_id": 0,
        "tree_path": ",",
        "name": "菜单",
        "type": 1,
        "path": "/menu",
        "component": None,
        "perm": None,
        "visible": 1,
        "status": 1,
        "sort": 1,
        "icon": "",
        "redirect": None,
        "always_show": 0,
        "keep_alive": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def stub_menu_repo(monkeypatch):
    class Stub:
        def __init__(self):
            self._parent = None
            self.exists_by_name_calls = []

        async def get_by_id(self, db, menu_id):
            return self._parent

        async def exists_by_name(self, db, parent_id, name, exclude_id=None):
            self.exists_by_name_calls.append((parent_id, name, exclude_id))
            return False

        async def exists_by_perm(self, db, perm, exclude_id=None):
            return False

    stub = Stub()
    monkeypatch.setattr(m.menu_repository, "get_by_id", stub.get_by_id)
    monkeypatch.setattr(m.menu_repository, "exists_by_name", stub.exists_by_name)
    monkeypatch.setattr(m.menu_repository, "exists_by_perm", stub.exists_by_perm)

    def set_name(returns):
        monkeypatch.setattr(m.menu_repository, "exists_by_name", _coro(returns))

    def set_perm(returns):
        monkeypatch.setattr(m.menu_repository, "exists_by_perm", _coro(returns))

    stub.set_name = set_name
    stub.set_perm = set_perm
    return stub


def _save_data(**overrides):
    base = {
        "parentId": 0,
        "name": "测试菜单",
        "type": 1,
        "path": "/test",
        "component": "test/index",
        "perm": None,
        "visible": 1,
        "sort": 1,
    }
    base.update(overrides)
    return base


class TestNameUnique:
    async def test_new_same_name_rejected(self, stub_menu_repo):
        stub_menu_repo.set_name(True)
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, _save_data())
        assert ei.value.code == ResultCode.DATA_EXISTS
        assert "菜单名称已存在" in ei.value.message

    async def test_update_same_name_self_excluded(self, stub_menu_repo):
        await menu_service._validate_menu_form(None, _save_data(), current_id=1)
        assert stub_menu_repo.exists_by_name_calls == [(0, "测试菜单", 1)]


class TestPermUnique:
    async def test_dup_perm_rejected(self, stub_menu_repo):
        stub_menu_repo.set_perm(True)
        data = _save_data(perm="sys:user:add")
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.DATA_EXISTS
        assert "权限标识已存在" in ei.value.message


class TestParentType:
    async def test_parent_is_button_rejected(self, stub_menu_repo):
        stub_menu_repo._parent = _menu(id=5, type=4)
        data = _save_data(parentId=5)
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "按钮" in ei.value.message

    async def test_parent_is_extlink_rejected(self, stub_menu_repo):
        stub_menu_repo._parent = _menu(id=5, type=3)
        data = _save_data(parentId=5)
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "外链" in ei.value.message


class TestConditionalRequired:
    async def test_menu_requires_path(self, stub_menu_repo):
        data = _save_data(type=1, path="")
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "路由地址不能为空" in ei.value.message

    async def test_catalog_requires_path(self, stub_menu_repo):
        data = _save_data(type=2, path="")
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "路由地址不能为空" in ei.value.message

    async def test_button_requires_perm(self, stub_menu_repo):
        data = _save_data(type=4, perm="")
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "权限标识不能为空" in ei.value.message

    async def test_extlink_requires_path(self, stub_menu_repo):
        data = _save_data(type=3, path="")
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "外链地址不能为空" in ei.value.message


class TestDepthLimit:
    async def test_depth5_under_5_rejected(self, stub_menu_repo):
        stub_menu_repo._parent = _menu(id=5, type=2, tree_path=",1,2,3,4,")
        data = _save_data(parentId=5)
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "菜单层级不能超过5级" in ei.value.message

    async def test_depth4_allowed(self, stub_menu_repo):
        stub_menu_repo._parent = _menu(id=5, type=2, tree_path=",1,2,3,")
        await menu_service._validate_menu_form(None, _save_data(parentId=5))


class TestSelfParent:
    async def test_parent_is_self_rejected(self, stub_menu_repo):
        data = _save_data(parentId=9)
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data, current_id=9)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "上级菜单不能是自己" in ei.value.message


class TestCycleDetection:
    async def test_parent_is_own_descendant_rejected(self, stub_menu_repo):
        stub_menu_repo._parent = _menu(id=2, type=2, tree_path=",1,2,")
        data = _save_data(parentId=2)
        with pytest.raises(BusinessException) as ei:
            await menu_service._validate_menu_form(None, data, current_id=1)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "不能设置自己的子菜单为父菜单" in ei.value.message

    async def test_normal_parent_allowed(self, stub_menu_repo):
        stub_menu_repo._parent = _menu(id=5, type=2, tree_path=",")
        await menu_service._validate_menu_form(None, _save_data(parentId=5), current_id=1)


class TestMenuOptionsSkipButton:
    def test_button_not_in_options(self):
        parent = _menu(id=1, type=1)
        button = _menu(id=2, type=4, parent_id=1)
        child = _menu(id=3, type=2, parent_id=1)
        children_map = {
            0: [parent],
            1: [button, child],
        }
        options = menu_service._build_menu_options(0, children_map)
        assert len(options) == 1
        assert options[0]["children"] == [{"value": 3, "label": child.name}]


def _coro(val):
    async def _f(*args, **kwargs):
        return val
    return _f
