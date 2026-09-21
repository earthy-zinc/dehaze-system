"""菜单服务层单元测试（纯逻辑校验，真实 DB 行为见 test_menu_service_db.py）。

对应菜单管理测试用例.md：T-MM-015~032（表单校验）、可见性参数校验（T-MM-045 前置）、
路由 VO 构建语义（T-MM-058/059/060 meta 字段）。
"""

from types import SimpleNamespace

import pytest

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.repository.role_repository import RoleRepository
from app.service.menu_service import MenuService


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


class StubMenuRepository:
    def __init__(self):
        self.parent = None
        self.name_exists = False
        self.perm_exists = False
        self.name_calls = []

    async def get_by_id(self, db, menu_id):
        return self.parent

    async def exists_by_name(self, db, parent_id, name, exclude_id=None):
        self.name_calls.append((parent_id, name, exclude_id))
        return self.name_exists

    async def exists_by_perm(self, db, perm, exclude_id=None):
        return self.perm_exists


@pytest.fixture
def stub_repo():
    return StubMenuRepository()


@pytest.fixture
def service(stub_repo):
    return MenuService(menu_repository=stub_repo, role_repository=RoleRepository())


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
    async def test_new_same_name_rejected(self, service, stub_repo):
        stub_repo.name_exists = True
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, _save_data())
        assert ei.value.code == ResultCode.DATA_EXISTS
        assert "菜单名称已存在" in ei.value.message

    async def test_update_same_name_self_excluded(self, service, stub_repo):
        await service._validate_menu_form(None, _save_data(), current_id=1)
        assert stub_repo.name_calls == [(0, "测试菜单", 1)]


class TestPermUnique:
    async def test_dup_perm_rejected(self, service, stub_repo):
        stub_repo.perm_exists = True
        data = _save_data(perm="sys:user:add")
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.DATA_EXISTS
        assert "权限标识已存在" in ei.value.message

    async def test_empty_perm_skips_unique_check(self, service, stub_repo):
        """perm 为空（目录/菜单类型）不应触发唯一性校验"""
        await service._validate_menu_form(None, _save_data(type=2, perm=None))
        assert stub_repo.name_calls  # 走到了重名校验之后


class TestParentType:
    async def test_parent_is_button_rejected(self, service, stub_repo):
        stub_repo.parent = _menu(id=5, type=4)
        data = _save_data(parentId=5)
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "按钮" in ei.value.message

    async def test_parent_is_extlink_rejected(self, service, stub_repo):
        stub_repo.parent = _menu(id=5, type=3)
        data = _save_data(parentId=5)
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "外链" in ei.value.message

    async def test_parent_not_found_rejected(self, service, stub_repo):
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, _save_data(parentId=999))
        assert ei.value.code == ResultCode.PARAM_ERROR
        assert "父菜单不存在" in ei.value.message


class TestConditionalRequired:
    async def test_menu_requires_path(self, service):
        data = _save_data(type=1, path="")
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "路由地址不能为空" in ei.value.message

    async def test_catalog_requires_path(self, service):
        data = _save_data(type=2, path="")
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "路由地址不能为空" in ei.value.message

    async def test_button_requires_perm(self, service):
        data = _save_data(type=4, perm="")
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "权限标识不能为空" in ei.value.message

    async def test_extlink_requires_path(self, service):
        data = _save_data(type=3, path="")
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "外链地址不能为空" in ei.value.message


class TestDepthLimit:
    async def test_depth5_under_5_rejected(self, service, stub_repo):
        stub_repo.parent = _menu(id=5, type=2, tree_path=",1,2,3,4,")
        data = _save_data(parentId=5)
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "菜单层级不能超过5级" in ei.value.message

    async def test_depth4_allowed(self, service, stub_repo):
        stub_repo.parent = _menu(id=5, type=2, tree_path=",1,2,3,")
        await service._validate_menu_form(None, _save_data(parentId=5))


class TestSelfParent:
    async def test_parent_is_self_rejected(self, service):
        data = _save_data(parentId=9)
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data, current_id=9)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "上级菜单不能是自己" in ei.value.message


class TestCycleDetection:
    async def test_parent_is_own_descendant_rejected(self, service, stub_repo):
        stub_repo.parent = _menu(id=2, type=2, tree_path=",1,2,")
        data = _save_data(parentId=2)
        with pytest.raises(BusinessException) as ei:
            await service._validate_menu_form(None, data, current_id=1)
        assert ei.value.code == ResultCode.OPERATION_NOT_ALLOW
        assert "不能设置自己的子菜单为父菜单" in ei.value.message

    async def test_normal_parent_allowed(self, service, stub_repo):
        stub_repo.parent = _menu(id=5, type=2, tree_path=",")
        await service._validate_menu_form(None, _save_data(parentId=5), current_id=1)


class TestMenuOptionsSkipButton:
    def test_button_not_in_options(self, service):
        parent = _menu(id=1, type=1)
        button = _menu(id=2, type=4, parent_id=1)
        child = _menu(id=3, type=2, parent_id=1)
        children_map = {
            0: [parent],
            1: [button, child],
        }
        options = service._build_menu_options(0, children_map)
        assert len(options) == 1
        assert options[0]["children"] == [{"value": 3, "label": child.name}]


class TestVisibleParam:
    async def test_visible_invalid_rejected(self, service):
        with pytest.raises(BusinessException) as ei:
            await service.update_menu_visible(None, None, 1, 2)
        assert ei.value.code == ResultCode.PARAM_ERROR
        assert "显示状态只能为0或1" in ei.value.message

    async def test_visible_missing_menu_rejected(self, service, stub_repo):
        stub_repo.parent = None
        with pytest.raises(BusinessException) as ei:
            await service.update_menu_visible(None, None, 999, 1)
        assert ei.value.code == ResultCode.RESOURCE_NOT_FOUND
        assert "菜单不存在" in ei.value.message


class TestRouteVoBuilding:
    def test_hidden_when_invisible(self, service):
        route = service._to_route_vo(_menu(visible=0), set())
        assert route["meta"]["hidden"] is True

    def test_not_hidden_when_visible(self, service):
        route = service._to_route_vo(_menu(visible=1), set())
        assert route["meta"]["hidden"] is False

    def test_keep_alive_only_for_menu_type(self, service):
        route = service._to_route_vo(_menu(type=1, keep_alive=1), set())
        assert route["meta"]["keepAlive"] is True
        route = service._to_route_vo(_menu(type=2, keep_alive=1), set())
        assert "keepAlive" not in route["meta"]

    def test_always_show_only_for_catalog(self, service):
        route = service._to_route_vo(_menu(type=2, always_show=1), set())
        assert route["meta"]["alwaysShow"] is True
        route = service._to_route_vo(_menu(type=1, always_show=1), set())
        assert "alwaysShow" not in route["meta"]

    def test_route_meta_carries_title_and_icon(self, service):
        route = service._to_route_vo(_menu(name="系统管理", icon="setting"), set())
        assert route["meta"]["title"] == "系统管理"
        assert route["meta"]["icon"] == "setting"

    def test_route_meta_carries_roles(self, service):
        route = service._to_route_vo(_menu(), {"ADMIN", "ROOT"})
        assert route["meta"]["roles"] == ["ADMIN", "ROOT"]
        route = service._to_route_vo(_menu(), set())
        assert route["meta"]["roles"] == []

    def test_route_tree_nesting(self, service):
        parent = _menu(id=1, type=2, path="/p")
        child = _menu(id=2, type=1, path="/p/c", parent_id=1)
        routes = service._build_routes(0, {0: [parent], 1: [child]}, {2: {"ROOT"}})
        assert routes[0]["path"] == "/p"
        assert routes[0]["children"][0]["path"] == "/p/c"
        assert routes[0]["meta"]["roles"] == []
        assert routes[0]["children"][0]["meta"]["roles"] == ["ROOT"]
