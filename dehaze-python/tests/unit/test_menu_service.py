"""
菜单服务测试

测试 MenuService 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.menu_service import MenuService, MENU_TYPE_CATALOG, MENU_TYPE_MENU, MENU_TYPE_BUTTON
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestMenuService:
    """菜单服务测试"""

    @pytest.mark.asyncio
    async def test_list_menus(self):
        """测试获取菜单列表"""
        mock_db = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.get_list = AsyncMock(return_value=[])

            result = await MenuService.list_menus(mock_db)

            assert result == []

    @pytest.mark.asyncio
    async def test_list_menu_options(self):
        """测试获取菜单选项"""
        mock_db = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.get_list = AsyncMock(return_value=[])

            result = await MenuService.list_menu_options(mock_db)

            assert result == []

    @pytest.mark.asyncio
    async def test_save_menu_create(self):
        """测试创建菜单"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            with patch.object(MenuService, "_validate_menu_data", new_callable=AsyncMock):
                with patch.object(MenuService, "_clear_menu_cache", new_callable=AsyncMock):
                    mock_repo.create_menu = AsyncMock(return_value=MagicMock(id=1))
                    mock_db.commit = AsyncMock()

                    result = await MenuService.save_menu(
                        db=mock_db,
                        redis=mock_redis,
                        data={"name": "测试菜单", "path": "/test", "type": MENU_TYPE_CATALOG, "parentId": 0},
                    )

                    assert result is not None

    @pytest.mark.asyncio
    async def test_save_menu_update(self):
        """测试更新菜单"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_menu = MagicMock()
        mock_menu.id = 1
        mock_menu.type = MENU_TYPE_CATALOG

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            with patch.object(MenuService, "_validate_menu_data", new_callable=AsyncMock):
                with patch.object(MenuService, "_clear_menu_cache", new_callable=AsyncMock):
                    mock_repo.get_by_id = AsyncMock(return_value=mock_menu)
                    mock_repo.update_menu = AsyncMock(return_value=mock_menu)
                    mock_db.commit = AsyncMock()

                    result = await MenuService.save_menu(
                        db=mock_db,
                        redis=mock_redis,
                        data={"id": 1, "name": "更新菜单", "path": "/test", "type": MENU_TYPE_CATALOG},
                    )

                    assert result is not None

    @pytest.mark.asyncio
    async def test_save_menu_not_found(self):
        """测试更新菜单时菜单不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            with patch.object(MenuService, "_validate_menu_data", new_callable=AsyncMock):
                mock_repo.get_by_id = AsyncMock(return_value=None)

                with pytest.raises(BusinessException, match="菜单不存在"):
                    await MenuService.save_menu(
                        db=mock_db,
                        redis=mock_redis,
                        data={"id": 999, "name": "测试", "path": "/test", "type": MENU_TYPE_CATALOG},
                    )

    @pytest.mark.asyncio
    async def test_save_menu_name_duplicate(self):
        """测试创建菜单时名称重复"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=True)

            with pytest.raises(BusinessException, match="同一父级下菜单名称已存在"):
                await MenuService.save_menu(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "重复菜单", "type": MENU_TYPE_CATALOG, "parentId": 0},
                )

    @pytest.mark.asyncio
    async def test_save_menu_button_as_parent(self):
        """测试按钮类型作为父级菜单"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_parent = MagicMock()
            mock_parent.type = MENU_TYPE_BUTTON
            mock_repo.get_by_id = AsyncMock(return_value=mock_parent)
            mock_repo.check_name_exists = AsyncMock(return_value=False)

            with pytest.raises(BusinessException, match="按钮类型不能作为父级菜单"):
                await MenuService.save_menu(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "测试菜单", "type": MENU_TYPE_MENU, "parentId": 1},
                )

    @pytest.mark.asyncio
    async def test_save_menu_missing_path(self):
        """测试菜单类型缺少路由地址"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=False)

            with pytest.raises(BusinessException, match="菜单类型必须配置路由地址"):
                await MenuService.save_menu(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "测试菜单", "type": MENU_TYPE_MENU, "parentId": 0},
                )

    @pytest.mark.asyncio
    async def test_save_menu_missing_perm_for_button(self):
        """测试按钮类型缺少权限标识"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_parent = MagicMock()
            mock_parent.type = MENU_TYPE_CATALOG  # 父级是目录类型，允许作为父级
            mock_repo.check_name_exists = AsyncMock(return_value=False)
            mock_repo.get_by_id = AsyncMock(return_value=mock_parent)

            with pytest.raises(BusinessException, match="按钮类型必须配置权限标识"):
                await MenuService.save_menu(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "测试按钮", "type": MENU_TYPE_BUTTON, "parentId": 1},
                )

    @pytest.mark.asyncio
    async def test_list_routes(self):
        """测试获取路由列表"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            with patch("app.service.menu_service.CacheService") as mock_cache_class:
                mock_cache = AsyncMock()
                mock_cache.get_json = AsyncMock(return_value=None)
                mock_cache.set_json = AsyncMock()
                mock_cache_class.return_value = mock_cache

                mock_repo.get_route_menus = AsyncMock(return_value=[])

                result = await MenuService.list_routes(mock_db, mock_redis)

                assert result == []

    @pytest.mark.asyncio
    async def test_list_routes_from_cache(self):
        """测试从缓存获取路由列表"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        cached_routes = [{"name": "System", "path": "/system"}]

        with patch("app.service.menu_service.CacheService") as mock_cache_class:
            mock_cache = AsyncMock()
            mock_cache.get_json = AsyncMock(return_value=cached_routes)
            mock_cache_class.return_value = mock_cache

            result = await MenuService.list_routes(mock_db, mock_redis)

            assert result == cached_routes

    @pytest.mark.asyncio
    async def test_update_menu_visible_success(self):
        """测试更新菜单显示状态"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_menu = MagicMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            with patch.object(MenuService, "_clear_menu_cache", new_callable=AsyncMock):
                mock_repo.get_by_id = AsyncMock(return_value=mock_menu)
                mock_db.commit = AsyncMock()

                await MenuService.update_menu_visible(
                    db=mock_db,
                    redis=mock_redis,
                    menu_id=1,
                    visible=0,
                )

    @pytest.mark.asyncio
    async def test_update_menu_visible_invalid(self):
        """测试更新菜单显示状态值无效"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="显示状态只能为0或1"):
            await MenuService.update_menu_visible(
                db=mock_db,
                redis=mock_redis,
                menu_id=1,
                visible=2,
            )

    @pytest.mark.asyncio
    async def test_update_menu_visible_not_found(self):
        """测试更新菜单显示状态时菜单不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="菜单不存在"):
                await MenuService.update_menu_visible(
                    db=mock_db,
                    redis=mock_redis,
                    menu_id=999,
                    visible=0,
                )

    @pytest.mark.asyncio
    async def test_get_menu_form(self):
        """测试获取菜单表单数据"""
        mock_db = AsyncMock()
        mock_menu = MagicMock()
        mock_menu.id = 1
        mock_menu.parent_id = 0
        mock_menu.name = "测试菜单"
        mock_menu.type = MENU_TYPE_CATALOG
        mock_menu.path = "/test"
        mock_menu.component = "Test"
        mock_menu.perm = "test:list"
        mock_menu.visible = 1
        mock_menu.sort = 1
        mock_menu.icon = "test"
        mock_menu.redirect = None
        mock_menu.always_show = 0
        mock_menu.keep_alive = 0

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_menu)

            result = await MenuService.get_menu_form(mock_db, 1)

            assert result is not None
            assert result["name"] == "测试菜单"

    @pytest.mark.asyncio
    async def test_get_menu_form_not_found(self):
        """测试获取菜单表单数据时菜单不存在"""
        mock_db = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            result = await MenuService.get_menu_form(mock_db, 999)

            assert result is None

    @pytest.mark.asyncio
    async def test_delete_menu_success(self):
        """测试删除菜单"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_menu = MagicMock()
        mock_menu.id = 1

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            with patch.object(MenuService, "_clear_menu_cache", new_callable=AsyncMock):
                mock_repo.get_by_id = AsyncMock(return_value=mock_menu)
                mock_repo.delete_role_menus_by_menu_id = AsyncMock()
                mock_repo.delete_menu_and_children = AsyncMock()
                mock_db.commit = AsyncMock()

                await MenuService.delete_menu(mock_db, mock_redis, 1)

                mock_repo.delete_role_menus_by_menu_id.assert_called_once()
                mock_repo.delete_menu_and_children.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_menu_not_found(self):
        """测试删除菜单时菜单不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.menu_service.menu_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="菜单不存在"):
                await MenuService.delete_menu(mock_db, mock_redis, 999)


@pytest.mark.unit
class TestMenuTreeBuilding:
    """菜单树构建测试"""

    def test_build_empty_tree(self):
        """测试构建空树"""
        result = MenuService._build_menu_tree(0, [])
        assert result == []

    def test_build_single_level_tree(self):
        """测试构建单层树"""
        class MockMenu:
            def __init__(self, id, name, parent_id, type, path, component, perm, visible, sort, icon, redirect, always_show, keep_alive, create_time):
                self.id = id
                self.name = name
                self.parent_id = parent_id
                self.type = type
                self.path = path
                self.component = component
                self.perm = perm
                self.visible = visible
                self.sort = sort
                self.icon = icon
                self.redirect = redirect
                self.always_show = always_show
                self.keep_alive = keep_alive
                self.create_time = create_time

        mock_menus = [
            MockMenu(1, "系统管理", 0, MENU_TYPE_CATALOG, "/system", "Layout", None, 1, 1, "system", None, 0, 0, None),
        ]

        result = MenuService._build_menu_tree(0, mock_menus)

        assert len(result) == 1
        assert result[0]["name"] == "系统管理"

    def test_build_multi_level_tree(self):
        """测试构建多层树"""
        class MockMenu:
            def __init__(self, id, name, parent_id, type, path, component, perm, visible, sort, icon, redirect, always_show, keep_alive, create_time):
                self.id = id
                self.name = name
                self.parent_id = parent_id
                self.type = type
                self.path = path
                self.component = component
                self.perm = perm
                self.visible = visible
                self.sort = sort
                self.icon = icon
                self.redirect = redirect
                self.always_show = always_show
                self.keep_alive = keep_alive
                self.create_time = create_time

        mock_menus = [
            MockMenu(1, "系统管理", 0, MENU_TYPE_CATALOG, "/system", "Layout", None, 1, 1, "system", None, 0, 0, None),
            MockMenu(2, "用户管理", 1, MENU_TYPE_MENU, "/system/user", "system/user/index", "system:user:list", 1, 1, "user", None, 0, 0, None),
            MockMenu(3, "角色管理", 1, MENU_TYPE_MENU, "/system/role", "system/role/index", "system:role:list", 1, 2, "role", None, 0, 0, None),
        ]

        result = MenuService._build_menu_tree(0, mock_menus)

        assert len(result) == 1
        assert result[0]["name"] == "系统管理"
        assert len(result[0]["children"]) == 2
