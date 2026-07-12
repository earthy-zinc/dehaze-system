"""
角色服务测试

测试 RoleService 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.role_service import RoleService
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestRoleService:
    """角色服务测试"""

    @pytest.mark.asyncio
    async def test_get_role_list(self):
        """测试获取角色列表"""
        mock_db = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_list = AsyncMock(return_value=([], 0))

            roles, total = await RoleService.get_role_list(
                db=mock_db,
                page=1,
                page_size=10,
            )

            assert roles == []
            assert total == 0

    @pytest.mark.asyncio
    async def test_get_role_options_with_cache(self):
        """测试获取角色选项（非超级管理员，不显示 ROOT 角色）"""
        mock_db = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_role_options = AsyncMock(return_value=[{"id": 1, "name": "Admin"}])

            options = await RoleService.get_role_options(mock_db, is_root=False)

            assert len(options) == 1
            assert options[0]["name"] == "Admin"

    @pytest.mark.asyncio
    async def test_get_role_options_without_cache(self):
        """测试获取角色选项（超级管理员，显示 ROOT 角色）"""
        mock_db = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_role_options = AsyncMock(return_value=[{"id": 1, "name": "ROOT"}, {"id": 2, "name": "Admin"}])

            options = await RoleService.get_role_options(mock_db, is_root=True)

            assert len(options) == 2

    @pytest.mark.asyncio
    async def test_create_role_success(self):
        """测试创建角色成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=False)
            mock_repo.check_code_exists = AsyncMock(return_value=False)
            mock_repo.create = AsyncMock(return_value=MagicMock(id=1))

            role = await RoleService.create_role(
                db=mock_db,
                redis=mock_redis,
                data={"name": "测试角色", "code": "TEST_ROLE"},
            )

            assert role is not None

    @pytest.mark.asyncio
    async def test_create_role_empty_params(self):
        """测试创建角色时参数为空"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="角色名称和编码不能为空"):
            await RoleService.create_role(
                db=mock_db,
                redis=mock_redis,
                data={"name": "", "code": ""},
            )

    @pytest.mark.asyncio
    async def test_create_role_invalid_code(self):
        """测试创建角色时编码格式错误"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="角色编码格式错误"):
            await RoleService.create_role(
                db=mock_db,
                redis=mock_redis,
                data={"name": "测试角色", "code": "invalid-code"},
            )

    @pytest.mark.asyncio
    async def test_create_role_duplicate_name(self):
        """测试创建角色时名称已存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=True)

            with pytest.raises(BusinessException, match="角色名称已存在"):
                await RoleService.create_role(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "Admin", "code": "ADMIN"},
                )

    @pytest.mark.asyncio
    async def test_create_role_duplicate_code(self):
        """测试创建角色时编码已存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.check_name_exists = AsyncMock(return_value=False)
            mock_repo.check_code_exists = AsyncMock(return_value=True)

            with pytest.raises(BusinessException, match="角色编码已存在"):
                await RoleService.create_role(
                    db=mock_db,
                    redis=mock_redis,
                    data={"name": "Admin", "code": "ADMIN"},
                )

    @pytest.mark.asyncio
    async def test_update_role_success(self):
        """测试更新角色成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.code = "TEST"
        mock_role.sort = 1
        mock_role.status = 1
        mock_role.data_scope = 1

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)
            mock_repo.check_name_exists = AsyncMock(return_value=False)
            mock_repo.update_by_id = AsyncMock()

            await RoleService.update_role(
                db=mock_db,
                redis=mock_redis,
                role_id=1,
                data={"name": "Updated Role"},
            )

    @pytest.mark.asyncio
    async def test_update_role_not_found(self):
        """测试更新角色时角色不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="角色不存在"):
                await RoleService.update_role(
                    db=mock_db,
                    redis=mock_redis,
                    role_id=999,
                    data={"name": "Test"},
                )

    @pytest.mark.asyncio
    async def test_update_role_duplicate_name(self):
        """测试更新角色时名称已存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.id = 1
        mock_role.code = "TEST"

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)
            mock_repo.check_name_exists = AsyncMock(return_value=True)

            with pytest.raises(BusinessException, match="角色名称已存在"):
                await RoleService.update_role(
                    db=mock_db,
                    redis=mock_redis,
                    role_id=1,
                    data={"name": "Existing Name"},
                )

    @pytest.mark.asyncio
    async def test_delete_role_protect_root(self):
        """测试不能删除 ROOT 角色"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.code = "ROOT"
        mock_role.name = "超级管理员"

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)

            with pytest.raises(BusinessException, match="超级管理员角色不可删除"):
                await RoleService.delete_roles(
                    db=mock_db,
                    redis=mock_redis,
                    ids="1",
                )

    @pytest.mark.asyncio
    async def test_delete_role_assigned_to_user(self):
        """测试删除已分配给用户的角色"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.code = "USER"
        mock_role.name = "普通用户"

        with patch("app.service.role_service.role_repository") as mock_repo, \
             patch("app.service.role_service.user_repository") as mock_user_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)
            mock_user_repo.count_users_by_role = AsyncMock(return_value=5)

            with pytest.raises(BusinessException, match="已分配给用户"):
                await RoleService.delete_roles(
                    db=mock_db,
                    redis=mock_redis,
                    ids="1",
                )

    @pytest.mark.asyncio
    async def test_delete_role_success(self):
        """测试删除角色成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.code = "USER"
        mock_role.name = "普通用户"

        with patch("app.service.role_service.role_repository") as mock_repo, \
             patch("app.service.role_service.user_repository") as mock_user_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)
            mock_repo.delete_role_menus = AsyncMock()
            mock_repo.delete = AsyncMock()
            mock_user_repo.count_users_by_role = AsyncMock(return_value=0)
            mock_db.commit = AsyncMock()

            await RoleService.delete_roles(
                db=mock_db,
                redis=mock_redis,
                ids="1",
            )

            mock_repo.delete_role_menus.assert_called_once()
            mock_repo.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_role_status_protect_root(self):
        """测试不能禁用 ROOT 角色"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.code = "ROOT"

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)

            with pytest.raises(BusinessException, match="超级管理员角色不可禁用"):
                await RoleService.update_role_status(
                    db=mock_db,
                    redis=mock_redis,
                    role_id=1,
                    status=0,
                )

    @pytest.mark.asyncio
    async def test_update_role_status_success(self):
        """测试更新角色状态成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.code = "USER"

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)
            mock_repo.update_by_id = AsyncMock()

            await RoleService.update_role_status(
                db=mock_db,
                redis=mock_redis,
                role_id=1,
                status=0,
            )

    @pytest.mark.asyncio
    async def test_assign_menus_to_role(self):
        """测试分配菜单给角色"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_role = MagicMock()
        mock_role.code = "TEST"

        with patch("app.service.role_service.role_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_role)
            mock_repo.replace_role_menus = AsyncMock()
            mock_db.commit = AsyncMock()

            await RoleService.assign_menus_to_role(
                db=mock_db,
                redis=mock_redis,
                role_id=1,
                menu_ids=[1, 2, 3],
            )

            mock_repo.replace_role_menus.assert_called_once()
