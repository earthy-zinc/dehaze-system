"""
用户服务测试

测试 UserService 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.user_service import UserService, generate_random_password
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestPasswordGeneration:
    """密码生成测试"""

    def test_generate_random_password_length(self):
        """测试随机密码长度"""
        password = generate_random_password(12)
        assert len(password) == 12

    def test_generate_random_password_uniqueness(self):
        """测试随机密码唯一性"""
        passwords = [generate_random_password() for _ in range(100)]
        # 100 个随机密码应该几乎不可能重复
        assert len(set(passwords)) > 95

    def test_generate_random_password_complexity(self):
        """测试随机密码复杂度"""
        password = generate_random_password()
        has_letter = any(c.isalpha() for c in password)
        has_digit = any(c.isdigit() for c in password)
        assert has_letter and has_digit


@pytest.mark.unit
class TestUserService:
    """用户服务测试"""

    @pytest.mark.asyncio
    async def test_create_user_success(self):
        """测试创建用户成功"""
        mock_db = AsyncMock()

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_username = AsyncMock(return_value=None)
            mock_repo.create_user = AsyncMock(return_value=MagicMock(id=1, username="testuser"))

            user = await UserService.create_user_with_roles(
                db=mock_db,
                data={"username": "testuser", "nickname": "Test User"},
            )

            assert user is not None
            mock_repo.get_by_username.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_user_empty_username(self):
        """测试创建用户时用户名为空"""
        mock_db = AsyncMock()

        with pytest.raises(BusinessException, match="用户名不能为空"):
            await UserService.create_user_with_roles(
                db=mock_db,
                data={"username": "", "nickname": "Test"},
            )

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self):
        """测试创建用户时用户名已存在"""
        mock_db = AsyncMock()

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_username = AsyncMock(return_value=MagicMock())

            with pytest.raises(BusinessException, match="用户名已存在"):
                await UserService.create_user_with_roles(
                    db=mock_db,
                    data={"username": "existing_user", "nickname": "Test"},
                )

    @pytest.mark.asyncio
    async def test_update_user_success(self):
        """测试更新用户成功"""
        mock_db = AsyncMock()
        mock_user = MagicMock()
        mock_user.id = 1

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_user)
            mock_repo.replace_user_roles = AsyncMock()

            await UserService.update_user_with_roles(
                db=mock_db,
                user_id=1,
                data={"nickname": "Updated User"},
            )

            mock_repo.get_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_not_found(self):
        """测试更新用户时用户不存在"""
        mock_db = AsyncMock()

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="用户不存在"):
                await UserService.update_user_with_roles(
                    db=mock_db,
                    user_id=999,
                    data={"nickname": "Test"},
                )

    @pytest.mark.asyncio
    async def test_update_password_success(self):
        """测试更新密码成功"""
        mock_db = AsyncMock()
        mock_user = MagicMock()
        mock_user.id = 1

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_user)

            await UserService.update_password(
                db=mock_db,
                user_id=1,
                new_password="newpassword123",
            )

            mock_repo.get_by_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_password_complexity_check(self):
        """测试密码复杂度验证"""
        mock_db = AsyncMock()

        # 测试密码太短
        with pytest.raises(BusinessException, match="密码长度不能少于"):
            await UserService.update_password(
                db=mock_db,
                user_id=1,
                new_password="abc12",
            )

        # 测试密码缺少数字
        with pytest.raises(BusinessException, match="密码必须包含字母和数字"):
            await UserService.update_password(
                db=mock_db,
                user_id=1,
                new_password="abcdefgh",
            )

        # 测试密码缺少字母
        with pytest.raises(BusinessException, match="密码必须包含字母和数字"):
            await UserService.update_password(
                db=mock_db,
                user_id=1,
                new_password="12345678",
            )

    @pytest.mark.asyncio
    async def test_update_password_user_not_found(self):
        """测试更新密码时用户不存在"""
        mock_db = AsyncMock()

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="用户不存在"):
                await UserService.update_password(
                    db=mock_db,
                    user_id=999,
                    new_password="newpassword123",  # 符合复杂度要求
                )

    @pytest.mark.asyncio
    async def test_update_user_status_protect_root(self):
        """测试不能禁用 root 用户"""
        mock_db = AsyncMock()
        mock_user = MagicMock()
        mock_user.username = "root"

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_user)

            with pytest.raises(BusinessException, match="超级管理员不可禁用"):
                await UserService.update_user_status(
                    db=mock_db,
                    user_id=1,
                    status=0,
                )

    @pytest.mark.asyncio
    async def test_delete_users_success(self):
        """测试删除用户成功"""
        mock_db = AsyncMock()

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_protected_user_ids = AsyncMock(return_value=[])
            mock_repo.soft_delete_by_ids = AsyncMock()

            result = await UserService.delete_users(
                db=mock_db,
                ids="1,2,3",
            )

            assert result["deleted_count"] == 3
            assert result["protected_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_users_with_protected(self):
        """测试删除用户时包含受保护用户"""
        mock_db = AsyncMock()

        with patch("app.service.user_service.user_repository") as mock_repo:
            mock_repo.get_protected_user_ids = AsyncMock(return_value=[1])
            mock_repo.soft_delete_by_ids = AsyncMock()

            result = await UserService.delete_users(
                db=mock_db,
                ids="1,2,3",
            )

            assert result["deleted_count"] == 2
            assert result["protected_count"] == 1

    @pytest.mark.asyncio
    async def test_delete_users_empty_ids(self):
        """测试删除用户时 ID 为空"""
        mock_db = AsyncMock()

        with pytest.raises(BusinessException, match="未指定要删除的用户"):
            await UserService.delete_users(
                db=mock_db,
                ids="",
            )

    def test_generate_import_template(self):
        """测试生成导入模板"""
        output = UserService.generate_import_template()
        assert output is not None
        # 验证是有效的 Excel 文件
        import openpyxl
        wb = openpyxl.load_workbook(output)
        assert wb.active is not None
