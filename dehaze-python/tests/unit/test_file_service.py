"""
文件服务测试

测试 FileService 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.file_service import FileService, calculate_bytes_md5, generate_object_name
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestFileUtils:
    """文件工具函数测试"""

    def test_calculate_bytes_md5(self):
        """测试计算 MD5"""
        content = b"Hello, World!"
        md5 = calculate_bytes_md5(content)
        assert len(md5) == 32  # MD5 是 32 位十六进制
        assert md5 == "65a8e27d8879283831b664bd8b7f0ad4"

    def test_generate_object_name(self):
        """测试生成对象名称"""
        md5 = "abc123"
        extension = "jpg"
        object_name = generate_object_name(md5, extension)
        assert md5 in object_name
        assert object_name.endswith(".jpg")
        assert "upload/" in object_name


@pytest.mark.unit
class TestFileService:
    """文件服务测试"""

    @pytest.mark.asyncio
    async def test_upload_file_new(self):
        """测试上传新文件"""
        mock_db = AsyncMock()

        with patch("app.service.file_service.file_repository") as mock_repo, \
             patch("app.service.file_service.get_minio_client") as mock_minio:
            mock_repo.get_by_md5 = AsyncMock(return_value=None)
            mock_repo.create = AsyncMock(return_value=MagicMock(id=1))
            mock_minio.return_value.bucket_exists = MagicMock(return_value=True)
            mock_minio.return_value.put_object = MagicMock()

            result = await FileService.upload_file(
                db=mock_db,
                filename="test.jpg",
                content=b"test content",
                content_type="image/jpeg",
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_upload_file_duplicate(self):
        """测试上传重复文件（MD5 去重）"""
        mock_db = AsyncMock()
        mock_existing_file = MagicMock()
        mock_existing_file.md5 = "existing_md5"

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_md5 = AsyncMock(return_value=mock_existing_file)

            result = await FileService.upload_file(
                db=mock_db,
                filename="test.jpg",
                content=b"test content",
                content_type="image/jpeg",
            )

            assert result == mock_existing_file
            # 不应该调用 create
            mock_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_file_success(self):
        """测试删除文件成功"""
        mock_db = AsyncMock()
        mock_file = MagicMock()
        mock_file.id = 1

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_file)
            mock_repo.delete = AsyncMock()

            result = await FileService.delete_file(mock_db, 1)

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self):
        """测试删除文件时文件不存在"""
        mock_db = AsyncMock()

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="文件不存在"):
                await FileService.delete_file(mock_db, 999)

    @pytest.mark.asyncio
    async def test_check_file_exists_true(self):
        """测试检查文件存在"""
        mock_db = AsyncMock()

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_md5 = AsyncMock(return_value=MagicMock())

            result = await FileService.check_file_exists(mock_db, "existing_md5")

            assert result is True

    @pytest.mark.asyncio
    async def test_check_file_exists_false(self):
        """测试检查文件不存在"""
        mock_db = AsyncMock()

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_md5 = AsyncMock(return_value=None)

            result = await FileService.check_file_exists(mock_db, "nonexistent_md5")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_file_by_id(self):
        """测试根据 ID 获取文件"""
        mock_db = AsyncMock()
        mock_file = MagicMock()
        mock_file.id = 1

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=mock_file)

            result = await FileService.get_file_by_id(mock_db, 1)

            assert result == mock_file

    @pytest.mark.asyncio
    async def test_get_file_by_md5(self):
        """测试根据 MD5 获取文件"""
        mock_db = AsyncMock()
        mock_file = MagicMock()
        mock_file.md5 = "test_md5"

        with patch("app.service.file_service.file_repository") as mock_repo:
            mock_repo.get_by_md5 = AsyncMock(return_value=mock_file)

            result = await FileService.get_file_by_md5(mock_db, "test_md5")

            assert result == mock_file
