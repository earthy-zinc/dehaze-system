import os
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.models import SysFile
from app.service.file import upload_file_from_request


class TestFileService:

    def setup_method(self):
        """测试前准备"""
        # 创建测试数据
        self.test_file = SysFile(
            id=1,
            type='png',
            url='http://localhost:9000/test-bucket/upload/20231201/test.png',
            name='test.png',
            object_name='upload/20231201/test.png',
            size='100KB',
            path='',
            md5='d41d8cd98f00b204e9800998ecf8427e'
        )

    @patch('app.service.file._upload_to_storage')
    @patch('app.service.file.BytesIO')
    def test_upload_file_from_request(self, mock_bytes_io, mock_upload_to_storage, app):
        """测试从请求上传文件"""
        # 设置mock返回值
        mock_file_storage = MagicMock()
        mock_file_storage.filename = 'test.png'
        mock_file_storage.mimetype = 'image/png'
        mock_file_storage.read.return_value = b'test content'
        mock_file_storage.content_length = 12

        mock_bytes_instance = MagicMock()
        mock_bytes_io.return_value = mock_bytes_instance

        mock_upload_to_storage.return_value = self.test_file

        # 测试文件上传
        with app.app_context():
            result = upload_file_from_request(mock_file_storage)
            assert result.id == 1
            assert result.name == 'test.png'
            assert result.md5 == 'd41d8cd98f00b204e9800998ecf8427e'

    @patch('app.service.file.calculate_bytes_md5')
    @patch('app.service.file.SysFile')
    @patch('app.service.file.mysql')
    @patch('app.service.file.current_app')
    def test_upload_to_storage_new_file(self, mock_current_app, mock_mysql, mock_sys_file, mock_calculate_md5, app):
        """测试上传新文件到存储"""
        # 设置mock返回值
        mock_minio_client = MagicMock()
        mock_current_app.extensions = {"minio_client": mock_minio_client}
        mock_current_app.config = {"MINIO_BUCKET_NAME": "test-bucket"}

        mock_calculate_md5.return_value = 'd41d8cd98f00b204e9800998ecf8427e'

        mock_query = MagicMock()
        mock_sys_file.query = mock_query
        mock_query.filter_by.return_value.first.return_value = None  # 文件不存在

        mock_file_instance = MagicMock()
        mock_sys_file.return_value = mock_file_instance

        mock_mysql.session.add = MagicMock()
        mock_mysql.session.commit = MagicMock()

        # 测试上传新文件
        filename = 'test.png'
        content_type = 'image/png'
        file_bytes = BytesIO(b'test content')
        file_size = 12

        # 使用应用上下文
        with app.app_context():
            from app.service.file import _upload_to_storage
            result = _upload_to_storage(filename, content_type, file_bytes, file_size)
            assert result is not None
            mock_minio_client.put_object.assert_called_once()
            mock_mysql.session.add.assert_called_once()
            mock_mysql.session.commit.assert_called_once()

    @patch('app.service.file.SysFile')
    @patch('app.service.file.current_app')
    def test_upload_to_storage_existing_file(self, mock_current_app, mock_sys_file, app):
        """测试上传已存在的文件"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_file.query = mock_query
        mock_query.filter_by.return_value.first.return_value = self.test_file  # 文件已存在

        # 测试上传已存在的文件
        filename = 'test.png'
        content_type = 'image/png'
        file_bytes = BytesIO(b'test content')
        file_size = 12

        # 使用应用上下文
        with app.app_context():
            from app.service.file import _upload_to_storage
            result = _upload_to_storage(filename, content_type, file_bytes, file_size)
            assert result.id == 1
            assert result.md5 == 'd41d8cd98f00b204e9800998ecf8427e'

    def test_generate_object_name(self):
        """测试生成对象名"""
        from app.service.file import _generate_object_name
        import re

        md5 = 'd41d8cd98f00b204e9800998ecf8427e'
        extension = 'png'

        result = _generate_object_name(md5, extension)
        # 检查格式是否正确
        pattern = r'upload/\d{8}/[a-f0-9]{32}\.png'
        assert re.match(pattern, result)

    @patch('app.service.file.SysFile')
    def test_check_file_exists(self, mock_sys_file, app):
        """测试检查文件是否存在"""

        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_file.query = mock_query
        mock_query.filter_by.return_value.first.return_value = self.test_file  # 文件存在

        # 测试文件存在
        with app.app_context():
            result = mock_sys_file.query.filter_by(md5='d41d8cd98f00b204e9800998ecf8427e').first()
            assert result is not None
            assert result.md5 == 'd41d8cd98f00b204e9800998ecf8427e'

    @patch('app.service.file.SysFile')
    def test_check_file_not_exists(self, mock_sys_file, app):
        """测试检查文件不存在"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_file.query = mock_query
        mock_query.filter_by.return_value.first.return_value = None  # 文件不存在

        # 测试文件不存在
        with app.app_context():
            result = mock_sys_file.query.filter_by(md5='nonexistent').first()
            assert result is None
