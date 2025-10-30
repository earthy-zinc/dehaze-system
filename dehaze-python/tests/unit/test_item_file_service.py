import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.models import SysItemFile, SysFile
from app.service.item_file_service import ItemFileService


class TestItemFileService(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        self.test_item_file = SysItemFile(
            id=1,
            item_id=100,
            file_id=200,
            thumbnail_file_id=300,
            type='haze',
            description='雾霾图像'
        )

        self.test_file = SysFile(
            id=200,
            type='png',
            url='http://localhost:9000/test-bucket/upload/20231201/test.png',
            name='test.png',
            object_name='upload/20231201/test.png',
            size='100KB',
            path='',
            md5='d41d8cd98f00b204e9800998ecf8427e'
        )

    @patch('app.service.item_file_service.SysItemFile')
    @patch('app.service.item_file_service.SysFile')
    def test_get_image_urls(self, mock_sys_file, mock_sys_item_file):
        """测试获取图片URL列表"""
        # 设置mock返回值
        mock_item_file_query = MagicMock()
        mock_sys_item_file.query = mock_item_file_query
        mock_item_file_query.filter.return_value.all.return_value = [self.test_item_file]

        mock_sys_file.query.get.return_value = self.test_file

        # 测试获取图片URL列表
        result = ItemFileService.get_image_urls(100)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[0]['url'], 'http://localhost:9000/test-bucket/upload/20231201/test.png')

    @patch('app.service.item_file_service.SysItemFile')
    @patch('app.service.item_file_service.SysFile')
    def test_get_image_urls_empty(self, mock_sys_file, mock_sys_item_file):
        """测试获取空图片URL列表"""
        # 设置mock返回值
        mock_item_file_query = MagicMock()
        mock_sys_item_file.query = mock_item_file_query
        mock_item_file_query.filter.return_value.all.return_value = []

        # 测试获取空图片URL列表
        result = ItemFileService.get_image_urls(100)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    @patch('app.service.item_file_service.SysItemFile')
    @patch('app.service.item_file_service.mysql')
    def test_delete_item_file(self, mock_mysql, mock_sys_item_file):
        """测试删除数据项文件"""
        # 设置mock返回值
        mock_sys_item_file.query.get.return_value = self.test_item_file
        mock_mysql.session.delete = MagicMock()
        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功删除
        result = ItemFileService.delete_item_file(1)
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '删除成功')
        mock_mysql.session.delete.assert_called_once()
        mock_mysql.session.commit.assert_called_once()

    @patch('app.service.item_file_service.SysItemFile')
    @patch('app.service.item_file_service.mysql')
    def test_delete_item_file_not_found(self, mock_mysql, mock_sys_item_file):
        """测试删除不存在的数据项文件"""
        # 设置mock返回值
        mock_sys_item_file.query.get.return_value = None

        # 测试删除不存在的数据项文件
        result = ItemFileService.delete_item_file(999)
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], '未查询到对应数据项')

    @patch('app.service.item_file_service.SysItemFile')
    @patch('app.service.item_file_service.mysql')
    def test_delete_item_file_failure(self, mock_mysql, mock_sys_item_file):
        """测试删除数据项文件失败"""
        # 设置mock返回值
        mock_sys_item_file.query.get.return_value = self.test_item_file
        mock_mysql.session.delete.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        # 测试删除数据项文件失败
        result = ItemFileService.delete_item_file(1)
        self.assertFalse(result['success'])
        self.assertIn('删除数据项文件失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
