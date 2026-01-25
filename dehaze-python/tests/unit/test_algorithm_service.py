import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.models import SysAlgorithm
from app.service.algorithm_service import AlgorithmService


class TestAlgorithmService(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        self.test_algorithm1 = SysAlgorithm(
            id=1,
            parent_id=0,
            type='dehaze',
            name='AOD-Net',
            path='/path/to/aodnet.pth',
            size='10MB',
            img='/path/to/aodnet.jpg',
            params='100K',
            flops='10G',
            import_path='models.aodnet',
            description='AOD-Net去雾算法',
            status=1
        )

        self.test_algorithm2 = SysAlgorithm(
            id=2,
            parent_id=1,
            type='dehaze',
            name='AOD-Net Improved',
            path='/path/to/aodnet_improved.pth',
            size='12MB',
            img='/path/to/aodnet_improved.jpg',
            params='120K',
            flops='12G',
            import_path='models.aodnet_improved',
            description='改进的AOD-Net去雾算法',
            status=1
        )

        self.test_algorithm3 = SysAlgorithm(
            id=3,
            parent_id=0,
            type='segmentation',
            name='DeepLabv3',
            path='/path/to/deeplabv3.pth',
            size='20MB',
            img='/path/to/deeplabv3.jpg',
            params='200K',
            flops='20G',
            import_path='models.deeplabv3',
            description='DeepLabv3分割算法',
            status=0  # 禁用状态
        )

    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_get_algorithm_list(self, mock_sys_algorithm):
        """测试获取算法列表"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_algorithm.query = mock_query
        mock_query.all.return_value = [self.test_algorithm1, self.test_algorithm2, self.test_algorithm3]

        # 测试无关键词搜索
        result = AlgorithmService.get_algorithm_list()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # 应该只有2个根节点算法
        self.assertTrue(any(alg['id'] == 1 for alg in result))
        self.assertTrue(any(alg['id'] == 3 for alg in result))

        # 测试带关键词搜索
        mock_query.filter.return_value = mock_query
        result = AlgorithmService.get_algorithm_list('AOD')
        mock_query.filter.assert_called()  # 确保调用了filter方法

    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_get_algorithm_options(self, mock_sys_algorithm):
        """测试获取算法下拉选项"""
        # 设置mock返回值，只返回启用的算法
        mock_query = MagicMock()
        mock_sys_algorithm.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [self.test_algorithm1, self.test_algorithm2]

        result = AlgorithmService.get_algorithm_options()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # 只有一个根节点算法（启用的）
        self.assertEqual(result[0]['value'], 1)
        self.assertEqual(result[0]['label'], 'AOD-Net')

    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_get_algorithm_by_id(self, mock_sys_algorithm):
        """测试根据ID获取算法"""
        # 设置mock返回值
        mock_sys_algorithm.query.get.return_value = self.test_algorithm1

        # 测试存在的算法
        result = AlgorithmService.get_algorithm_by_id(1)
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 1)
        self.assertEqual(result['name'], 'AOD-Net')

        # 测试不存在的算法
        mock_sys_algorithm.query.get.return_value = None
        result = AlgorithmService.get_algorithm_by_id(999)
        self.assertIsNone(result)

    @patch('app.service.algorithm_service.mysql')
    @patch('app.service.algorithm_service.SysAlgorithm')
    @patch('app.service.algorithm_service.os')
    def test_create_algorithm(self, mock_os, mock_sys_algorithm, mock_mysql):
        """测试创建算法"""
        # 设置mock返回值
        mock_os.path.isfile.return_value = True
        mock_instance = MagicMock()
        mock_sys_algorithm.return_value = mock_instance
        mock_mysql.session.add = MagicMock()
        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功创建
        data = {
            'parent_id': 0,
            'type': 'test',
            'name': 'Test Algorithm',
            'path': '/path/to/test.pth',
            'import_path': 'models.test',
            'description': '测试算法',
            'status': 1
        }

        result = AlgorithmService.create_algorithm(data)
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '算法创建成功')
        mock_mysql.session.add.assert_called_once()
        mock_mysql.session.commit.assert_called_once()

    @patch('app.service.algorithm_service.mysql')
    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_create_algorithm_failure(self, mock_sys_algorithm, mock_mysql):
        """测试创建算法失败"""
        # 设置mock返回值
        mock_instance = MagicMock()
        mock_sys_algorithm.return_value = mock_instance
        mock_mysql.session.add.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        data = {
            'name': 'Test Algorithm'
        }

        result = AlgorithmService.create_algorithm(data)
        self.assertFalse(result['success'])
        self.assertIn('算法创建失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()

    @patch('app.service.algorithm_service.mysql')
    @patch('app.service.algorithm_service.SysAlgorithm')
    @patch('app.service.algorithm_service.os')
    def test_update_algorithm(self, mock_os, mock_sys_algorithm, mock_mysql):
        """测试更新算法"""
        # 设置mock返回值
        mock_os.path.isfile.return_value = True
        mock_algorithm = MagicMock()
        mock_sys_algorithm.query.get.return_value = mock_algorithm
        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功更新
        data = {
            'name': 'Updated Algorithm',
            'path': '/path/to/updated.pth'
        }

        result = AlgorithmService.update_algorithm(1, data)
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '算法更新成功')
        mock_mysql.session.commit.assert_called_once()
        self.assertEqual(mock_algorithm.name, 'Updated Algorithm')

    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_update_algorithm_not_found(self, mock_sys_algorithm):
        """测试更新不存在的算法"""
        # 设置mock返回值
        mock_sys_algorithm.query.get.return_value = None

        data = {
            'name': 'Updated Algorithm'
        }

        result = AlgorithmService.update_algorithm(999, data)
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], '算法不存在')

    @patch('app.service.algorithm_service.mysql')
    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_update_algorithm_failure(self, mock_sys_algorithm, mock_mysql):
        """测试更新算法失败"""
        # 设置mock返回值
        mock_algorithm = MagicMock()
        mock_sys_algorithm.query.get.return_value = mock_algorithm
        mock_mysql.session.commit.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        data = {
            'name': 'Updated Algorithm'
        }

        result = AlgorithmService.update_algorithm(1, data)
        self.assertFalse(result['success'])
        self.assertIn('算法更新失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()

    @patch('app.service.algorithm_service.mysql')
    @patch('app.service.algorithm_service.SysAlgorithm')
    @patch('app.service.algorithm_service.or_')
    def test_delete_algorithms(self, mock_or, mock_sys_algorithm, mock_mysql):
        """测试删除算法"""
        # 设置mock返回值
        mock_query1 = MagicMock()
        mock_query2 = MagicMock()
        mock_query1.all.return_value = [self.test_algorithm1, self.test_algorithm2]
        mock_query2.delete.return_value = 2  # 返回删除的记录数

        # 设置filter方法的链式调用
        mock_filter = MagicMock()
        mock_filter.filter.side_effect = [mock_query1, mock_query2]
        mock_sys_algorithm.query = mock_filter

        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功删除
        result = AlgorithmService.delete_algorithms([1])
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '算法删除成功')
        mock_query2.delete.assert_called_once()
        mock_mysql.session.commit.assert_called_once()

    @patch('app.service.algorithm_service.mysql')
    @patch('app.service.algorithm_service.SysAlgorithm')
    def test_delete_algorithms_failure(self, mock_sys_algorithm, mock_mysql):
        """测试删除算法失败"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_algorithm.query.filter.return_value = mock_query
        mock_query.delete.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        result = AlgorithmService.delete_algorithms([1])
        self.assertFalse(result['success'])
        self.assertIn('算法删除失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
