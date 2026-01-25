import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.models import SysDept
from app.service.dept_service import DeptService


class TestDeptService(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        self.test_dept1 = SysDept(
            id=1,
            name='研发部',
            parent_id=0,
            tree_path='0',
            sort=1,
            status=1,
            deleted=0
        )

        self.test_dept2 = SysDept(
            id=2,
            name='前端组',
            parent_id=1,
            tree_path='0,1',
            sort=1,
            status=1,
            deleted=0
        )

        self.test_dept3 = SysDept(
            id=3,
            name='后端组',
            parent_id=1,
            tree_path='0,1',
            sort=2,
            status=1,
            deleted=0
        )

    @patch('app.service.dept_service.SysDept')
    def test_get_dept_list(self, mock_sys_dept):
        """测试获取部门列表"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.all.return_value = [self.test_dept1, self.test_dept2, self.test_dept3]
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        # 测试无参数查询
        result = DeptService.get_dept_list()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # 应该只有1个根节点
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(len(result[0]['children']), 2)  # 应该有2个子节点

        # 测试带关键字查询
        result = DeptService.get_dept_list(keywords='研发')
        mock_query.filter.assert_called()

        # 测试带状态查询
        result = DeptService.get_dept_list(status=1)
        mock_query.filter.assert_called()

    @patch('app.service.dept_service.SysDept')
    def test_get_dept_list_empty(self, mock_sys_dept):
        """测试获取空部门列表"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.all.return_value = []

        result = DeptService.get_dept_list()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    @patch('app.service.dept_service.SysDept')
    def test_get_dept_options(self, mock_sys_dept):
        """测试获取部门下拉选项"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [self.test_dept1, self.test_dept2, self.test_dept3]

        result = DeptService.get_dept_options()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # 应该只有1个根节点
        self.assertEqual(result[0]['value'], 1)
        self.assertEqual(result[0]['label'], '研发部')
        self.assertEqual(len(result[0]['children']), 2)  # 应该有2个子节点

    @patch('app.service.dept_service.SysDept')
    def test_get_dept_options_empty(self, mock_sys_dept):
        """测试获取空部门下拉选项"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        result = DeptService.get_dept_options()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    @patch('app.service.dept_service.SysDept')
    def test_get_dept_form(self, mock_sys_dept):
        """测试获取部门表单数据"""
        # 设置mock返回值
        mock_sys_dept.query.get.return_value = self.test_dept1

        # 测试存在的部门
        result = DeptService.get_dept_form(1)
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], 1)
        self.assertEqual(result['name'], '研发部')

        # 测试不存在的部门
        mock_sys_dept.query.get.return_value = None
        result = DeptService.get_dept_form(999)
        self.assertIsNone(result)

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_create_dept(self, mock_sys_dept, mock_mysql):
        """测试创建部门"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0  # 部门名称不存在

        mock_instance = MagicMock()
        mock_instance.id = 1
        mock_sys_dept.return_value = mock_instance

        mock_mysql.session.add = MagicMock()
        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功创建
        data = {
            'name': '测试部',
            'parent_id': 0,
            'status': 1,
            'sort': 1
        }

        result = DeptService.create_dept(data)
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '部门创建成功')
        self.assertEqual(result['data'], 1)
        mock_mysql.session.add.assert_called_once()
        mock_mysql.session.commit.assert_called_once()

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_create_dept_duplicate_name(self, mock_sys_dept, mock_mysql):
        """测试创建部门时名称重复"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1  # 部门名称已存在

        data = {
            'name': '测试部',
            'parent_id': 0,
            'status': 1,
            'sort': 1
        }

        result = DeptService.create_dept(data)
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], '部门名称已存在')

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_create_dept_failure(self, mock_sys_dept, mock_mysql):
        """测试创建部门失败"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0  # 部门名称不存在

        mock_instance = MagicMock()
        mock_sys_dept.return_value = mock_instance
        mock_mysql.session.add.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        data = {
            'name': '测试部',
            'parent_id': 0,
            'status': 1,
            'sort': 1
        }

        result = DeptService.create_dept(data)
        self.assertFalse(result['success'])
        self.assertIn('部门创建失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_update_dept(self, mock_sys_dept, mock_mysql):
        """测试更新部门"""
        # 设置mock返回值
        mock_dept = MagicMock()
        mock_dept.id = 1
        mock_dept.name = '研发部'
        mock_dept.parent_id = 0
        mock_dept.tree_path = '0'
        mock_dept.status = 1
        mock_dept.sort = 1

        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_sys_dept.query.get.return_value = mock_dept
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0  # 部门名称不存在

        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功更新
        data = {
            'name': '研发部Updated',
            'status': 0,
            'sort': 2
        }

        result = DeptService.update_dept(1, data)
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '部门更新成功')
        self.assertEqual(result['data'], 1)
        mock_mysql.session.commit.assert_called_once()
        self.assertEqual(mock_dept.name, '研发部Updated')

    @patch('app.service.dept_service.SysDept')
    def test_update_dept_not_found(self, mock_sys_dept):
        """测试更新不存在的部门"""
        # 设置mock返回值
        mock_sys_dept.query.get.return_value = None

        data = {
            'name': '研发部Updated'
        }

        result = DeptService.update_dept(999, data)
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], '部门不存在')

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_update_dept_duplicate_name(self, mock_sys_dept, mock_mysql):
        """测试更新部门时名称重复"""
        # 设置mock返回值
        mock_dept = MagicMock()
        mock_sys_dept.query.get.return_value = mock_dept
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1  # 部门名称已存在

        data = {
            'name': '测试部'
        }

        result = DeptService.update_dept(1, data)
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], '部门名称已存在')

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_update_dept_failure(self, mock_sys_dept, mock_mysql):
        """测试更新部门失败"""
        # 设置mock返回值
        mock_dept = MagicMock()
        mock_sys_dept.query.get.return_value = mock_dept
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0  # 部门名称不存在

        mock_mysql.session.commit.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        data = {
            'name': '研发部Updated'
        }

        result = DeptService.update_dept(1, data)
        self.assertFalse(result['success'])
        self.assertIn('部门更新失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    @patch('app.service.dept_service.or_')
    def test_delete_depts(self, mock_or, mock_sys_dept, mock_mysql):
        """测试删除部门"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_filtered_query = MagicMock()
        mock_filtered_query.delete.return_value = 1
        mock_query.filter.return_value = mock_filtered_query
        mock_mysql.session.commit = MagicMock()
        mock_mysql.session.rollback = MagicMock()

        # 测试成功删除
        result = DeptService.delete_depts([1])
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], '部门删除成功')
        mock_mysql.session.commit.assert_called_once()

    @patch('app.service.dept_service.mysql')
    @patch('app.service.dept_service.SysDept')
    def test_delete_depts_failure(self, mock_sys_dept, mock_mysql):
        """测试删除部门失败"""
        # 设置mock返回值
        mock_query = MagicMock()
        mock_sys_dept.query = mock_query
        mock_query.filter.side_effect = Exception('数据库错误')
        mock_mysql.session.rollback = MagicMock()

        result = DeptService.delete_depts([1])
        self.assertFalse(result['success'])
        self.assertIn('部门删除失败', result['message'])
        mock_mysql.session.rollback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
