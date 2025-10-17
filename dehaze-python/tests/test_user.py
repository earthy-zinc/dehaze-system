import unittest
import json
from app.extensions import mysql
from app.models import SysUser, SysRole, SysUserRole
from app.service.user import UserService
from werkzeug.security import generate_password_hash


class UserServiceTestCase(unittest.TestCase):
    """用户服务测试用例"""

    def setUp(self):
        """测试前准备"""
        self.app_context = mysql.app.app_context()
        self.app_context.push()
        mysql.create_all()

        # 创建测试角色
        role1 = SysRole(name='管理员', code='admin', status=1)
        role2 = SysRole(name='普通用户', code='user', status=1)
        mysql.session.add(role1)
        mysql.session.add(role2)
        mysql.session.commit()
        self.role1_id = role1.id
        self.role2_id = role2.id

    def tearDown(self):
        """测试后清理"""
        mysql.session.remove()
        mysql.drop_all()
        self.app_context.pop()

    def test_create_user(self):
        """测试创建用户"""
        user = UserService.create_user('testuser', 'password123', 'Test User')
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.nickname, 'Test User')
        
        # 验证密码是否正确加密
        self.assertTrue(user.password.startswith('pbkdf2:sha256:'))

    def test_create_user_with_roles(self):
        """测试创建用户并关联角色"""
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User',
            'gender': 1,
            'deptId': 1,
            'mobile': '13800138000',
            'email': 'test@example.com',
            'roleIds': [self.role1_id, self.role2_id]
        }
        
        result = UserService.create_user_with_roles(user_data)
        self.assertNotIn('error', result)
        self.assertIn('data', result)
        
        # 验证用户创建成功
        user = UserService.get_user_by_username('testuser')
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'testuser')
        self.assertEqual(user.nickname, 'Test User')
        self.assertEqual(user.gender, 1)
        self.assertEqual(user.dept_id, 1)
        self.assertEqual(user.mobile, '13800138000')
        self.assertEqual(user.email, 'test@example.com')
        
        # 验证角色关联成功
        roles = UserService.get_user_roles(user.id)
        role_ids = [role.id for role in roles]
        self.assertIn(self.role1_id, role_ids)
        self.assertIn(self.role2_id, role_ids)

    def test_authenticate_user(self):
        """测试用户认证"""
        # 创建测试用户
        UserService.create_user('testuser', 'password123', 'Test User')
        
        # 验证正确凭据
        user = UserService.authenticate_user('testuser', 'password123')
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'testuser')
        
        # 验证错误密码
        user = UserService.authenticate_user('testuser', 'wrongpassword')
        self.assertIsNone(user)
        
        # 验证不存在的用户
        user = UserService.authenticate_user('nonexistent', 'password123')
        self.assertIsNone(user)

    def test_get_user_by_username(self):
        """测试根据用户名获取用户"""
        # 创建测试用户
        created_user = UserService.create_user('testuser', 'password123', 'Test User')
        
        # 获取用户
        user = UserService.get_user_by_username('testuser')
        self.assertIsNotNone(user)
        self.assertEqual(user.id, created_user.id)
        self.assertEqual(user.username, 'testuser')

    def test_get_user_by_id(self):
        """测试根据ID获取用户"""
        # 创建测试用户
        created_user = UserService.create_user('testuser', 'password123', 'Test User')
        
        # 获取用户
        user = UserService.get_user_by_id(created_user.id)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, 'testuser')

    def test_get_user_form_data(self):
        """测试获取用户表单数据"""
        # 创建测试用户
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User',
            'gender': 1,
            'deptId': 1,
            'mobile': '13800138000',
            'email': 'test@example.com',
            'roleIds': [self.role1_id]
        }
        
        result = UserService.create_user_with_roles(user_data)
        user_id = result['data']['id']
        
        # 获取表单数据
        form_data = UserService.get_user_form_data(user_id)
        self.assertIsNotNone(form_data)
        self.assertEqual(form_data['username'], 'testuser')
        self.assertEqual(form_data['nickname'], 'Test User')
        self.assertEqual(form_data['gender'], 1)
        self.assertEqual(form_data['deptId'], 1)
        self.assertEqual(form_data['mobile'], '13800138000')
        self.assertEqual(form_data['email'], 'test@example.com')
        self.assertIn(self.role1_id, form_data['roleIds'])

    def test_update_user_with_roles(self):
        """测试更新用户信息并关联角色"""
        # 创建测试用户
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User',
            'roleIds': [self.role1_id]
        }
        
        result = UserService.create_user_with_roles(user_data)
        user_id = result['data']['id']
        
        # 更新用户信息
        update_data = {
            'username': 'updateduser',
            'nickname': 'Updated User',
            'gender': 2,
            'deptId': 2,
            'mobile': '13900139000',
            'email': 'updated@example.com',
            'status': 0,
            'roleIds': [self.role2_id]
        }
        
        result = UserService.update_user_with_roles(user_id, update_data)
        self.assertNotIn('error', result)
        
        # 验证用户信息更新成功
        user = UserService.get_user_by_id(user_id)
        self.assertEqual(user.username, 'updateduser')
        self.assertEqual(user.nickname, 'Updated User')
        self.assertEqual(user.gender, 2)
        self.assertEqual(user.dept_id, 2)
        self.assertEqual(user.mobile, '13900139000')
        self.assertEqual(user.email, 'updated@example.com')
        self.assertEqual(user.status, 0)
        
        # 验证角色关联已更新
        roles = UserService.get_user_roles(user_id)
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].id, self.role2_id)

    def test_update_password(self):
        """测试更新用户密码"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        old_password_hash = user.password
        
        # 更新密码
        result = UserService.update_password(user.id, 'newpassword456')
        self.assertTrue(result)
        
        # 验证密码已更新
        updated_user = UserService.get_user_by_id(user.id)
        self.assertNotEqual(updated_user.password, old_password_hash)
        
        # 验证新密码可以认证
        authenticated_user = UserService.authenticate_user('testuser', 'newpassword456')
        self.assertIsNotNone(authenticated_user)

    def test_get_user_list(self):
        """测试获取用户列表"""
        # 创建测试用户
        UserService.create_user('user1', 'password123', 'User One')
        UserService.create_user('user2', 'password123', 'User Two')
        UserService.create_user('testuser', 'password123', 'Test User')
        
        # 获取所有用户
        users, total = UserService.get_user_list()
        self.assertEqual(len(users), 3)
        self.assertEqual(total, 3)
        
        # 按用户名搜索
        users, total = UserService.get_user_list(username='test')
        self.assertEqual(len(users), 1)
        self.assertEqual(total, 1)
        self.assertEqual(users[0].username, 'testuser')

    def test_update_user_status(self):
        """测试更新用户状态"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        self.assertEqual(user.status, 1)  # 默认状态为启用
        
        # 禁用用户
        result = UserService.update_user_status(user.id, 0)
        self.assertTrue(result)
        
        # 验证状态已更新
        updated_user = UserService.get_user_by_id(user.id)
        self.assertEqual(updated_user.status, 0)
        
        # 启用用户
        result = UserService.update_user_status(user.id, 1)
        self.assertTrue(result)
        
        # 验证状态已更新
        updated_user = UserService.get_user_by_id(user.id)
        self.assertEqual(updated_user.status, 1)

    def test_delete_user(self):
        """测试删除用户"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        self.assertEqual(user.deleted, 0)  # 默认未删除
        
        # 删除用户
        result = UserService.delete_user(user.id)
        self.assertTrue(result)
        
        # 验证用户已被标记为删除
        updated_user = UserService.get_user_by_id(user.id)
        self.assertEqual(updated_user.deleted, 1)
        
        # 验证无法再通过用户名获取到该用户
        user_by_name = UserService.get_user_by_username('testuser')
        self.assertIsNone(user_by_name)


if __name__ == '__main__':
    unittest.main()