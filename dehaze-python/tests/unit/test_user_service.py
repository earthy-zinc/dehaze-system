"""
用户服务测试
"""
import pytest

from app.models import SysRole
from app.service.user import UserService


@pytest.mark.unit
@pytest.mark.requires_db
class TestUserService:
    """用户服务测试类"""

    def test_create_user(self, db_session):
        """测试创建用户"""
        user = UserService.create_user('testuser', 'password123', 'Test User')

        assert user is not None
        assert user.username == 'testuser'
        assert user.nickname == 'Test User'
        # 验证密码是否正确加密
        assert user.password.startswith('$2b$')

    def test_create_user_with_roles(self, db_session, sample_roles):
        """测试创建用户并关联角色"""
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User',
            'gender': 1,
            'deptId': 1,
            'mobile': '13800138000',
            'email': 'test@example.com',
            'roleIds': [sample_roles['admin'].id, sample_roles['user'].id]
        }

        result = UserService.create_user_with_roles(user_data)
        assert 'error' not in result
        assert 'data' in result

        # 验证用户创建成功
        user = UserService.get_user_by_username('testuser')
        assert user is not None
        assert user.username == 'testuser'
        assert user.nickname == 'Test User'
        assert user.gender == 1
        assert user.dept_id == 1
        assert user.mobile == '13800138000'
        assert user.email == 'test@example.com'

        # 验证角色关联成功
        roles = UserService.get_user_roles(user.id)
        role_ids = [role.id for role in roles]
        assert sample_roles['admin'].id in role_ids
        assert sample_roles['user'].id in role_ids

    def test_authenticate_user(self, db_session):
        """测试用户认证"""
        # 创建测试用户
        UserService.create_user('testuser', 'password123', 'Test User')

        # 验证正确凭据
        user = UserService.authenticate_user('testuser', 'password123')
        assert user is not None
        assert user.username == 'testuser'

        # 验证错误密码
        user = UserService.authenticate_user('testuser', 'wrongpassword')
        assert user is None

        # 验证不存在的用户
        user = UserService.authenticate_user('nonexistent', 'password123')
        assert user is None

    def test_get_user_by_username(self, db_session):
        """测试根据用户名获取用户"""
        # 创建测试用户
        created_user = UserService.create_user(
            'testuser', 'password123', 'Test User')

        # 获取用户
        user = UserService.get_user_by_username('testuser')
        assert user is not None
        assert user.id == created_user.id
        assert user.username == 'testuser'

    def test_get_user_by_id(self, db_session):
        """测试根据ID获取用户"""
        # 创建测试用户
        created_user = UserService.create_user(
            'testuser', 'password123', 'Test User')

        # 获取用户
        user = UserService.get_user_by_id(created_user.id)
        assert user is not None
        assert user.username == 'testuser'

    def test_get_user_form_data(self, db_session, sample_roles):
        """测试获取用户表单数据"""
        # 创建测试用户
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User',
            'gender': 1,
            'deptId': 1,
            'mobile': '13800138000',
            'email': 'test@example.com',
            'roleIds': [sample_roles['admin'].id]
        }

        result = UserService.create_user_with_roles(user_data)
        user_id = result['data']['id']

        # 获取表单数据
        form_data = UserService.get_user_form_data(user_id)
        assert form_data is not None
        assert form_data['username'] == 'testuser'
        assert form_data['nickname'] == 'Test User'
        assert form_data['gender'] == 1
        assert form_data['deptId'] == 1
        assert form_data['mobile'] == '13800138000'
        assert form_data['email'] == 'test@example.com'
        assert sample_roles['admin'].id in form_data['roleIds']

    def test_update_user_with_roles(self, db_session, sample_roles):
        """测试更新用户信息并关联角色"""
        # 创建测试用户
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User',
            'roleIds': [sample_roles['admin'].id]
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
            'roleIds': [sample_roles['user'].id]
        }

        result = UserService.update_user_with_roles(user_id, update_data)
        assert 'error' not in result

        # 验证用户信息更新成功
        user = UserService.get_user_by_id(user_id)
        assert user.username == 'updateduser'
        assert user.nickname == 'Updated User'
        assert user.gender == 2
        assert user.dept_id == 2
        assert user.mobile == '13900139000'
        assert user.email == 'updated@example.com'
        assert user.status == 0

        # 验证角色关联已更新
        roles = UserService.get_user_roles(user_id)
        assert len(roles) == 1
        assert roles[0].id == sample_roles['user'].id

    def test_update_password(self, db_session):
        """测试更新用户密码"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        old_password_hash = user.password

        # 更新密码
        result = UserService.update_password(user.id, 'newpassword456')
        assert result is True

        # 验证密码已更新
        updated_user = UserService.get_user_by_id(user.id)
        assert updated_user.password != old_password_hash

        # 验证新密码可以认证
        authenticated_user = UserService.authenticate_user(
            'testuser', 'newpassword456')
        assert authenticated_user is not None

    def test_get_user_list(self, db_session):
        """测试获取用户列表"""
        # 创建测试用户
        UserService.create_user('user1', 'password123', 'User One')
        UserService.create_user('user2', 'password123', 'User Two')
        UserService.create_user('testuser', 'password123', 'Test User')

        # 获取所有用户
        users, total = UserService.get_user_list()
        assert len(users) == 3
        assert total == 3

        # 按用户名搜索
        users, total = UserService.get_user_list(username='test')
        assert len(users) == 1
        assert total == 1

    def test_get_user_list_pagination(self, db_session):
        """测试用户列表分页"""
        # 创建测试用户
        for i in range(15):
            user = UserService.create_user(f'user{i}', 'password123', f'User {i}')

        # 测试第一页
        users, total = UserService.get_user_list(page=1, page_size=10)
        assert len(users) == 10
        assert total == 15

        # 测试第二页
        users, total = UserService.get_user_list(page=2, page_size=10)
        assert len(users) == 5
        assert total == 15

    def test_get_user_list_empty(self, db_session):
        """测试获取空用户列表"""
        users, total = UserService.get_user_list()
        assert len(users) == 0
        assert total == 0

    def test_get_user_roles_empty(self, db_session):
        """测试获取用户角色（无角色）"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')

        roles = UserService.get_user_roles(user.id)
        assert len(roles) == 0

    def test_get_user_permissions_empty(self, db_session):
        """测试获取用户权限（无角色）"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')

        permissions = UserService.get_user_permissions(user.id)
        assert len(permissions) == 0

    def test_get_user_permissions_with_roles(self, db_session):
        """测试获取用户权限（有角色）"""
        # 创建测试角色
        role1 = SysRole(name='管理员', code='ADMIN', status=1, deleted=0)
        role2 = SysRole(name='用户', code='USER', status=1, deleted=0)
        db_session.add(role1)
        db_session.add(role2)
        db_session.commit()

        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')

        # 关联角色
        from app.models import SysUserRole
        user_role1 = SysUserRole(user_id=user.id, role_id=role1.id)
        user_role2 = SysUserRole(user_id=user.id, role_id=role2.id)
        db_session.add(user_role1)
        db_session.add(user_role2)
        db_session.commit()

        permissions = UserService.get_user_permissions(user.id)
        assert len(permissions) == 2
        assert 'role_ADMIN' in permissions
        assert 'role_USER' in permissions

    def test_create_user_with_roles_missing_username(self, db_session):
        """测试创建用户时缺少用户名"""
        user_data = {
            'nickname': 'Test User'
        }

        result = UserService.create_user_with_roles(user_data)
        assert 'error' in result
        assert result['error'] == '用户名不能为空'

    def test_create_user_with_duplicate_username(self, db_session):
        """测试创建重复用户名的用户"""
        # 创建第一个用户
        UserService.create_user('testuser', 'password123', 'Test User 1')

        # 尝试创建同名用户
        user_data = {
            'username': 'testuser',
            'nickname': 'Test User 2'
        }

        result = UserService.create_user_with_roles(user_data)
        assert 'error' in result
        assert result['error'] == '用户名已存在'

    def test_update_user_with_roles_user_not_found(self, db_session):
        """测试更新不存在的用户"""
        update_data = {
            'username': 'updateduser'
        }

        result = UserService.update_user_with_roles(999999, update_data)
        assert 'error' in result
        assert result['error'] == '用户不存在'

    def test_update_user_with_duplicate_username(self, db_session):
        """测试更新用户时使用已存在的用户名"""
        # 创建两个用户
        UserService.create_user('user1', 'password123', 'User One')
        user2 = UserService.create_user('user2', 'password123', 'User Two')

        # 尝试将user2的用户名更新为user1的用户名
        update_data = {
            'username': 'user1'
        }

        result = UserService.update_user_with_roles(user2.id, update_data)
        assert 'error' in result
        assert result['error'] == '用户名已存在'

    def test_update_password_user_not_found(self, db_session):
        """测试更新不存在用户的密码"""
        result = UserService.update_password(999999, 'newpassword')
        assert result is False

    def test_get_user_by_username_deleted(self, db_session):
        """测试获取已删除的用户"""
        # 创建测试用户并标记为已删除
        from app.models import SysUser
        user = SysUser(
            username='deleteduser',
            password='password123',
            nickname='Deleted User',
            deleted=1
        )
        db_session.add(user)
        db_session.commit()

        retrieved_user = UserService.get_user_by_username('deleteduser')
        assert retrieved_user is None

    def test_get_user_by_id_deleted(self, db_session):
        """测试根据ID获取已删除的用户"""
        # 创建测试用户并标记为已删除
        from app.models import SysUser
        user = SysUser(
            username='deleteduser',
            password='password123',
            nickname='Deleted User',
            deleted=1
        )
        db_session.add(user)
        db_session.commit()

        retrieved_user = UserService.get_user_by_id(user.id)
        assert retrieved_user is None

    def test_get_user_by_id_not_found(self, db_session):
        """测试根据ID获取不存在的用户"""
        retrieved_user = UserService.get_user_by_id(999999)
        assert retrieved_user is None

    def test_authenticate_user_deleted(self, db_session):
        """测试认证已删除的用户"""
        # 创建测试用户并标记为已删除
        from app.models import SysUser
        user = SysUser(
            username='deleteduser',
            password=UserService._hash_password('password123'),
            nickname='Deleted User',
            deleted=1
        )
        db_session.add(user)
        db_session.commit()

        authenticated_user = UserService.authenticate_user('deleteduser', 'password123')
        assert authenticated_user is None

    def test_get_user_form_data_user_not_found(self, db_session):
        """测试获取不存在用户的表单数据"""
        form_data = UserService.get_user_form_data(999999)
        assert form_data is None

    def test_hash_password(self, db_session):
        """测试密码哈希"""
        password = 'testpassword'
        hashed = UserService._hash_password(password)
        assert hashed.startswith('$2b$')  # bcrypt哈希前缀
        assert UserService._check_password(password, hashed) is True
        assert UserService._check_password('wrongpassword', hashed) is False

    def test_update_user_status(self, db_session):
        """测试更新用户状态"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        assert user.status == 1  # 默认状态

        # 禁用用户
        result = UserService.update_user_status(user.id, 0)
        assert result is True

        updated_user = UserService.get_user_by_id(user.id)
        assert updated_user.status == 0

        # 启用用户
        result = UserService.update_user_status(user.id, 1)
        assert result is True

        updated_user = UserService.get_user_by_id(user.id)
        assert updated_user.status == 1

    def test_update_user_status_user_not_found(self, db_session):
        """测试更新不存在用户的状态"""
        result = UserService.update_user_status(999999, 1)
        assert result is False

    def test_delete_user(self, db_session):
        """测试删除用户（逻辑删除）"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        assert user.deleted == 0

        # 删除用户
        result = UserService.delete_user(user.id)
        assert result is True

        # 验证用户已被逻辑删除
        deleted_user = UserService.get_user_by_id(user.id)
        assert deleted_user is None  # get_user_by_id过滤已删除用户

        # 直接从数据库查询确认deleted字段已更新
        from app.models import SysUser
        deleted_user = db_session.query(SysUser).get(user.id)
        assert deleted_user.deleted == 1

    def test_delete_user_not_found(self, db_session):
        """测试删除不存在的用户"""
        result = UserService.delete_user(999999)
        assert result is False

    def test_update_user_status(self, db_session):
        """测试更新用户状态"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        assert user.status == 1  # 默认状态为启用

        # 禁用用户
        result = UserService.update_user_status(user.id, 0)
        assert result is True

        # 验证状态已更新
        updated_user = UserService.get_user_by_id(user.id)
        assert updated_user.status == 0

        # 启用用户
        result = UserService.update_user_status(user.id, 1)
        assert result is True

        # 验证状态已更新
        updated_user = UserService.get_user_by_id(user.id)
        assert updated_user.status == 1

    def test_delete_user(self, db_session):
        """测试删除用户"""
        # 创建测试用户
        user = UserService.create_user('testuser', 'password123', 'Test User')
        assert user.deleted == 0  # 默认未删除

        # 删除用户
        result = UserService.delete_user(user.id)
        assert result is True

        # 验证用户已被标记为删除
        updated_user = UserService.get_user_by_id(user.id)
        assert updated_user is None or updated_user.deleted == 1

        # 验证无法再通过用户名获取到该用户
        user_by_name = UserService.get_user_by_username('testuser')
        assert user_by_name is None
