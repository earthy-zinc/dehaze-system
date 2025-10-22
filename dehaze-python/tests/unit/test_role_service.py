"""
角色服务测试
"""
import pytest
from app.models import SysRole
from app.service.role import RoleService


@pytest.mark.unit
@pytest.mark.requires_db
class TestRoleService:
    """角色服务测试类"""

    def test_create_role(self, db_session):
        """测试创建角色"""
        role_data = {
            'name': '测试角色',
            'code': 'TEST_ROLE',
            'sort': 1,
            'status': 1,
            'dataScope': 1
        }

        result = RoleService.create_role(role_data)
        assert 'error' not in result
        assert 'data' in result

        # 验证角色创建成功
        role = RoleService.get_role_by_id(result['data']['id'])
        assert role is not None
        assert role.name == '测试角色'
        assert role.code == 'TEST_ROLE'
        assert role.sort == 1
        assert role.status == 1
        assert role.data_scope == 1

    def test_create_role_with_duplicate_name(self, db_session):
        """测试创建重复名称的角色"""
        # 先创建一个角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 尝试创建同名角色
        role_data = {
            'name': '测试角色',
            'code': 'TEST_ROLE_2'
        }

        result = RoleService.create_role(role_data)
        assert 'error' in result
        assert result['error'] == '角色名称或编码已存在'

    def test_create_role_with_duplicate_code(self, db_session):
        """测试创建重复编码的角色"""
        # 先创建一个角色
        role = SysRole(name='测试角色1', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 尝试创建相同编码的角色
        role_data = {
            'name': '测试角色2',
            'code': 'TEST_ROLE'
        }

        result = RoleService.create_role(role_data)
        assert 'error' in result
        assert result['error'] == '角色名称或编码已存在'

    def test_get_role_by_id(self, db_session):
        """测试根据ID获取角色"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 获取角色
        retrieved_role = RoleService.get_role_by_id(role.id)
        assert retrieved_role is not None
        assert retrieved_role.id == role.id
        assert retrieved_role.name == '测试角色'
        assert retrieved_role.code == 'TEST_ROLE'

    def test_get_role_by_id_not_found(self, db_session):
        """测试获取不存在的角色"""
        role = RoleService.get_role_by_id(999999)
        assert role is None

    def test_update_role(self, db_session):
        """测试更新角色"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 更新角色
        update_data = {
            'name': '更新后的角色',
            'code': 'UPDATED_ROLE',
            'sort': 2,
            'status': 0,
            'dataScope': 2
        }

        result = RoleService.update_role(role.id, update_data)
        assert 'error' not in result

        # 验证更新成功
        updated_role = RoleService.get_role_by_id(role.id)
        assert updated_role.name == '更新后的角色'
        assert updated_role.code == 'UPDATED_ROLE'
        assert updated_role.sort == 2
        assert updated_role.status == 0
        assert updated_role.data_scope == 2

    def test_update_role_not_found(self, db_session):
        """测试更新不存在的角色"""
        update_data = {
            'name': '更新后的角色',
            'code': 'UPDATED_ROLE'
        }

        result = RoleService.update_role(999999, update_data)
        assert 'error' in result
        assert result['error'] == '角色不存在'

    def test_update_role_with_duplicate_name(self, db_session):
        """测试更新角色时名称重复"""
        # 创建两个角色
        role1 = SysRole(name='测试角色1', code='TEST_ROLE_1')
        role2 = SysRole(name='测试角色2', code='TEST_ROLE_2')
        db_session.add(role1)
        db_session.add(role2)
        db_session.commit()

        # 尝试将role2的名称更新为role1的名称
        update_data = {
            'name': '测试角色1',
            'code': 'TEST_ROLE_2'
        }

        result = RoleService.update_role(role2.id, update_data)
        assert 'error' in result
        assert result['error'] == '角色名称或编码已存在'

    def test_get_role_list(self, db_session):
        """测试获取角色列表"""
        # 创建测试角色
        role1 = SysRole(name='角色1', code='ROLE_1', sort=1)
        role2 = SysRole(name='角色2', code='ROLE_2', sort=2)
        role3 = SysRole(name='测试角色', code='TEST_ROLE', sort=3)
        db_session.add(role1)
        db_session.add(role2)
        db_session.add(role3)
        db_session.commit()

        # 获取所有角色
        roles, total = RoleService.get_role_list()
        assert len(roles) == 3
        assert total == 3

        # 按名称搜索
        roles, total = RoleService.get_role_list(keywords='测试')
        assert len(roles) == 1
        assert total == 1
        assert roles[0].name == '测试角色'

        # 测试分页
        roles, total = RoleService.get_role_list(page=1, page_size=2)
        assert len(roles) == 2
        assert total == 3

    def test_get_role_options(self, db_session):
        """测试获取角色下拉选项"""
        # 创建测试角色
        role1 = SysRole(name='角色1', code='ROLE_1', sort=1, status=1)
        role2 = SysRole(name='角色2', code='ROLE_2', sort=2, status=1)
        role3 = SysRole(name='禁用角色', code='DISABLED_ROLE', sort=3, status=0)
        db_session.add(role1)
        db_session.add(role2)
        db_session.add(role3)
        db_session.commit()

        # 获取角色选项
        options = RoleService.get_role_options()
        assert len(options) == 2  # 不包含禁用的角色
        assert options[0]['label'] == '角色1'
        assert options[1]['label'] == '角色2'

    def test_delete_roles(self, db_session):
        """测试删除角色"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 删除角色
        result = RoleService.delete_roles(str(role.id))
        assert 'error' not in result

    def test_delete_roles_multiple(self, db_session):
        """测试批量删除角色"""
        # 创建测试角色
        role1 = SysRole(name='测试角色1', code='TEST_ROLE_1')
        role2 = SysRole(name='测试角色2', code='TEST_ROLE_2')
        db_session.add(role1)
        db_session.add(role2)
        db_session.commit()

        # 批量删除角色
        result = RoleService.delete_roles(f'{role1.id},{role2.id}')
        assert 'error' not in result

        # 验证角色已删除
        deleted_role1 = RoleService.get_role_by_id(role1.id)
        deleted_role2 = RoleService.get_role_by_id(role2.id)
        assert deleted_role1 is None
        assert deleted_role2 is None

    def test_delete_roles_not_found(self, db_session):
        """测试删除不存在的角色"""
        result = RoleService.delete_roles('999999')
        assert 'error' in result
        assert result['error'] == '角色ID 999999 不存在'

    def test_delete_roles_assigned_to_user(self, db_session):
        """测试删除已分配给用户的角色"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 模拟角色已分配给用户
        from app.models import SysUserRole
        user_role = SysUserRole(user_id=1, role_id=role.id)
        db_session.add(user_role)
        db_session.commit()

        # 尝试删除角色
        result = RoleService.delete_roles(str(role.id))
        assert 'error' in result
        assert '已分配给用户' in result['error']

    def test_update_role_status(self, db_session):
        """测试更新角色状态"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 启用角色
        result = RoleService.update_role_status(role.id, 1)
        assert 'error' not in result

        updated_role = RoleService.get_role_by_id(role.id)
        assert updated_role.status == 1

        # 禁用角色
        result = RoleService.update_role_status(role.id, 0)
        assert 'error' not in result

        updated_role = RoleService.get_role_by_id(role.id)
        assert updated_role.status == 0

    def test_update_role_status_invalid(self, db_session):
        """测试更新角色状态为无效值"""
        result = RoleService.update_role_status(1, 2)  # 无效状态
        assert 'error' in result
        assert result['error'] == '状态值只能为0或1'

    def test_update_role_status_not_found(self, db_session):
        """测试更新不存在的角色状态"""
        result = RoleService.update_role_status(999999, 1)
        assert 'error' in result
        assert result['error'] == '角色不存在'

    def test_get_role_list_pagination(self, db_session):
        """测试角色列表分页"""
        # 创建测试角色
        for i in range(15):
            role = SysRole(name=f'测试角色{i}', code=f'TEST_ROLE_{i}', sort=i)
            db_session.add(role)
        db_session.commit()

        # 测试第一页
        roles, total = RoleService.get_role_list(page=1, page_size=10)
        assert len(roles) == 10
        assert total == 15

        # 测试第二页
        roles, total = RoleService.get_role_list(page=2, page_size=10)
        assert len(roles) == 5
        assert total == 15

    def test_get_role_list_empty(self, db_session):
        """测试获取空角色列表"""
        roles, total = RoleService.get_role_list()
        assert len(roles) == 0
        assert total == 0

    def test_get_role_options_empty(self, db_session):
        """测试获取空角色选项列表"""
        options = RoleService.get_role_options()
        assert len(options) == 0

    def test_get_role_options_only_active(self, db_session):
        """测试只获取启用的角色选项"""
        # 创建测试角色
        role1 = SysRole(name='启用角色', code='ACTIVE_ROLE', status=1)
        role2 = SysRole(name='禁用角色', code='INACTIVE_ROLE', status=0)
        db_session.add(role1)
        db_session.add(role2)
        db_session.commit()

        options = RoleService.get_role_options()
        assert len(options) == 1
        assert options[0]['label'] == '启用角色'

    def test_create_role_missing_fields(self, db_session):
        """测试创建角色时缺少必要字段"""
        # 缺少名称
        role_data = {
            'code': 'TEST_ROLE'
        }
        result = RoleService.create_role(role_data)
        assert 'error' in result
        assert result['error'] == '角色名称和编码不能为空'

        # 缺少编码
        role_data = {
            'name': '测试角色'
        }
        result = RoleService.create_role(role_data)
        assert 'error' in result
        assert result['error'] == '角色名称和编码不能为空'

    def test_update_role_missing_fields(self, db_session):
        """测试更新角色时缺少必要字段"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 缺少名称
        update_data = {
            'code': 'UPDATED_ROLE'
        }
        result = RoleService.update_role(role.id, update_data)
        assert 'error' in result
        assert result['error'] == '角色名称和编码不能为空'

        # 缺少编码
        update_data = {
            'name': '更新角色'
        }
        result = RoleService.update_role(role.id, update_data)
        assert 'error' in result
        assert result['error'] == '角色名称和编码不能为空'

    def test_get_role_by_id_deleted(self, db_session):
        """测试获取已删除的角色"""
        # 创建测试角色并标记为已删除
        role = SysRole(name='测试角色', code='TEST_ROLE', deleted=1)
        db_session.add(role)
        db_session.commit()

        retrieved_role = RoleService.get_role_by_id(role.id)
        assert retrieved_role is None

    def test_get_role_list_with_deleted(self, db_session):
        """测试角色列表不包含已删除的角色"""
        # 创建测试角色
        role1 = SysRole(name='正常角色', code='NORMAL_ROLE', deleted=0)
        role2 = SysRole(name='已删除角色', code='DELETED_ROLE', deleted=1)
        db_session.add(role1)
        db_session.add(role2)
        db_session.commit()

        roles, total = RoleService.get_role_list()
        assert len(roles) == 1
        assert total == 1
        assert roles[0].name == '正常角色'

    def test_delete_roles_assigned_to_user(self, db_session):
        """测试删除已分配给用户的角色"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 模拟角色已分配给用户
        from app.models import SysUserRole
        user_role = SysUserRole(user_id=1, role_id=role.id)
        db_session.add(user_role)
        db_session.commit()

        # 尝试删除角色
        result = RoleService.delete_roles(str(role.id))
        assert 'error' in result
        assert '已分配给用户' in result['error']

        # 验证角色未被删除
        not_deleted_role = RoleService.get_role_by_id(role.id)
        assert not_deleted_role is not None
        assert not_deleted_role.deleted == 0

    def test_delete_roles_not_found(self, db_session):
        """测试删除不存在的角色"""
        result = RoleService.delete_roles('999999')
        assert 'error' in result
        assert result['error'] == '角色ID 999999 不存在'

    def test_update_role_status(self, db_session):
        """测试更新角色状态"""
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE', status=1)
        db_session.add(role)
        db_session.commit()
        assert role.status == 1  # 默认启用

        # 禁用角色
        result = RoleService.update_role_status(role.id, 0)
        assert 'error' not in result

        # 验证状态已更新
        updated_role = RoleService.get_role_by_id(role.id)
        assert updated_role.status == 0

        # 启用角色
        result = RoleService.update_role_status(role.id, 1)
        assert 'error' not in result

        # 验证状态已更新
        updated_role = RoleService.get_role_by_id(role.id)
        assert updated_role.status == 1

    def test_update_role_status_invalid_status(self, db_session):
        """测试更新角色状态时传入无效状态值"""
        result = RoleService.update_role_status(1, 2)  # 2是无效状态
        assert 'error' in result
        assert result['error'] == '状态值只能为0或1'

    def test_assign_menus_to_role(self, db_session):
        """测试分配菜单给角色"""
        from app.models import SysRoleMenu
        
        # 创建测试角色
        role = SysRole(name='测试角色', code='TEST_ROLE')
        db_session.add(role)
        db_session.commit()

        # 分配菜单
        menu_ids = [1, 2, 3]
        result = RoleService.assign_menus_to_role(role.id, menu_ids)
        assert 'error' not in result

        # 验证菜单已分配
        assigned_menu_ids = RoleService.get_role_menu_ids(role.id)
        assert set(assigned_menu_ids) == set(menu_ids)

        # 重新分配菜单
        new_menu_ids = [3, 4, 5]
        result = RoleService.assign_menus_to_role(role.id, new_menu_ids)
        assert 'error' not in result

        # 验证菜单已更新
        assigned_menu_ids = RoleService.get_role_menu_ids(role.id)
        assert set(assigned_menu_ids) == set(new_menu_ids)

    def test_get_maximum_data_scope(self, db_session):
        """测试获取最大范围的数据权限"""
        # 创建测试角色
        role1 = SysRole(name='角色1', code='ROLE_1', data_scope=1)  # 数据权限1
        role2 = SysRole(name='角色2', code='ROLE_2', data_scope=2)  # 数据权限2
        role3 = SysRole(name='角色3', code='ROLE_3', data_scope=3)  # 数据权限3
        db_session.add(role1)
        db_session.add(role2)
        db_session.add(role3)
        db_session.commit()

        # 测试获取最大数据权限范围
        roles = ['ROLE_1', 'ROLE_2', 'ROLE_3']
        max_scope = RoleService.get_maximum_data_scope(roles)
        assert max_scope == 1  # 应该返回最小值，即最大权限范围

        # 测试部分角色
        roles = ['ROLE_2', 'ROLE_3']
        max_scope = RoleService.get_maximum_data_scope(roles)
        assert max_scope == 2

        # 测试空角色列表
        max_scope = RoleService.get_maximum_data_scope([])
        assert max_scope is None

        # 测试不存在的角色
        roles = ['NON_EXISTENT_ROLE']
        max_scope = RoleService.get_maximum_data_scope(roles)
        assert max_scope is None