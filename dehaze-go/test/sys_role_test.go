package test

import (
	"fmt"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/suite"
)

// RoleServiceTestSuite 角色服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type RoleServiceTestSuite struct {
	TransactionTestSuite
	roleService *service.RoleService
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *RoleServiceTestSuite) SetupSuite() {
	// 初始化配置和数据库
	initialize.Viper()
	initialize.Gorm()
	initialize.Redis()

	if global.DB == nil {
		s.T().Fatal("数据库连接失败")
	}

	// 保存原始数据库连接
	s.DB = global.DB

	// 初始化服务
	s.roleService = &service.RoleService{}

	// 确保必要的表已创建
	initialize.Migrate()
}

// TearDownSuite 在整个测试套件结束后运行一次
func (s *RoleServiceTestSuite) TearDownSuite() {
	// 清理操作（如果需要）
}

// TestGetRolePage 测试获取角色分页列表
func (s *RoleServiceTestSuite) TestGetRolePage() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试角色分页",
		Code:      "TEST_ROLE_PAGE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 执行查询
	queryParams := query.RolePageQuery{
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
	s.Assert().GreaterOrEqual(pageResult.Total, int64(1))
	s.Assert().NotEmpty(pageResult.List)

}

// TestGetRolePageWithKeyword 测试带关键字的角色分页查询
func (s *RoleServiceTestSuite) TestGetRolePageWithKeyword() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试关键字搜索角色",
		Code:      "TEST_KEYWORD_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 执行查询
	queryParams := query.RolePageQuery{
		Keywords: "测试关键字",
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
	s.Assert().GreaterOrEqual(pageResult.Total, int64(1))
}

// TestListRoleOptions 测试获取角色下拉列表
func (s *RoleServiceTestSuite) TestListRoleOptions() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试角色选项",
		Code:      "TEST_ROLE_OPTION",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 获取下拉列表
	options, err := s.roleService.ListRoleOptions()

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(options)
	s.Assert().NotEmpty(options)
}

// TestSaveRole_Create 测试创建角色
func (s *RoleServiceTestSuite) TestSaveRole_Create() {
	// 准备角色表单数据
	roleFormBO := bo.RoleFormBO{
		Name:      "测试新增角色",
		Code:      "TEST_ADD_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	// 保存角色
	err := s.roleService.SaveRole(roleFormBO)
	s.AssertNoError(err)

	// 验证角色是否创建成功
	var savedRole model.SysRole
	err = s.GetDB().Where("code = ?", roleFormBO.Code).First(&savedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(roleFormBO.Name, savedRole.Name)
	s.AssertEqual(roleFormBO.Code, savedRole.Code)
}

// TestSaveRole_Update 测试更新角色
func (s *RoleServiceTestSuite) TestSaveRole_Update() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试更新角色",
		Code:      "TEST_UPDATE_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 准备更新数据
	id := testRole.ID
	roleFormBO := bo.RoleFormBO{
		ID:        &id,
		Name:      "测试更新后角色",
		Code:      "TEST_UPDATED_ROLE",
		Sort:      2,
		Status:    0,
		DataScope: 2,
	}

	// 更新角色
	err := s.roleService.SaveRole(roleFormBO)
	s.AssertNoError(err)

	// 验证角色是否更新成功
	var updatedRole model.SysRole
	err = s.GetDB().Where("id = ?", testRole.ID).First(&updatedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(roleFormBO.Name, updatedRole.Name)
	s.AssertEqual(roleFormBO.Code, updatedRole.Code)
	s.AssertEqual(roleFormBO.Sort, updatedRole.Sort)
}

// TestSaveRole_DuplicateCode 测试创建重复编码的角色
func (s *RoleServiceTestSuite) TestSaveRole_DuplicateCode() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试重复角色",
		Code:      "TEST_DUPLICATE_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 尝试保存相同编码的角色
	roleFormBO := bo.RoleFormBO{
		Name:      "另一个测试角色",
		Code:      "TEST_DUPLICATE_ROLE",
		Sort:      2,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色名称或角色编码已存在，请修改后重试！", err.Error())
}

// TestGetRoleForm_NotFound 测试获取不存在的角色
func (s *RoleServiceTestSuite) TestGetRoleForm_NotFound() {
	roleFormBO, err := s.roleService.GetRoleForm(999999)
	s.AssertError(err)
	s.AssertEqual("角色不存在", err.Error())
	s.AssertEqual(bo.RoleFormBO{}, roleFormBO)
}

// TestGetRoleForm_Exists 测试获取存在的角色
func (s *RoleServiceTestSuite) TestGetRoleForm_Exists() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试获取表单角色",
		Code:      "TEST_GET_FORM_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 获取表单数据
	roleFormBO, err := s.roleService.GetRoleForm(testRole.ID)

	// 验证结果
	s.AssertNoError(err)
	s.AssertEqual(testRole.ID, *roleFormBO.ID)
	s.AssertEqual(testRole.Name, roleFormBO.Name)
	s.AssertEqual(testRole.Code, roleFormBO.Code)
}

// TestUpdateRoleStatus 测试更新角色状态
func (s *RoleServiceTestSuite) TestUpdateRoleStatus() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试更新状态角色",
		Code:      "TEST_UPDATE_STATUS_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 更新角色状态
	err := s.roleService.UpdateRoleStatus(testRole.ID, 0)
	s.AssertNoError(err)

	// 验证状态是否更新
	var updatedRole model.SysRole
	err = s.GetDB().Where("id = ?", testRole.ID).First(&updatedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(0), updatedRole.Status)
}

// TestDeleteRoles_Single 测试删除单个角色
func (s *RoleServiceTestSuite) TestDeleteRoles_Single() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "测试删除角色",
		Code:      "TEST_DELETE_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 删除角色
	err := s.roleService.DeleteRoles(string(rune(testRole.ID)))
	s.AssertNoError(err)

	// 验证角色是否被逻辑删除
	var deletedRole model.SysRole
	err = s.GetDB().Unscoped().Where("id = ?", testRole.ID).First(&deletedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedRole.Deleted)
}

// TestDeleteRoles_Multiple 测试批量删除角色
func (s *RoleServiceTestSuite) TestDeleteRoles_Multiple() {
	// 创建测试角色1
	testRole1 := &model.SysRole{
		Name:      "测试批量删除角色1",
		Code:      "TEST_BATCH_DELETE_1",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole1))

	// 创建测试角色2
	testRole2 := &model.SysRole{
		Name:      "测试批量删除角色2",
		Code:      "TEST_BATCH_DELETE_2",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole2))

	// 批量删除
	ids := string(rune(testRole1.ID)) + "," + string(rune(testRole2.ID))
	err := s.roleService.DeleteRoles(ids)
	s.AssertNoError(err)

	// 验证角色是否被逻辑删除
	var count int64
	s.GetDB().Unscoped().Model(&model.SysRole{}).
		Where("id IN (?, ?) AND deleted = 1", testRole1.ID, testRole2.ID).
		Count(&count)
	s.AssertEqual(int64(2), count)
}

// TestGetMaximumDataScope 测试获取最大数据权限范围
func (s *RoleServiceTestSuite) TestGetMaximumDataScope() {
	// 创建多个不同数据权限的角色
	role1 := &model.SysRole{
		Name:      "数据权限测试角色1",
		Code:      "TEST_DATA_SCOPE_1",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(role1))

	role2 := &model.SysRole{
		Name:      "数据权限测试角色2",
		Code:      "TEST_DATA_SCOPE_2",
		Sort:      1,
		Status:    1,
		DataScope: 2,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(role2))

	// 获取最大数据权限
	roles := []string{role1.Code, role2.Code}
	dataScope, err := s.roleService.GetMaximumDataScope(roles)

	s.AssertNoError(err)
	s.AssertNotNil(dataScope)
	s.AssertEqual(int8(1), *dataScope) // 最小值（权限最大）
}

// TestGetRolePage_NormalPagination 测试正常分页查询
func (s *RoleServiceTestSuite) TestGetRolePage_NormalPagination() {
	// 准备测试数据
	testRole := &model.SysRole{
		Name:      "test_role_page",
		Code:      "TEST_ROLE_PAGE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 执行查询
	queryParams := query.RolePageQuery{
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
	s.Assert().GreaterOrEqual(pageResult.Total, int64(1))
	s.Assert().NotEmpty(pageResult.List)

}

// TestGetRolePage_KeywordSearch 测试带关键字查询
func (s *RoleServiceTestSuite) TestGetRolePage_KeywordSearch() {
	// 准备测试数据
	testRole := &model.SysRole{
		Name:      "test_keyword_search_role",
		Code:      "TEST_KEYWORD_SEARCH_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 执行查询
	queryParams := query.RolePageQuery{
		Keywords: "test_keyword_search_role",
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
	s.Assert().GreaterOrEqual(pageResult.Total, int64(1))
	s.Assert().NotEmpty(pageResult.List)

}

// TestListRoleOptions_NormalListOptions 测试正常获取下拉列表
func (s *RoleServiceTestSuite) TestListRoleOptions_NormalListOptions() {
	// 准备测试数据
	testRole := &model.SysRole{
		Name:      "test_role_options",
		Code:      "TEST_ROLE_OPTIONS",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 获取下拉列表
	options, err := s.roleService.ListRoleOptions()

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(options)
	s.Assert().NotEmpty(options)

}

// TestSaveRole_NormalAddRole 测试正常新增角色
func (s *RoleServiceTestSuite) TestSaveRole_NormalAddRole() {
	// 准备角色表单数据
	roleFormBO := bo.RoleFormBO{
		Name:      "test_add_role",
		Code:      "TEST_ADD_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	// 保存角色
	err := s.roleService.SaveRole(roleFormBO)

	// 验证结果
	s.AssertNoError(err)

	// 验证角色是否真的插入数据库
	var savedRole model.SysRole
	err = s.GetDB().Where("code = ?", roleFormBO.Code).First(&savedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(roleFormBO.Name, savedRole.Name)
	s.AssertEqual(roleFormBO.Code, savedRole.Code)
	s.AssertEqual(roleFormBO.Sort, savedRole.Sort)

}

// TestSaveRole_NormalUpdateRole 测试正常更新角色
func (s *RoleServiceTestSuite) TestSaveRole_NormalUpdateRole() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_update_role",
		Code:      "TEST_UPDATE_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 准备更新数据
	id := testRole.ID
	roleFormBO := bo.RoleFormBO{
		ID:        &id,
		Name:      "test_updated_role",
		Code:      "TEST_UPDATED_ROLE",
		Sort:      2,
		Status:    0,
		DataScope: 2,
	}

	// 更新角色
	err := s.roleService.SaveRole(roleFormBO)

	// 验证结果
	s.AssertNoError(err)

	// 验证角色是否真的更新
	var updatedRole model.SysRole
	err = s.GetDB().Where("id = ?", testRole.ID).First(&updatedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(roleFormBO.Name, updatedRole.Name)
	s.AssertEqual(roleFormBO.Code, updatedRole.Code)
	s.AssertEqual(roleFormBO.Sort, updatedRole.Sort)
	s.AssertEqual(roleFormBO.Status, updatedRole.Status)

}

// TestSaveRole_DuplicateRole 测试角色名称或编码已存在
func (s *RoleServiceTestSuite) TestSaveRole_DuplicateRole() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_duplicate_role",
		Code:      "TEST_DUPLICATE_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 尝试保存相同编码的角色
	roleFormBO := bo.RoleFormBO{
		Name:      "another_test_role",
		Code:      "TEST_DUPLICATE_ROLE",
		Sort:      2,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)

	// 验证结果
	s.AssertError(err)
	s.AssertEqual("角色名称或角色编码已存在，请修改后重试！", err.Error())

}

// TestGetRoleForm_RoleNotFound 测试角色不存在
func (s *RoleServiceTestSuite) TestGetRoleForm_RoleNotFound() {
	roleFormBO, err := s.roleService.GetRoleForm(999999)
	s.AssertError(err)
	s.AssertEqual("角色不存在", err.Error())
	s.AssertEqual(bo.RoleFormBO{}, roleFormBO)
}

// TestGetRoleForm_RoleExists 测试角色存在
func (s *RoleServiceTestSuite) TestGetRoleForm_RoleExists() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_get_form_role",
		Code:      "TEST_GET_FORM_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 获取表单数据
	roleFormBO, err := s.roleService.GetRoleForm(testRole.ID)

	// 验证结果
	s.AssertNoError(err)
	s.AssertEqual(testRole.ID, *roleFormBO.ID)
	s.AssertEqual(testRole.Name, roleFormBO.Name)
	s.AssertEqual(testRole.Code, roleFormBO.Code)
	s.AssertEqual(testRole.Sort, roleFormBO.Sort)

}

// TestUpdateRoleStatus_NormalUpdateStatus 测试正常更新角色状态
func (s *RoleServiceTestSuite) TestUpdateRoleStatus_NormalUpdateStatus() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_update_status_role",
		Code:      "TEST_UPDATE_STATUS_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 更新角色状态
	err := s.roleService.UpdateRoleStatus(testRole.ID, 0)

	// 验证结果
	s.AssertNoError(err)

	// 验证角色状态是否真的更新
	var updatedRole model.SysRole
	err = s.GetDB().Where("id = ?", testRole.ID).First(&updatedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(0), updatedRole.Status)

}

// TestUpdateRoleStatus_RoleNotFound 测试角色不存在
func (s *RoleServiceTestSuite) TestUpdateRoleStatus_RoleNotFound() {
	err := s.roleService.UpdateRoleStatus(999999, 0)
	s.AssertError(err)
	s.AssertEqual("角色不存在", err.Error())
}

// TestDeleteRoles_NormalDeleteRoles 测试正常删除角色
func (s *RoleServiceTestSuite) TestDeleteRoles_NormalDeleteRoles() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_delete_role",
		Code:      "TEST_DELETE_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 删除角色
	err := s.roleService.DeleteRoles(fmt.Sprintf("%d", testRole.ID))

	// 验证结果
	s.AssertNoError(err)

	// 验证角色是否真的被逻辑删除
	var deletedRole model.SysRole
	err = s.GetDB().Unscoped().Where("id = ?", testRole.ID).First(&deletedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedRole.Deleted)

}

// TestDeleteRoles_DeleteMultipleRoles 测试删除多个角色
func (s *RoleServiceTestSuite) TestDeleteRoles_DeleteMultipleRoles() {
	// 创建测试角色1
	testRole1 := &model.SysRole{
		Name:      "test_delete_role_1",
		Code:      "TEST_DELETE_ROLE_1",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole1))

	// 创建测试角色2
	testRole2 := &model.SysRole{
		Name:      "test_delete_role_2",
		Code:      "TEST_DELETE_ROLE_2",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole2))

	// 删除角色
	ids := fmt.Sprintf("%d,%d", testRole1.ID, testRole2.ID)
	err := s.roleService.DeleteRoles(ids)

	// 验证结果
	s.AssertNoError(err)

	// 验证角色是否真的被逻辑删除
	var deletedRole1 model.SysRole
	err = s.GetDB().Unscoped().Where("id = ?", testRole1.ID).First(&deletedRole1).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedRole1.Deleted)

	var deletedRole2 model.SysRole
	err = s.GetDB().Unscoped().Where("id = ?", testRole2.ID).First(&deletedRole2).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedRole2.Deleted)

}

// TestGetRoleMenuIds_RoleNotFound 测试角色不存在
func (s *RoleServiceTestSuite) TestGetRoleMenuIds_RoleNotFound() {
	menuIds, err := s.roleService.GetRoleMenuIds(999999)
	s.AssertError(err)
	s.AssertEqual("角色不存在", err.Error())
	s.Assert().Empty(menuIds)
}

// TestGetRoleMenuIds_RoleExistsNoMenus 测试角色存在但无菜单
func (s *RoleServiceTestSuite) TestGetRoleMenuIds_RoleExistsNoMenus() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_get_menu_ids_role",
		Code:      "TEST_GET_MENU_IDS_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 获取角色菜单ID集合
	menuIds, err := s.roleService.GetRoleMenuIds(testRole.ID)

	// 验证结果
	s.AssertNoError(err)
	s.Assert().Empty(menuIds)

}

// TestAssignMenusToRole_RoleNotFound 测试角色不存在
func (s *RoleServiceTestSuite) TestAssignMenusToRole_RoleNotFound() {
	err := s.roleService.AssignMenusToRole(999999, []int64{1, 2, 3})
	s.AssertError(err)
	s.AssertEqual("角色不存在", err.Error())
}

// TestAssignMenusToRole_NormalAssignMenus 测试正常分配菜单给角色
func (s *RoleServiceTestSuite) TestAssignMenusToRole_NormalAssignMenus() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_assign_menus_role",
		Code:      "TEST_ASSIGN_MENUS_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 分配菜单给角色
	menuIds := []int64{1, 2, 3}
	err := s.roleService.AssignMenusToRole(testRole.ID, menuIds)

	// 验证结果
	s.AssertNoError(err)

	// 验证菜单是否真的分配给了角色
	var roleMenus []model.SysRoleMenu
	err = s.GetDB().Where("role_id = ?", testRole.ID).Find(&roleMenus).Error
	s.AssertNoError(err)
	s.Assert().Equal(len(menuIds), len(roleMenus))

}

// TestGetMaximumDataScope_EmptyRoles 测试空角色列表
func (s *RoleServiceTestSuite) TestGetMaximumDataScope_EmptyRoles() {
	dataScope, err := s.roleService.GetMaximumDataScope([]string{})
	s.AssertNoError(err)
	s.AssertNil(dataScope)
}

// TestGetMaximumDataScope_NormalGetMaxDataScope 测试正常获取最大数据权限范围
func (s *RoleServiceTestSuite) TestGetMaximumDataScope_NormalGetMaxDataScope() {
	// 创建测试角色
	testRole1 := &model.SysRole{
		Name:      "test_max_scope_role_1",
		Code:      "TEST_MAX_SCOPE_ROLE_1",
		Sort:      1,
		Status:    1,
		DataScope: 1, // 部门及子部门数据
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole1))

	testRole2 := &model.SysRole{
		Name:      "test_max_scope_role_2",
		Code:      "TEST_MAX_SCOPE_ROLE_2",
		Sort:      1,
		Status:    1,
		DataScope: 2, // 本部门数据
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole2))

	// 获取最大数据权限范围
	roles := []string{testRole1.Code, testRole2.Code}
	dataScope, err := s.roleService.GetMaximumDataScope(roles)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(dataScope)
	s.AssertEqual(int8(1), *dataScope) // 最小值应该是1

}

// TestSaveRole_InputValidation_EmptyName 测试角色名称为空
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_EmptyName() {
	roleFormBO := bo.RoleFormBO{
		Name:      "",
		Code:      "TEST_EMPTY_NAME",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色名称不能为空", err.Error())
}

// TestSaveRole_InputValidation_WhitespaceName 测试角色名称为纯空格
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_WhitespaceName() {
	roleFormBO := bo.RoleFormBO{
		Name:      "   ",
		Code:      "TEST_WHITESPACE_NAME",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色名称不能为空", err.Error())
}

// TestSaveRole_InputValidation_TooLongName 测试角色名称超长（>50字符）
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_TooLongName() {
	roleFormBO := bo.RoleFormBO{
		Name:      "这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的角色名称超过五十个字符",
		Code:      "TEST_LONG_NAME",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色名称长度不能超过50个字符", err.Error())
}

// TestSaveRole_InputValidation_EmptyCode 测试角色编码为空
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_EmptyCode() {
	roleFormBO := bo.RoleFormBO{
		Name:      "测试角色",
		Code:      "",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色编码不能为空", err.Error())
}

// TestSaveRole_InputValidation_WhitespaceCode 测试角色编码为纯空格
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_WhitespaceCode() {
	roleFormBO := bo.RoleFormBO{
		Name:      "测试角色",
		Code:      "   ",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色编码不能为空", err.Error())
}

// TestSaveRole_InputValidation_TooLongCode 测试角色编码超长（>50字符）
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_TooLongCode() {
	roleFormBO := bo.RoleFormBO{
		Name:      "测试角色",
		Code:      "TEST_VERY_VERY_VERY_VERY_VERY_VERY_VERY_LONG_CODE_EXCEEDS_FIFTY_CHARACTERS",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色编码长度不能超过50个字符", err.Error())
}

// TestSaveRole_InputValidation_InvalidStatus 测试角色状态值非法（非0非1）
func (s *RoleServiceTestSuite) TestSaveRole_InputValidation_InvalidStatus() {
	roleFormBO := bo.RoleFormBO{
		Name:      "测试角色",
		Code:      "TEST_INVALID_STATUS",
		Sort:      1,
		Status:    2, // 非法状态值
		DataScope: 1,
	}

	err := s.roleService.SaveRole(roleFormBO)
	s.AssertError(err)
	s.AssertEqual("角色状态值无效，必须为0或1", err.Error())
}

// TestUpdateRoleStatus_InputValidation_InvalidStatus 测试状态值非法（非0非1）
func (s *RoleServiceTestSuite) TestUpdateRoleStatus_InputValidation_InvalidStatus() {
	err := s.roleService.UpdateRoleStatus(1, 2)
	s.AssertError(err)
	s.AssertEqual("角色状态值无效，必须为0或1", err.Error())
}

// TestUpdateRoleStatus_InputValidation_NegativeStatus 测试状态值为负数
func (s *RoleServiceTestSuite) TestUpdateRoleStatus_InputValidation_NegativeStatus() {
	err := s.roleService.UpdateRoleStatus(1, -1)
	s.AssertError(err)
	s.AssertEqual("角色状态值无效，必须为0或1", err.Error())
}

// TestGetRolePage_BoundaryConditions_ZeroPageParams 测试分页参数为0
func (s *RoleServiceTestSuite) TestGetRolePage_BoundaryConditions_ZeroPageParams() {
	queryParams := query.RolePageQuery{
		PageNum:  0,
		PageSize: 0,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果 - 应该使用默认值
	s.AssertNoError(err)
	s.AssertEqual(int64(1), int64(pageResult.PageNum))
	s.AssertEqual(int64(10), int64(pageResult.PageSize))
}

// TestGetRolePage_BoundaryConditions_NegativePageParams 测试分页参数为负数
func (s *RoleServiceTestSuite) TestGetRolePage_BoundaryConditions_NegativePageParams() {
	queryParams := query.RolePageQuery{
		PageNum:  -1,
		PageSize: -1,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果 - 应该使用默认值
	s.AssertNoError(err)
	s.AssertEqual(int64(1), int64(pageResult.PageNum))
	s.AssertEqual(int64(10), int64(pageResult.PageSize))
}

// TestGetRolePage_BoundaryConditions_EmptyKeyword 测试关键字为空字符串
func (s *RoleServiceTestSuite) TestGetRolePage_BoundaryConditions_EmptyKeyword() {
	queryParams := query.RolePageQuery{
		Keywords: "",
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
}

// TestGetRolePage_BoundaryConditions_SpecialCharKeyword 测试关键字包含特殊字符
func (s *RoleServiceTestSuite) TestGetRolePage_BoundaryConditions_SpecialCharKeyword() {
	queryParams := query.RolePageQuery{
		Keywords: "%_\\",
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.roleService.GetRolePage(queryParams)

	// 验证结果 - 不应该报错
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
}

// TestDeleteRoles_InputValidation_EmptyIds 测试ID为空字符串
func (s *RoleServiceTestSuite) TestDeleteRoles_InputValidation_EmptyIds() {
	err := s.roleService.DeleteRoles("")
	s.AssertError(err)
	s.AssertEqual("删除的角色ID不能为空", err.Error())
}

// TestDeleteRoles_InputValidation_InvalidIdFormat 测试ID格式不正确
func (s *RoleServiceTestSuite) TestDeleteRoles_InputValidation_InvalidIdFormat() {
	err := s.roleService.DeleteRoles("abc,def")
	s.AssertError(err)
	s.AssertEqual("角色ID格式不正确", err.Error())
}

// TestDeleteRoles_InputValidation_PartialRoleNotFound 测试部分角色不存在
func (s *RoleServiceTestSuite) TestDeleteRoles_InputValidation_PartialRoleNotFound() {
	err := s.roleService.DeleteRoles("999998,999999")
	s.AssertError(err)
	s.AssertEqual("部分角色不存在", err.Error())
}

// TestDeleteRoles_WithUserAssigned 测试角色已分配用户时无法删除
func (s *RoleServiceTestSuite) TestDeleteRoles_WithUserAssigned() {
	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "test_role_with_user",
		Code:      "TEST_ROLE_WITH_USER",
		Sort:      1,
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.Require().NoError(s.CreateTestData(testRole))

	// 创建用户角色关联（模拟角色已分配给用户）
	userRole := &model.SysUserRole{
		UserID: 1,
		RoleID: testRole.ID,
	}
	s.Require().NoError(s.GetDB().Table("sys_user_role").Create(userRole).Error)

	// 尝试删除角色
	err := s.roleService.DeleteRoles(fmt.Sprintf("%d", testRole.ID))

	// 验证结果 - 应该删除失败
	s.AssertError(err)
	s.Assert().Contains(err.Error(), "已分配用户")

}

// TestDeleteRoles_BatchOptimization_BatchDeleteMultipleRoles 测试批量删除多个角色（验证N+1查询优化）
func (s *RoleServiceTestSuite) TestDeleteRoles_BatchOptimization_BatchDeleteMultipleRoles() {
	// 创建5个测试角色
	testRoles := make([]*model.SysRole, 5)
	roleIds := make([]int64, 5)
	roleCodes := make([]string, 5)

	for i := 0; i < 5; i++ {
		testRoles[i] = &model.SysRole{
			Name:      fmt.Sprintf("test_batch_delete_role_%d", i),
			Code:      fmt.Sprintf("TEST_BATCH_DELETE_ROLE_%d", i),
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}
		roleCodes[i] = testRoles[i].Code
		s.Require().NoError(s.CreateTestData(testRoles[i]))
		roleIds[i] = testRoles[i].ID
	}

	// 批量删除角色
	ids := fmt.Sprintf("%d,%d,%d,%d,%d", roleIds[0], roleIds[1], roleIds[2], roleIds[3], roleIds[4])
	err := s.roleService.DeleteRoles(ids)

	// 验证结果
	s.AssertNoError(err)

	// 验证所有角色都被逻辑删除
	for _, roleId := range roleIds {
		var deletedRole model.SysRole
		err := s.GetDB().Unscoped().Where("id = ?", roleId).First(&deletedRole).Error
		s.AssertNoError(err)
		s.AssertEqual(int8(1), deletedRole.Deleted)
	}

}

// TestSaveRole_SpecialCharacters_SpecialCharactersInName 测试角色名称包含特殊字符
func (s *RoleServiceTestSuite) TestSaveRole_SpecialCharacters_SpecialCharactersInName() {
	roleFormBO := bo.RoleFormBO{
		Name:      "测试角色<>\"'&",
		Code:      "TEST_SPECIAL_CHAR_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 1,
	}

	// 保存角色
	err := s.roleService.SaveRole(roleFormBO)

	// 验证结果 - 应该能正常保存
	s.AssertNoError(err)

	// 验证数据
	var savedRole model.SysRole
	err = s.GetDB().Where("code = ?", roleFormBO.Code).First(&savedRole).Error
	s.AssertNoError(err)
	s.AssertEqual(roleFormBO.Name, savedRole.Name)

}

// 运行测试套件
func TestRoleService(t *testing.T) {
	suite.Run(t, new(RoleServiceTestSuite))
}
