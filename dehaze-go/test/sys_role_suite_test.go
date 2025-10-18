package test

import (
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

	// 无需手动清理 - 事务会自动回滚
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

// 运行测试套件
func TestRoleServiceSuite(t *testing.T) {
	suite.Run(t, new(RoleServiceTestSuite))
}
