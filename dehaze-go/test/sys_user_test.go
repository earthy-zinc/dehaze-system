package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/suite"
)

// UserServiceTestSuite 用户服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type UserServiceTestSuite struct {
	TransactionTestSuite
	userService *service.UserService
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *UserServiceTestSuite) SetupSuite() {
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
	s.userService = &service.UserService{}

	// 确保必要的表已创建
	initialize.Migrate()
}

// TestGetUserAuthInfo_UserNotFound 测试用户不存在的情况
func (s *UserServiceTestSuite) TestGetUserAuthInfo_UserNotFound() {
	userAuthInfo, err := s.userService.GetUserAuthInfo("nonexistent")
	s.AssertError(err)
	s.AssertNil(userAuthInfo)
}

// TestGetUserAuthInfo_UserExistsWithoutRoles 测试用户存在但没有角色
func (s *UserServiceTestSuite) TestGetUserAuthInfo_UserExistsWithoutRoles() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_user_no_roles",
		Nickname: "Test User No Roles",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 调用测试方法
	userAuthInfo, err := s.userService.GetUserAuthInfo(testUser.Username)
	s.AssertNoError(err)
	s.AssertNotNil(userAuthInfo)
	s.AssertEqual(testUser.ID, userAuthInfo.UserId)
	s.AssertEqual(testUser.Username, userAuthInfo.Username)
	s.AssertEqual(testUser.Nickname, userAuthInfo.Nickname)
	s.AssertEqual(testUser.Password, userAuthInfo.Password)
	s.AssertEqual(testUser.Status, userAuthInfo.Status)
	s.AssertEqual(testUser.DeptID, userAuthInfo.DeptId)
	s.Assert().Empty(userAuthInfo.Roles)
	s.Assert().Empty(userAuthInfo.Perms)
	s.AssertEqual(int8(0), userAuthInfo.DataScope)

}

// TestGetUserAuthInfo_UserExistsWithRolesNoPerms 测试用户存在且有角色但角色无权限
func (s *UserServiceTestSuite) TestGetUserAuthInfo_UserExistsWithRolesNoPerms() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_user_with_roles",
		Nickname: "Test User With Roles",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "Test Role",
		Code:      "TEST_ROLE",
		Status:    1,
		DataScope: 2,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 创建用户角色关联
	userRole := &model.SysUserRole{
		UserID: testUser.ID,
		RoleID: testRole.ID,
	}
	s.AssertNoError(s.GetDB().Table("sys_user_role").Create(userRole).Error)

	// 调用测试方法
	userAuthInfo, err := s.userService.GetUserAuthInfo(testUser.Username)
	s.AssertNoError(err)
	s.AssertNotNil(userAuthInfo)
	s.AssertEqual(testUser.ID, userAuthInfo.UserId)
	s.AssertEqual(testUser.Username, userAuthInfo.Username)
	s.AssertEqual(testUser.Nickname, userAuthInfo.Nickname)
	s.AssertEqual(testUser.Password, userAuthInfo.Password)
	s.AssertEqual(testUser.Status, userAuthInfo.Status)
	s.AssertEqual(testUser.DeptID, userAuthInfo.DeptId)
	s.Assert().Contains(userAuthInfo.Roles, testRole.Code)
	s.Assert().Empty(userAuthInfo.Perms)
	s.AssertEqual(testRole.DataScope, userAuthInfo.DataScope)

}

// TestGetUserAuthInfo_UserExistsWithRolesAndPerms 测试用户存在且有角色和权限
func (s *UserServiceTestSuite) TestGetUserAuthInfo_UserExistsWithRolesAndPerms() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_user_with_perms",
		Nickname: "Test User With Perms",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 创建测试角色
	testRole := &model.SysRole{
		Name:      "Test Role With Perms",
		Code:      "TEST_ROLE_PERMS",
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}
	s.AssertNoError(s.CreateTestData(testRole))

	// 创建测试菜单权限
	testMenu := &model.SysMenu{
		Name: "Test Menu",
		Type: 4, // 按钮类型
		Perm: "test:permission",
	}
	s.AssertNoError(s.CreateTestData(testMenu))

	// 创建用户角色关联
	userRole := &model.SysUserRole{
		UserID: testUser.ID,
		RoleID: testRole.ID,
	}
	s.AssertNoError(s.GetDB().Table("sys_user_role").Create(userRole).Error)

	// 创建角色菜单关联
	roleMenu := &model.SysRoleMenu{
		RoleID: testRole.ID,
		MenuID: testMenu.ID,
	}
	s.AssertNoError(s.GetDB().Table("sys_role_menu").Create(roleMenu).Error)

	// 调用测试方法
	userAuthInfo, err := s.userService.GetUserAuthInfo(testUser.Username)
	s.AssertNoError(err)
	s.AssertNotNil(userAuthInfo)
	s.AssertEqual(testUser.ID, userAuthInfo.UserId)
	s.AssertEqual(testUser.Username, userAuthInfo.Username)
	s.AssertEqual(testUser.Nickname, userAuthInfo.Nickname)
	s.AssertEqual(testUser.Password, userAuthInfo.Password)
	s.AssertEqual(testUser.Status, userAuthInfo.Status)
	s.AssertEqual(testUser.DeptID, userAuthInfo.DeptId)
	s.Assert().Contains(userAuthInfo.Roles, testRole.Code)
	s.Assert().Contains(userAuthInfo.Perms, testMenu.Perm)
	s.AssertEqual(int8(testRole.DataScope), userAuthInfo.DataScope)

}

// TestGetUserAuthInfo_UserExistsButDisabled 测试用户存在但已被禁用
func (s *UserServiceTestSuite) TestGetUserAuthInfo_UserExistsButDisabled() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_user_disabled",
		Nickname: "Test User Disabled",
		Password: "test_password",
		Status:   0, // 禁用状态
		DeptID:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 调用测试方法
	userAuthInfo, err := s.userService.GetUserAuthInfo(testUser.Username)
	s.AssertNoError(err)
	s.AssertNotNil(userAuthInfo)
	s.AssertEqual(testUser.ID, userAuthInfo.UserId)
	s.AssertEqual(testUser.Username, userAuthInfo.Username)
	s.AssertEqual(testUser.Nickname, userAuthInfo.Nickname)
	s.AssertEqual(testUser.Password, userAuthInfo.Password)
	s.AssertEqual(testUser.Status, userAuthInfo.Status)
	s.AssertEqual(testUser.DeptID, userAuthInfo.DeptId)

}

// TestGetUserAuthInfo_UserExistsButDeleted 测试用户存在但已被逻辑删除
func (s *UserServiceTestSuite) TestGetUserAuthInfo_UserExistsButDeleted() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_user_deleted",
		Nickname: "Test User Deleted",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  1, // 已删除
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 调用测试方法
	userAuthInfo, err := s.userService.GetUserAuthInfo(testUser.Username)
	s.AssertError(err)
	s.AssertNil(userAuthInfo)

}

// TestLogin_InvalidCredentials 测试用户名或密码错误
func (s *UserServiceTestSuite) TestLogin_InvalidCredentials() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_login_user",
		Nickname: "Test Login User",
		// 使用bcrypt加密密码
		Password: "$2a$10$N47IXmT8C.sKUFXs1EBS9uJf8JiKEGz4rY14M1SX3w2w1aW99Mj9K", // "password"的bcrypt hash
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 尝试使用错误密码登录
	loginUser := &model.SysUser{
		Username: "test_login_user",
		Password: "wrong_password",
	}
	userAuthInfo, err := s.userService.Login(loginUser)

	// 验证结果
	s.AssertError(err)
	s.AssertNil(userAuthInfo)

}

// TestLogin_ValidLogin 测试正常登录
func (s *UserServiceTestSuite) TestLogin_ValidLogin() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_valid_login",
		Nickname: "Test Valid Login",
		// 使用bcrypt加密密码
		Password: "$2a$10$BQm8di9VTUfOlmr/VcFyB.BhurfGZVjCXYdgDPN1ZeI0yEMeByAQq", // "password"的bcrypt hash
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testUser))

	// 使用正确密码登录
	loginUser := &model.SysUser{
		Username: "test_valid_login",
		Password: "password",
	}
	userAuthInfo, err := s.userService.Login(loginUser)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(userAuthInfo)
	s.AssertEqual(testUser.ID, userAuthInfo.UserId)
	s.AssertEqual(testUser.Username, userAuthInfo.Username)

}

// 运行测试套件
func TestUserService(t *testing.T) {
	suite.Run(t, new(UserServiceTestSuite))
}
