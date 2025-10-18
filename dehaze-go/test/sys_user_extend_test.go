package test

import (
	"fmt"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/suite"
	"golang.org/x/crypto/bcrypt"
)

// UserServiceExtendTraditionalTestSuite 用户扩展服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type UserServiceExtendTraditionalTestSuite struct {
	TransactionTestSuite
	userService *service.UserServiceExtend
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *UserServiceExtendTraditionalTestSuite) SetupSuite() {
	// 初始化服务
	s.userService = &service.UserServiceExtend{}

	// 检查数据库连接是否可用
	if global.DB == nil {
		s.T().Skip("数据库连接不可用，跳过测试")
	}
}

// TestListPagedUsers_NormalPagination 测试正常分页查询
func (s *UserServiceExtendTraditionalTestSuite) TestListPagedUsers_NormalPagination() {
	// 准备测试数据
	testUser := &model.SysUser{
		Username: "test_list_paged_user",
		Nickname: "Test List Paged User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 执行查询
	queryParams := query.UserPageQuery{
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.userService.ListPagedUsers(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
	s.Assert().GreaterOrEqual(pageResult.Total, int64(1))
	s.Assert().NotEmpty(pageResult.List)

}

// TestListPagedUsers_KeywordSearch 测试带关键字查询
func (s *UserServiceExtendTraditionalTestSuite) TestListPagedUsers_KeywordSearch() {
	// 准备测试数据
	testUser := &model.SysUser{
		Username: "test_keyword_search_user",
		Nickname: "Test Keyword Search User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 执行查询
	queryParams := query.UserPageQuery{
		Keywords: "test_keyword_search_user",
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.userService.ListPagedUsers(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(pageResult)
	s.Assert().GreaterOrEqual(pageResult.Total, int64(1))
	s.Assert().NotEmpty(pageResult.List)

}

// TestGetUserFormData_UserNotFound 测试用户不存在
func (s *UserServiceExtendTraditionalTestSuite) TestGetUserFormData_UserNotFound() {
	userFormBO, err := s.userService.GetUserFormData(999999)
	s.AssertError(err)
	s.AssertEqual("用户不存在", err.Error())
	s.AssertEqual(bo.UserFormBO{}, userFormBO)
}

// TestGetUserFormData_UserExists 测试用户存在
func (s *UserServiceExtendTraditionalTestSuite) TestGetUserFormData_UserExists() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_get_form_user",
		Nickname: "Test Get Form User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 获取表单数据
	userFormBO, err := s.userService.GetUserFormData(testUser.ID)

	// 验证结果
	s.AssertNoError(err)
	s.AssertEqual(testUser.ID, userFormBO.ID)
	s.AssertEqual(testUser.Username, userFormBO.Username)
	s.AssertEqual(testUser.Nickname, userFormBO.Nickname)
	s.AssertEqual(testUser.DeptID, userFormBO.DeptID)

}

// TestSaveUser_NormalSaveUser 测试正常保存用户
func (s *UserServiceExtendTraditionalTestSuite) TestSaveUser_NormalSaveUser() {
	// 准备用户表单数据
	userFormBO := bo.UserFormBO{
		Username: "test_save_user",
		Nickname: "Test Save User",
		Gender:   1,
		DeptID:   1,
		Mobile:   "13800138000",
		Status:   1,
		Email:    "test@example.com",
		RoleIds:  []int64{},
	}

	// 保存用户
	err := s.userService.SaveUser(userFormBO)

	// 验证结果
	s.AssertNoError(err)

	// 验证用户是否真的插入数据库
	var savedUser model.SysUser
	err = s.GetDB().Where("username = ?", userFormBO.Username).First(&savedUser).Error
	s.AssertNoError(err)
	s.AssertEqual(userFormBO.Username, savedUser.Username)
	s.AssertEqual(userFormBO.Nickname, savedUser.Nickname)
	s.AssertEqual(userFormBO.Gender, savedUser.Gender)

}

// TestSaveUser_DuplicateUsername 测试用户名已存在
func (s *UserServiceExtendTraditionalTestSuite) TestSaveUser_DuplicateUsername() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_duplicate_user",
		Nickname: "Test Duplicate User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 尝试保存相同用户名的用户
	userFormBO := bo.UserFormBO{
		Username: "test_duplicate_user",
		Nickname: "Another Test User",
		Gender:   2,
		DeptID:   2,
		Mobile:   "13900139000",
		Status:   1,
		Email:    "another@example.com",
		RoleIds:  []int64{},
	}

	err := s.userService.SaveUser(userFormBO)

	// 验证结果
	s.AssertError(err)
	s.AssertEqual("用户名已存在", err.Error())

}

// TestUpdateUser_NormalUpdateUser 测试正常更新用户
func (s *UserServiceExtendTraditionalTestSuite) TestUpdateUser_NormalUpdateUser() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_update_user",
		Nickname: "Test Update User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 准备更新数据
	userFormBO := bo.UserFormBO{
		ID:       testUser.ID,
		Username: "test_updated_user",
		Nickname: "Test Updated User",
		Gender:   2,
		DeptID:   2,
		Mobile:   "13900139000",
		Status:   0,
		Email:    "updated@example.com",
		RoleIds:  []int64{},
	}

	// 更新用户
	err := s.userService.UpdateUser(testUser.ID, userFormBO)

	// 验证结果
	s.AssertNoError(err)

	// 验证用户是否真的更新
	var updatedUser model.SysUser
	err = s.GetDB().Where("id = ?", testUser.ID).First(&updatedUser).Error
	s.AssertNoError(err)
	s.AssertEqual(userFormBO.Username, updatedUser.Username)
	s.AssertEqual(userFormBO.Nickname, updatedUser.Nickname)
	s.AssertEqual(userFormBO.Gender, updatedUser.Gender)

}

// TestDeleteUsers_NormalDeleteUsers 测试正常删除用户
func (s *UserServiceExtendTraditionalTestSuite) TestDeleteUsers_NormalDeleteUsers() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_delete_user",
		Nickname: "Test Delete User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 删除用户
	err := s.userService.DeleteUsers(fmt.Sprintf("%d", testUser.ID))

	// 验证结果
	s.AssertNoError(err)

	// 验证用户是否真的被逻辑删除
	var deletedUser model.SysUser
	err = s.GetDB().Unscoped().Where("id = ?", testUser.ID).First(&deletedUser).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedUser.Deleted)

}

// TestDeleteUsers_DeleteMultipleUsers 测试删除多个用户
func (s *UserServiceExtendTraditionalTestSuite) TestDeleteUsers_DeleteMultipleUsers() {
	// 创建测试用户1
	testUser1 := &model.SysUser{
		Username: "test_delete_user_1",
		Nickname: "Test Delete User 1",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser1))

	// 创建测试用户2
	testUser2 := &model.SysUser{
		Username: "test_delete_user_2",
		Nickname: "Test Delete User 2",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser2))

	// 删除用户
	ids := fmt.Sprintf("%d,%d", testUser1.ID, testUser2.ID)
	err := s.userService.DeleteUsers(ids)

	// 验证结果
	s.AssertNoError(err)

	// 验证用户是否真的被逻辑删除
	var deletedUser1 model.SysUser
	err = s.GetDB().Unscoped().Where("id = ?", testUser1.ID).First(&deletedUser1).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedUser1.Deleted)

	var deletedUser2 model.SysUser
	err = s.GetDB().Unscoped().Where("id = ?", testUser2.ID).First(&deletedUser2).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(1), deletedUser2.Deleted)

}

// TestUpdatePassword_NormalUpdatePassword 测试正常修改密码
func (s *UserServiceExtendTraditionalTestSuite) TestUpdatePassword_NormalUpdatePassword() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_update_password",
		Nickname: "Test Update Password",
		Password: "old_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 修改密码
	newPassword := "new_password"
	err := s.userService.UpdatePassword(testUser.ID, newPassword)

	// 验证结果
	s.AssertNoError(err)

	// 验证密码是否真的更新
	var updatedUser model.SysUser
	err = s.GetDB().Where("id = ?", testUser.ID).First(&updatedUser).Error
	s.AssertNoError(err)
	s.Assert().NotEqual(testUser.Password, updatedUser.Password)
	// 验证新密码是否能正确验证
	s.AssertNoError(bcrypt.CompareHashAndPassword([]byte(updatedUser.Password), []byte(newPassword)))

}

// TestUpdateUserStatus_NormalUpdateUserStatus 测试正常更新用户状态
func (s *UserServiceExtendTraditionalTestSuite) TestUpdateUserStatus_NormalUpdateUserStatus() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_update_status",
		Nickname: "Test Update Status",
		Password: "test_password",
		Status:   1, // 初始状态为启用
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 更新用户状态为禁用
	err := s.userService.UpdateUserStatus(testUser.ID, 0)

	// 验证结果
	s.AssertNoError(err)

	// 验证状态是否真的更新
	var updatedUser model.SysUser
	err = s.GetDB().Where("id = ?", testUser.ID).First(&updatedUser).Error
	s.AssertNoError(err)
	s.AssertEqual(int8(0), updatedUser.Status)

}

// TestGetCurrentUserInfo_UserExistsWithoutRoles 测试用户存在但没有角色
func (s *UserServiceExtendTraditionalTestSuite) TestGetCurrentUserInfo_UserExistsWithoutRoles() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_current_user",
		Nickname: "Test Current User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 获取当前用户信息
	userInfoVO, err := s.userService.GetCurrentUserInfo(testUser.Username)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(userInfoVO)
	s.AssertEqual(testUser.ID, userInfoVO.UserId)
	s.AssertEqual(testUser.Username, userInfoVO.Username)
	s.AssertEqual(testUser.Nickname, userInfoVO.Nickname)
	s.Assert().Empty(userInfoVO.Roles)
	s.Assert().Empty(userInfoVO.Perms)

}

// TestListExportUsers_NormalListExportUsers 测试正常导出用户列表
func (s *UserServiceExtendTraditionalTestSuite) TestListExportUsers_NormalListExportUsers() {
	// 创建测试用户
	testUser := &model.SysUser{
		Username: "test_export_user",
		Nickname: "Test Export User",
		Password: "test_password",
		Status:   1,
		DeptID:   1,
		Deleted:  0,
	}
	s.Require().NoError(s.CreateTestData(testUser))

	// 导出用户列表
	queryParams := query.UserPageQuery{}
	userExportVOs, err := s.userService.ListExportUsers(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(userExportVOs)
	// 验证导出列表中包含测试用户
	found := false
	for _, user := range userExportVOs {
		if user.Username == testUser.Username {
			found = true
			break
		}
	}
	s.Assert().True(found)

}

// 运行测试套件
func TestUserServiceExtend(t *testing.T) {
	suite.Run(t, new(UserServiceExtendTraditionalTestSuite))
}
