package test

import (
	"fmt"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/assert"
	"golang.org/x/crypto/bcrypt"
)

func TestUserServiceExtend_ListPagedUsers(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常分页查询
	t.Run("NormalPagination", func(t *testing.T) {
		// 准备测试数据
		testUser := model.SysUser{
			Username: "test_list_paged_user",
			Nickname: "Test List Paged User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.UserPageQuery{
			PageNum:  1,
			PageSize: 10,
		}
		pageResult, err := userService.ListPagedUsers(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, pageResult)
		assert.GreaterOrEqual(t, pageResult.Total, int64(1))
		assert.NotEmpty(t, pageResult.List)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})

	// 测试用例2: 带关键字查询
	t.Run("KeywordSearch", func(t *testing.T) {
		// 准备测试数据
		testUser := model.SysUser{
			Username: "test_keyword_search_user",
			Nickname: "Test Keyword Search User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.UserPageQuery{
			Keywords: "test_keyword_search_user",
			PageNum:  1,
			PageSize: 10,
		}
		pageResult, err := userService.ListPagedUsers(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, pageResult)
		assert.GreaterOrEqual(t, pageResult.Total, int64(1))
		assert.NotEmpty(t, pageResult.List)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_GetUserFormData(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 用户不存在
	t.Run("UserNotFound", func(t *testing.T) {
		userFormBO, err := userService.GetUserFormData(999999)
		assert.Error(t, err)
		assert.Equal(t, "用户不存在", err.Error())
		assert.Equal(t, bo.UserFormBO{}, userFormBO)
	})

	// 测试用例2: 用户存在
	t.Run("UserExists", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_get_form_user",
			Nickname: "Test Get Form User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 获取表单数据
		userFormBO, err := userService.GetUserFormData(testUser.ID)
		
		// 验证结果
		assert.NoError(t, err)
		assert.Equal(t, testUser.ID, userFormBO.ID)
		assert.Equal(t, testUser.Username, userFormBO.Username)
		assert.Equal(t, testUser.Nickname, userFormBO.Nickname)
		assert.Equal(t, testUser.DeptID, userFormBO.DeptID)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_SaveUser(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常保存用户
	t.Run("NormalSaveUser", func(t *testing.T) {
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

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", userFormBO.Username).Delete(&model.SysUser{})

		// 保存用户
		err := userService.SaveUser(userFormBO)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证用户是否真的插入数据库
		var savedUser model.SysUser
		result := global.DB.Where("username = ?", userFormBO.Username).First(&savedUser)
		assert.NoError(t, result.Error)
		assert.Equal(t, userFormBO.Username, savedUser.Username)
		assert.Equal(t, userFormBO.Nickname, savedUser.Nickname)
		assert.Equal(t, userFormBO.Gender, savedUser.Gender)

		// 清理测试数据
		global.DB.Where("username = ?", userFormBO.Username).Delete(&model.SysUser{})
	})

	// 测试用例2: 用户名已存在
	t.Run("DuplicateUsername", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_duplicate_user",
			Nickname: "Test Duplicate User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

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

		err := userService.SaveUser(userFormBO)
		
		// 验证结果
		assert.Error(t, err)
		assert.Equal(t, "用户名已存在", err.Error())

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_UpdateUser(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常更新用户
	t.Run("NormalUpdateUser", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_update_user",
			Nickname: "Test Update User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

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
		err := userService.UpdateUser(testUser.ID, userFormBO)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证用户是否真的更新
		var updatedUser model.SysUser
		result = global.DB.Where("id = ?", testUser.ID).First(&updatedUser)
		assert.NoError(t, result.Error)
		assert.Equal(t, userFormBO.Username, updatedUser.Username)
		assert.Equal(t, userFormBO.Nickname, updatedUser.Nickname)
		assert.Equal(t, userFormBO.Gender, updatedUser.Gender)

		// 清理测试数据
		global.DB.Where("id = ?", testUser.ID).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_DeleteUsers(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常删除用户
	t.Run("NormalDeleteUsers", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_delete_user",
			Nickname: "Test Delete User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 删除用户
		err := userService.DeleteUsers(fmt.Sprintf("%d", testUser.ID))
		
		// 验证结果
		assert.NoError(t, err)

		// 验证用户是否真的被逻辑删除
		var deletedUser model.SysUser
		result = global.DB.Unscoped().Where("id = ?", testUser.ID).First(&deletedUser)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(1), deletedUser.Deleted)

		// 清理测试数据
		global.DB.Where("id = ?", testUser.ID).Delete(&model.SysUser{})
	})

	// 测试用例2: 删除多个用户
	t.Run("DeleteMultipleUsers", func(t *testing.T) {
		// 创建测试用户1
		testUser1 := model.SysUser{
			Username: "test_delete_user_1",
			Nickname: "Test Delete User 1",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 创建测试用户2
		testUser2 := model.SysUser{
			Username: "test_delete_user_2",
			Nickname: "Test Delete User 2",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username IN ?", []string{testUser1.Username, testUser2.Username}).Delete(&model.SysUser{})

		// 插入测试用户
		result1 := global.DB.Create(&testUser1)
		assert.NoError(t, result1.Error)
		result2 := global.DB.Create(&testUser2)
		assert.NoError(t, result2.Error)

		// 删除用户
		ids := fmt.Sprintf("%d,%d", testUser1.ID, testUser2.ID)
		err := userService.DeleteUsers(ids)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证用户是否真的被逻辑删除
		var deletedUser1 model.SysUser
		result := global.DB.Unscoped().Where("id = ?", testUser1.ID).First(&deletedUser1)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(1), deletedUser1.Deleted)

		var deletedUser2 model.SysUser
		result = global.DB.Unscoped().Where("id = ?", testUser2.ID).First(&deletedUser2)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(1), deletedUser2.Deleted)

		// 清理测试数据
		global.DB.Where("id IN ?", []int64{testUser1.ID, testUser2.ID}).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_UpdatePassword(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常修改密码
	t.Run("NormalUpdatePassword", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_update_password",
			Nickname: "Test Update Password",
			Password: "old_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 修改密码
		newPassword := "new_password"
		err := userService.UpdatePassword(testUser.ID, newPassword)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证密码是否真的更新
		var updatedUser model.SysUser
		result = global.DB.Where("id = ?", testUser.ID).First(&updatedUser)
		assert.NoError(t, result.Error)
		assert.NotEqual(t, testUser.Password, updatedUser.Password)
		// 验证新密码是否能正确验证
		assert.NoError(t, bcrypt.CompareHashAndPassword([]byte(updatedUser.Password), []byte(newPassword)))

		// 清理测试数据
		global.DB.Where("id = ?", testUser.ID).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_UpdateUserStatus(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常更新用户状态
	t.Run("NormalUpdateUserStatus", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_update_status",
			Nickname: "Test Update Status",
			Password: "test_password",
			Status:   1, // 初始状态为启用
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 更新用户状态为禁用
		err := userService.UpdateUserStatus(testUser.ID, 0)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证状态是否真的更新
		var updatedUser model.SysUser
		result = global.DB.Where("id = ?", testUser.ID).First(&updatedUser)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(0), updatedUser.Status)

		// 清理测试数据
		global.DB.Where("id = ?", testUser.ID).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_GetCurrentUserInfo(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 用户存在但没有角色
	t.Run("UserExistsWithoutRoles", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_current_user",
			Nickname: "Test Current User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 获取当前用户信息
		userInfoVO, err := userService.GetCurrentUserInfo(testUser.Username)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, userInfoVO)
		assert.Equal(t, testUser.ID, userInfoVO.UserId)
		assert.Equal(t, testUser.Username, userInfoVO.Username)
		assert.Equal(t, testUser.Nickname, userInfoVO.Nickname)
		assert.Empty(t, userInfoVO.Roles)
		assert.Empty(t, userInfoVO.Perms)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}

func TestUserServiceExtend_ListExportUsers(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserServiceExtend{}

	// 测试用例1: 正常导出用户列表
	t.Run("NormalListExportUsers", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_export_user",
			Nickname: "Test Export User",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 导出用户列表
		queryParams := query.UserPageQuery{}
		userExportVOs, err := userService.ListExportUsers(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, userExportVOs)
		// 验证导出列表中包含测试用户
		found := false
		for _, user := range userExportVOs {
			if user.Username == testUser.Username {
				found = true
				break
			}
		}
		assert.True(t, found)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}




