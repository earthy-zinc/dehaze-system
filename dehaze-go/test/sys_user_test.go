package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/assert"
)

func TestUserService_GetUserAuthInfo(t *testing.T) {
	userService := &service.UserService{}

	// 测试用例1: 用户不存在
	t.Run("UserNotFound", func(t *testing.T) {
		userAuthInfo, err := userService.GetUserAuthInfo("nonexistent")
		assert.Error(t, err)
		assert.Nil(t, userAuthInfo)
	})

	// 测试用例2: 用户存在但没有角色
	t.Run("UserExistsWithoutRoles", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_user_no_roles",
			Nickname: "Test User No Roles",
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

		// 调用测试方法
		userAuthInfo, err := userService.GetUserAuthInfo(testUser.Username)
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		assert.Equal(t, testUser.ID, userAuthInfo.UserId)
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, testUser.Status, userAuthInfo.Status)
		assert.Equal(t, testUser.DeptID, userAuthInfo.DeptId)  // 修复类型匹配
		assert.Empty(t, userAuthInfo.Roles)
		assert.Empty(t, userAuthInfo.Perms)
		assert.Equal(t, int8(0), userAuthInfo.DataScope)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})

	// 测试用例3: 用户存在且有角色但角色无权限
	t.Run("UserExistsWithRolesNoPerms", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_user_with_roles",
			Nickname: "Test User With Roles",
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

		// 创建测试角色
		testRole := model.SysRole{
			Name:      "Test Role",
			Code:      "TEST_ROLE",
			Status:    1,
			DataScope: 2,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result = global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 创建用户角色关联
		type SysUserRole struct {
			UserID int64 `gorm:"column:user_id"`
			RoleID int64 `gorm:"column:role_id"`
		}

		userRole := SysUserRole{
			UserID: testUser.ID,
			RoleID: testRole.ID,
		}

		// 插入用户角色关联
		result = global.DB.Table("sys_user_role").Create(&userRole)
		assert.NoError(t, result.Error)

		// 调用测试方法
		userAuthInfo, err := userService.GetUserAuthInfo(testUser.Username)
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		assert.Equal(t, testUser.ID, userAuthInfo.UserId)
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, testUser.Status, userAuthInfo.Status)
		assert.Equal(t, testUser.DeptID, userAuthInfo.DeptId)
		assert.Contains(t, userAuthInfo.Roles, testRole.Code)
		assert.Empty(t, userAuthInfo.Perms)
		assert.Equal(t, testRole.DataScope, userAuthInfo.DataScope)

		// 清理测试数据
		global.DB.Table("sys_user_role").Where("user_id = ? AND role_id = ?", testUser.ID, testRole.ID).Delete(&SysUserRole{})
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})

	// 测试用例4: 用户存在且有角色和权限
	t.Run("UserExistsWithRolesAndPerms", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_user_with_perms",
			Nickname: "Test User With Perms",
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

		// 创建测试角色
		testRole := model.SysRole{
			Name:      "Test Role With Perms",
			Code:      "TEST_ROLE_PERMS",
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result = global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 创建测试菜单权限
		testMenu := model.SysMenu{
			Name: "Test Menu",
			Type: 4, // 按钮类型
			Perm: "test:permission",
		}

		// 清理可能存在的测试数据
		global.DB.Where("perm = ?", testMenu.Perm).Delete(&model.SysMenu{})

		// 插入测试菜单
		result = global.DB.Create(&testMenu)
		assert.NoError(t, result.Error)

		// 创建用户角色关联
		type SysUserRole struct {
			UserID int64 `gorm:"column:user_id"`
			RoleID int64 `gorm:"column:role_id"`
		}

		userRole := SysUserRole{
			UserID: testUser.ID,
			RoleID: testRole.ID,
		}

		// 插入用户角色关联
		result = global.DB.Table("sys_user_role").Create(&userRole)
		assert.NoError(t, result.Error)

		// 创建角色菜单关联
		type SysRoleMenu struct {
			RoleID int64 `gorm:"column:role_id"`
			MenuID int64 `gorm:"column:menu_id"`
		}

		roleMenu := SysRoleMenu{
			RoleID: testRole.ID,
			MenuID: testMenu.ID,
		}

		// 插入角色菜单关联
		result = global.DB.Table("sys_role_menu").Create(&roleMenu)
		assert.NoError(t, result.Error)

		// 调用测试方法
		userAuthInfo, err := userService.GetUserAuthInfo(testUser.Username)
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		assert.Equal(t, testUser.ID, userAuthInfo.UserId)
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, testUser.Status, userAuthInfo.Status)
		assert.Equal(t, testUser.DeptID, userAuthInfo.DeptId)
		assert.Contains(t, userAuthInfo.Roles, testRole.Code)
		assert.Contains(t, userAuthInfo.Perms, testMenu.Perm)
		assert.Equal(t, int8(testRole.DataScope), userAuthInfo.DataScope)  // 修复类型转换

		// 清理测试数据
		global.DB.Table("sys_role_menu").Where("role_id = ? AND menu_id = ?", testRole.ID, testMenu.ID).Delete(&SysRoleMenu{})
		global.DB.Table("sys_user_role").Where("user_id = ? AND role_id = ?", testUser.ID, testRole.ID).Delete(&SysUserRole{})
		global.DB.Where("perm = ?", testMenu.Perm).Delete(&model.SysMenu{})
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})

	// 测试用例5: 用户存在但已被禁用
	t.Run("UserExistsButDisabled", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_user_disabled",
			Nickname: "Test User Disabled",
			Password: "test_password",
			Status:   0, // 禁用状态
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 调用测试方法
		userAuthInfo, err := userService.GetUserAuthInfo(testUser.Username)
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		assert.Equal(t, testUser.ID, userAuthInfo.UserId)
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, testUser.Status, userAuthInfo.Status)
		assert.Equal(t, testUser.DeptID, userAuthInfo.DeptId)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})

	// 测试用例6: 用户存在但已被逻辑删除
	t.Run("UserExistsButDeleted", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_user_deleted",
			Nickname: "Test User Deleted",
			Password: "test_password",
			Status:   1,
			DeptID:   1,
			Deleted:  1, // 已删除
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 调用测试方法
		userAuthInfo, err := userService.GetUserAuthInfo(testUser.Username)
		assert.Error(t, err)
		assert.Nil(t, userAuthInfo)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}

func TestUserService_Login(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	userService := &service.UserService{}

	// 测试用例1: 用户名或密码错误
	t.Run("InvalidCredentials", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_login_user",
			Nickname: "Test Login User",
			// 使用bcrypt加密密码
			Password: "$2a$10$N47IXmT8C.sKUFXs1EBS9uJf8JiKEGz4rY14M1SX3w2w1aW99Mj9K", // "password"的bcrypt hash
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 尝试使用错误密码登录
		loginUser := &model.SysUser{
			Username: "test_login_user",
			Password: "wrong_password",
		}
		userAuthInfo, err := userService.Login(loginUser)
		
		// 验证结果
		assert.Error(t, err)
		assert.Nil(t, userAuthInfo)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})

	// 测试用例2: 正常登录
	t.Run("ValidLogin", func(t *testing.T) {
		// 创建测试用户
		testUser := model.SysUser{
			Username: "test_valid_login",
			Nickname: "Test Valid Login",
			// 使用bcrypt加密密码
			Password: "$2a$10$BQm8di9VTUfOlmr/VcFyB.BhurfGZVjCXYdgDPN1ZeI0yEMeByAQq", // "password"的bcrypt hash
			Status:   1,
			DeptID:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})

		// 插入测试用户
		result := global.DB.Create(&testUser)
		assert.NoError(t, result.Error)

		// 使用正确密码登录
		loginUser := &model.SysUser{
			Username: "test_valid_login",
			Password: "password",
		}
		userAuthInfo, err := userService.Login(loginUser)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		assert.Equal(t, testUser.ID, userAuthInfo.UserId)
		assert.Equal(t, testUser.Username, userAuthInfo.Username)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}






