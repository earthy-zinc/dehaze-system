package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/assert"
	"gorm.io/gorm"
)

func TestUserService_GetUserAuthInfo(t *testing.T) {
	userService := &service.UserService{}

	// 测试用例1: 用户不存在
	t.Run("UserNotFound", func(t *testing.T) {
		userAuthInfo, err := userService.GetUserAuthInfo("nonexistent")
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		assert.Equal(t, int64(0), userAuthInfo.UserId)
		assert.Equal(t, "", userAuthInfo.Username)
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
		assert.Equal(t, testUser.ID, uint(userAuthInfo.UserId))
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, int(testUser.Status), userAuthInfo.Status)
		assert.Equal(t, int64(testUser.DeptID), userAuthInfo.DeptId)
		assert.Empty(t, userAuthInfo.Roles)
		assert.Empty(t, userAuthInfo.Perms)
		assert.Equal(t, 0, userAuthInfo.DataScope)

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
		assert.Equal(t, testUser.ID, uint(userAuthInfo.UserId))
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, int(testUser.Status), userAuthInfo.Status)
		assert.Equal(t, int64(testUser.DeptID), userAuthInfo.DeptId)
		assert.Contains(t, userAuthInfo.Roles, testRole.Code)
		assert.Empty(t, userAuthInfo.Perms)
		assert.Equal(t, int(testRole.DataScope), userAuthInfo.DataScope)

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
		assert.Equal(t, testUser.ID, uint(userAuthInfo.UserId))
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, int(testUser.Status), userAuthInfo.Status)
		assert.Equal(t, int64(testUser.DeptID), userAuthInfo.DeptId)
		assert.Contains(t, userAuthInfo.Roles, testRole.Code)
		assert.Contains(t, userAuthInfo.Perms, testMenu.Perm)
		assert.Equal(t, int(testRole.DataScope), userAuthInfo.DataScope)

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
		assert.Equal(t, testUser.ID, uint(userAuthInfo.UserId))
		assert.Equal(t, testUser.Username, userAuthInfo.Username)
		assert.Equal(t, testUser.Nickname, userAuthInfo.Nickname)
		assert.Equal(t, testUser.Password, userAuthInfo.Password)
		assert.Equal(t, int(testUser.Status), userAuthInfo.Status)
		assert.Equal(t, int64(testUser.DeptID), userAuthInfo.DeptId)

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
		assert.NoError(t, err)
		assert.NotNil(t, userAuthInfo)
		// 应该找不到已删除的用户
		assert.Equal(t, int64(0), userAuthInfo.UserId)

		// 清理测试数据
		global.DB.Where("username = ?", testUser.Username).Delete(&model.SysUser{})
	})
}

func TestUserService_GetUserAuthInfo_DBError(t *testing.T) {
	// 保存原始DB连接
	originalDB := global.DB
	defer func() {
		global.DB = originalDB
	}()

	// 模拟数据库连接错误
	global.DB = &gorm.DB{Error: gorm.ErrInvalidDB}

	userService := &service.UserService{}

	userAuthInfo, err := userService.GetUserAuthInfo("test")
	assert.Error(t, err)
	assert.Nil(t, userAuthInfo)
}
