package test

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/stretchr/testify/assert"
)

// setupTest 清理并设置测试环境
func setupTest() {
	// 初始化配置和日志
	initialize.Viper()
	initialize.Zap()

	// 初始化数据库
	initialize.Gorm()

	// 初始化本地缓存
	initialize.LocalCache()

	// 初始化Redis（如果配置了）
	initialize.Redis()
}

// teardownTest 清理测试数据
func teardownTest() {
	// 清理测试用的表数据
	if global.DB != nil {
		// 清理测试角色
		global.DB.Where("code LIKE ?", "TEST_%").Delete(&model.SysRole{})
		// 清理测试菜单
		global.DB.Where("perm LIKE ?", "test:%").Delete(&model.SysMenu{})
		// 清理角色菜单关联
		global.DB.Exec("DELETE FROM sys_role_menu WHERE role_id NOT IN (SELECT id FROM sys_role WHERE code NOT LIKE 'TEST_%')")
		// 清理用户角色关联
		global.DB.Exec("DELETE FROM sys_user_role WHERE role_id NOT IN (SELECT id FROM sys_role WHERE code NOT IN ('TEST_%'))")
	}
}

func TestInitRolePermsCache(t *testing.T) {
	setupTest()
	defer teardownTest()

	// 创建测试数据
	// 1. 创建测试角色
	testRole1 := model.SysRole{
		Name:      "Test Role 1",
		Code:      "TEST_ROLE_1",
		Status:    1,
		DataScope: 1,
		Deleted:   0,
	}

	testRole2 := model.SysRole{
		Name:      "Test Role 2",
		Code:      "TEST_ROLE_2",
		Status:    1,
		DataScope: 2,
		Deleted:   0,
	}

	// 清理可能存在的测试数据
	global.DB.Where("code IN ?", []string{testRole1.Code, testRole2.Code}).Delete(&model.SysRole{})

	// 插入测试角色
	result := global.DB.Create(&testRole1)
	assert.NoError(t, result.Error)
	result = global.DB.Create(&testRole2)
	assert.NoError(t, result.Error)

	// 2. 创建测试菜单权限
	testMenu1 := model.SysMenu{
		Name: "Test Menu 1",
		Type: 4, // 按钮类型
		Perm: "test:permission1",
	}

	testMenu2 := model.SysMenu{
		Name: "Test Menu 2",
		Type: 4, // 按钮类型
		Perm: "test:permission2",
	}

	testMenu3 := model.SysMenu{
		Name: "Test Menu 3",
		Type: 4, // 按钮类型
		Perm: "test:permission3",
	}

	// 清理可能存在的测试数据
	global.DB.Where("perm IN ?", []string{testMenu1.Perm, testMenu2.Perm, testMenu3.Perm}).Delete(&model.SysMenu{})

	// 插入测试菜单
	result = global.DB.Create(&testMenu1)
	assert.NoError(t, result.Error)
	result = global.DB.Create(&testMenu2)
	assert.NoError(t, result.Error)
	result = global.DB.Create(&testMenu3)
	assert.NoError(t, result.Error)

	// 3. 创建角色菜单关联
	type SysRoleMenu struct {
		RoleID uint `gorm:"column:role_id"`
		MenuID uint `gorm:"column:menu_id"`
	}

	// Role1 关联 Menu1 和 Menu2
	roleMenu1 := SysRoleMenu{RoleID: testRole1.ID, MenuID: testMenu1.ID}
	roleMenu2 := SysRoleMenu{RoleID: testRole1.ID, MenuID: testMenu2.ID}

	// Role2 关联 Menu2 和 Menu3
	roleMenu3 := SysRoleMenu{RoleID: testRole2.ID, MenuID: testMenu2.ID}
	roleMenu4 := SysRoleMenu{RoleID: testRole2.ID, MenuID: testMenu3.ID}

	// 清理可能存在的测试关联数据
	global.DB.Table("sys_role_menu").Where("role_id IN ? AND menu_id IN ?",
		[]uint{testRole1.ID, testRole2.ID},
		[]uint{testMenu1.ID, testMenu2.ID, testMenu3.ID}).Delete(&SysRoleMenu{})

	// 插入角色菜单关联
	result = global.DB.Table("sys_role_menu").Create(&roleMenu1)
	assert.NoError(t, result.Error)
	result = global.DB.Table("sys_role_menu").Create(&roleMenu2)
	assert.NoError(t, result.Error)
	result = global.DB.Table("sys_role_menu").Create(&roleMenu3)
	assert.NoError(t, result.Error)
	result = global.DB.Table("sys_role_menu").Create(&roleMenu4)
	assert.NoError(t, result.Error)

	// 执行测试
	err := initialize.InitRolePermsCache()
	assert.NoError(t, err)

	// 验证结果
	// 1. 验证Redis中的数据（如果Redis可用）
	if global.REDIS != nil {
		// 检查Role1的权限
		perms1, err := global.REDIS.HGet(context.Background(), "role_perms", testRole1.Code).Result()
		assert.NoError(t, err)
		assert.Contains(t, perms1, testMenu1.Perm)
		assert.Contains(t, perms1, testMenu2.Perm)
		assert.NotContains(t, perms1, testMenu3.Perm)

		// 检查Role2的权限
		perms2, err := global.REDIS.HGet(context.Background(), "role_perms", testRole2.Code).Result()
		assert.NoError(t, err)
		assert.Contains(t, perms2, testMenu2.Perm)
		assert.Contains(t, perms2, testMenu3.Perm)
		assert.NotContains(t, perms2, testMenu1.Perm)
	} else {
		// 验证本地缓存中的数据
		// 检查Role1的权限
		cachedPerms1, found := global.LOCAL_CACHE.Get(common.ROLE_PERMS_PREFIX + testRole1.Code)
		assert.True(t, found)
		perms1 := cachedPerms1.([]string)
		assert.Contains(t, perms1, testMenu1.Perm)
		assert.Contains(t, perms1, testMenu2.Perm)
		assert.NotContains(t, perms1, testMenu3.Perm)

		// 检查Role2的权限
		cachedPerms2, found := global.LOCAL_CACHE.Get(common.ROLE_PERMS_PREFIX + testRole2.Code)
		assert.True(t, found)
		perms2 := cachedPerms2.([]string)
		assert.Contains(t, perms2, testMenu2.Perm)
		assert.Contains(t, perms2, testMenu3.Perm)
		assert.NotContains(t, perms2, testMenu1.Perm)
	}
}

func TestClearRolePermsCache(t *testing.T) {
	setupTest()
	defer teardownTest()

	// 先添加一些测试数据到缓存中
	testRoleCode := "TEST_CLEANUP_ROLE"
	testPerms := []string{"perm1", "perm2", "perm3"}

	// 添加到本地缓存
	global.LOCAL_CACHE.Set(common.ROLE_PERMS_PREFIX+testRoleCode, testPerms, 0)

	// 如果Redis可用，也添加到Redis中
	if global.REDIS != nil {
		_, err := global.REDIS.HSet(context.Background(), "role_perms", testRoleCode, "perm1,perm2,perm3").Result()
		assert.NoError(t, err)
	}

	// 执行清理
	err := initialize.ClearRolePermsCache()
	assert.NoError(t, err)

	// 验证本地缓存已清理
	_, found := global.LOCAL_CACHE.Get(common.ROLE_PERMS_PREFIX + testRoleCode)
	assert.False(t, found)

	// 如果Redis可用，验证Redis已清理
	if global.REDIS != nil {
		exists, err := global.REDIS.HExists(context.Background(), "role_perms", testRoleCode).Result()
		assert.NoError(t, err)
		assert.False(t, exists)
	}
}
