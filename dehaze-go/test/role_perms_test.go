package test

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/stretchr/testify/suite"
)

// RolePermsTestSuite 角色权限缓存测试套件
type RolePermsTestSuite struct {
	BaseTestSuite
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *RolePermsTestSuite) SetupSuite() {
}

// TearDownSuite 在整个测试套件结束后运行一次
func (s *RolePermsTestSuite) TearDownSuite() {
}

// TestInitRolePermsCache 测试初始化角色权限缓存
func (s *RolePermsTestSuite) TestInitRolePermsCache() {
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

	s.AssertNoError(s.CreateTestData(&testRole1))
	s.AssertNoError(s.CreateTestData(&testRole2))

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

	s.AssertNoError(s.CreateTestData(&testMenu1))
	s.AssertNoError(s.CreateTestData(&testMenu2))
	s.AssertNoError(s.CreateTestData(&testMenu3))

	// 3. 创建角色菜单关联
	roleMenu1 := model.SysRoleMenu{RoleID: testRole1.ID, MenuID: testMenu1.ID}
	roleMenu2 := model.SysRoleMenu{RoleID: testRole1.ID, MenuID: testMenu2.ID}
	roleMenu3 := model.SysRoleMenu{RoleID: testRole2.ID, MenuID: testMenu2.ID}
	roleMenu4 := model.SysRoleMenu{RoleID: testRole2.ID, MenuID: testMenu3.ID}

	s.AssertNoError(s.CreateTestData(&roleMenu1))
	s.AssertNoError(s.CreateTestData(&roleMenu2))
	s.AssertNoError(s.CreateTestData(&roleMenu3))
	s.AssertNoError(s.CreateTestData(&roleMenu4))

	// 执行测试
	err := initialize.InitRolePermsCache()
	s.AssertNoError(err)

	// 验证结果
	// 验证本地缓存中的数据
	// 检查Role1的权限
	cachedPerms1, found := global.LOCAL_CACHE.Get(common.RolePermsPrefix + testRole1.Code)
	s.Assert().True(found)
	perms1 := cachedPerms1.([]string)
	s.Assert().Contains(perms1, testMenu1.Perm)
	s.Assert().Contains(perms1, testMenu2.Perm)
	s.Assert().NotContains(perms1, testMenu3.Perm)

	// 检查Role2的权限
	cachedPerms2, found := global.LOCAL_CACHE.Get(common.RolePermsPrefix + testRole2.Code)
	s.Assert().True(found)
	perms2 := cachedPerms2.([]string)
	s.Assert().Contains(perms2, testMenu2.Perm)
	s.Assert().Contains(perms2, testMenu3.Perm)
	s.Assert().NotContains(perms2, testMenu1.Perm)
}

// TestClearRolePermsCache 测试清理角色权限缓存
func (s *RolePermsTestSuite) TestClearRolePermsCache() {
	// 先添加一些测试数据到缓存中
	testRoleCode := "TEST_CLEANUP_ROLE"
	testPerms := []string{"perm1", "perm2", "perm3"}

	// 添加到本地缓存
	global.LOCAL_CACHE.Set(common.RolePermsPrefix+testRoleCode, testPerms, 0)

	// 如果Redis可用，也添加到Redis中
	if global.REDIS != nil {
		_, err := global.REDIS.HSet(context.Background(), "role_perms", testRoleCode, "perm1,perm2,perm3").Result()
		s.AssertNoError(err)
	}

	// 执行清理
	err := initialize.ClearRolePermsCache()
	s.AssertNoError(err)

	// 验证本地缓存已清理
	_, found := global.LOCAL_CACHE.Get(common.RolePermsPrefix + testRoleCode)
	s.Assert().False(found)

	// 如果Redis可用，验证Redis已清理
	if global.REDIS != nil {
		exists, err := global.REDIS.HExists(context.Background(), "role_perms", testRoleCode).Result()
		s.AssertNoError(err)
		s.Assert().False(exists)
	}
}

// TestRolePermsTestSuite 运行测试套件
func TestRolePermsTestSuite(t *testing.T) {
	suite.Run(t, new(RolePermsTestSuite))
}
