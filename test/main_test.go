package test

import (
	"os"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
)

// TestMain 设置测试环境
func TestMain(m *testing.M) {
	os.Setenv("DEHAZE_CONFIG", "../config.test.yaml")
	// 初始化配置和日志
	initialize.Viper()
	initialize.Zap()

	// 初始化数据库
	initialize.Gorm()

	// 初始化本地缓存
	initialize.LocalCache()

	// 初始化Redis（如果配置了）
	initialize.Redis()

	initialize.Migrate()

	defer teardownTest()

	// 运行测试
	code := m.Run()

	// 退出
	os.Exit(code)
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
