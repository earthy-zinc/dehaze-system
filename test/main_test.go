package test

import (
	"os"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
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
		global.DB.Raw(
			"drop table if exists dehaze_test.sys_menu",
		)
		global.DB.Raw(
			"drop table if exists dehaze_test.sys_role",
		)
		global.DB.Raw(
			"drop table if exists dehaze_test.sys_user",
		)
		global.DB.Raw(
			"drop table if exists dehaze_test.sys_role_menu",
		)
		global.DB.Raw(
			"drop table if exists dehaze_test.sys_user_role",
		)
	}
}
