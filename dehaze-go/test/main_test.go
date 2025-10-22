package test

import (
	"os"
	"testing"

	"github.com/earthyzinc/dehaze-go/initialize"
)

// TestMain 设置测试环境
func TestMain(m *testing.M) {
	err := os.Setenv("DEHAZE_CONFIG", "../config.test.yaml")
	if err != nil {
		os.Exit(-1)
	}
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

	// 运行测试
	code := m.Run()
	// 退出
	os.Exit(code)
}
