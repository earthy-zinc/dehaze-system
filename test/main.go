package test

import (
	"os"
	"testing"

	"github.com/earthyzinc/dehaze-go/initialize"
)

// TestMain 设置测试环境
func TestMain(m *testing.M) {
	// 设置测试配置文件
	os.Setenv("DEHAZE_CONFIG", "../config.test.yaml")

	// 初始化测试环境
	initialize.Viper()
	initialize.Zap()
	initialize.LocalCache()
	initialize.Gorm()
	initialize.Redis()

	// 运行测试
	code := m.Run()

	// 退出
	os.Exit(code)
}
