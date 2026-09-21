package testutil

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/go-playground/validator/v10"
	"github.com/joho/godotenv"
	"github.com/spf13/viper"

	"github.com/earthyzinc/dehaze-go/pkg/config"
)

var (
	loadConfigOnce sync.Once // 进程级只加载一次：go test ./... 各包是独立测试二进制，进程内重复加载无意义
	loadConfigErr  error
	testConfig     *config.AppConfig
)

// LoadTestConfig 加载 config/config.test.yaml 到全局 config.Config 并返回。
//
// 不走 pkg/config 的 viper.Init()：测试不需要 WatchConfig 与系统事件重载，且要求加载失败
// 直接暴露在 t.Fatalf 上。路径经 config.GoRoot() 锚定，config.test.yaml 中的
// ${MYSQL_HOST}/${MYSQL_PASSWORD} 等按基础设施分区的变量经 os.ExpandEnv 展开（凭证来自仓库根 .env），
// 与进程 CWD 完全解耦。任何一步失败 fail-fast，错误信息带具体原因。
func LoadTestConfig(t *testing.T) *config.AppConfig {
	t.Helper()
	loadConfigOnce.Do(func() {
		goRoot := config.GoRoot()
		envPath := filepath.Join(goRoot, "..", ".env")
		if err := godotenv.Load(envPath); err != nil {
			loadConfigErr = fmt.Errorf("加载仓库根 .env (%s) 失败: %w", envPath, err)
			return
		}
		configPath := filepath.Join(goRoot, "config", "config.test.yaml")
		raw, err := os.ReadFile(configPath)
		if err != nil {
			loadConfigErr = fmt.Errorf("读取测试配置 (%s) 失败: %w", configPath, err)
			return
		}
		v := viper.New()
		v.SetConfigType("yaml")
		if err := v.ReadConfig(strings.NewReader(os.ExpandEnv(string(raw)))); err != nil {
			loadConfigErr = fmt.Errorf("解析测试配置失败: %w", err)
			return
		}
		var c config.AppConfig
		if err := v.Unmarshal(&c); err != nil {
			loadConfigErr = fmt.Errorf("反序列化测试配置失败: %w", err)
			return
		}
		if err := validator.New().Struct(c); err != nil {
			loadConfigErr = fmt.Errorf("测试配置校验失败: %w", err)
			return
		}
		config.Config = &c
		testConfig = &c
	})
	if loadConfigErr != nil {
		t.Fatalf("testutil.LoadTestConfig: %v", loadConfigErr)
	}
	return testConfig
}

