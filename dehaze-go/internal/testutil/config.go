package testutil

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
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
// 为什么不用 pkg/config 的 viper.Init()：其 AddConfigPath(".")/godotenv.Load("../.env")
// 均基于进程 CWD，而 go test 的 CWD 是各包目录，配置永远找不到（死配置根因）。
// 这里用 runtime.Caller 定位 dehaze-go 根、显式绝对路径加载，config.test.yaml 中的
// ${MYSQL_HOST}/${MYSQL_PASSWORD} 等按基础设施分区的变量经 os.ExpandEnv 展开（凭证来自仓库根 .env），
// 与 CWD 完全解耦。不做 WatchConfig：测试进程生命周期短，不需要热重载。
// 任何一步失败 fail-fast，错误信息带具体原因。
func LoadTestConfig(t *testing.T) *config.AppConfig {
	t.Helper()
	loadConfigOnce.Do(func() {
		goRoot := goRepoRoot()
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

// goRepoRoot 定位 dehaze-go 根目录：testutil 位于 internal/testutil，上两级即根。
func goRepoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}
