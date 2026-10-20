package config

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/fsnotify/fsnotify"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"github.com/joho/godotenv"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

// 配置文件里的环境变量占位符：${VAR} 或 ${VAR:default}
var envPlaceholder = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}`)

// expandEnv 展开 ${VAR} / ${VAR:default}：
//   - 带默认值的占位符（含显式空默认 ${VAR:}）缺失时取默认值，供"生产取默认、pre 需覆盖"的
//     cookie.secure / 限流阈值这类配置，以及刻意可留空的 UPLOAD_DIR / XXLJOB_EXECUTOR_IP；
//   - 不带默认值的占位符视为必填，缺失即报错——库名 / bucket / vhost / 索引前缀这类环境身份
//     变量若静默展开为空，会写进生产预留命名空间，必须在启动时暴露。
//
// 变量已设置但为空串时按显式空值处理（不取默认值），与 LookupEnv 语义一致。
// os.ExpandEnv 既不支持默认值也无必填语义，故自行实现。
func expandEnv(content string) (string, error) {
	missing := make(map[string]struct{})
	expanded := envPlaceholder.ReplaceAllStringFunc(content, func(match string) string {
		groups := envPlaceholder.FindStringSubmatch(match)
		if value, ok := os.LookupEnv(groups[1]); ok {
			return value
		}
		if strings.Contains(match, ":") {
			return groups[2]
		}
		missing[groups[1]] = struct{}{}
		return match
	})
	if len(missing) > 0 {
		names := make([]string, 0, len(missing))
		for name := range missing {
			names = append(names, name)
		}
		sort.Strings(names)
		return "", fmt.Errorf("配置缺少必需的环境变量: %s（请在仓库根 .env 显式提供，模板见 .env.dev.example / .env.pre.example）", strings.Join(names, ", "))
	}
	return expanded, nil
}

func getConfigName() (config string) {
	switch gin.Mode() {
	case gin.DebugMode:
		config = ConfigName
	case gin.ReleaseMode:
		config = ConfigProductionName
	case gin.TestMode:
		config = ConfigTestName
	}

	return config
}

func Init() (*AppConfig, error) {
	envFile := filepath.Join(RepoRoot(), ".env")
	if err := godotenv.Load(envFile); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("加载.env(%s)失败: %w", envFile, err)
	}

	v := viper.New()
	v.AutomaticEnv()
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.SetConfigName(getConfigName())
	v.SetConfigType("yaml")
	v.AddConfigPath(GoRoot())
	v.AddConfigPath(filepath.Join(GoRoot(), "config"))

	// 先查找配置文件路径
	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("viper 读取配置失败: %w", err)
	}

	// 读取配置文件内容并展开环境变量
	configFile := v.ConfigFileUsed()
	configContent, err := os.ReadFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("读取配置文件失败: %w", err)
	}
	expandedContent, err := expandEnv(string(configContent))
	if err != nil {
		return nil, err
	}

	// 重新读取展开后的配置
	v.SetConfigFile(configFile)
	if err := v.ReadConfig(strings.NewReader(expandedContent)); err != nil {
		return nil, fmt.Errorf("解析配置失败: %w", err)
	}

	var cfg AppConfig
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("反序列化配置时发生了错误: %w", err)
	}
	normalizePaths(&cfg)

	validate := validator.New()
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("配置校验失败: %w", err)
	}

	setConfig(&cfg)

	v.WatchConfig()
	v.OnConfigChange(func(e fsnotify.Event) {
		zap.S().Infof("配置文件已变化: %s", e.Name)

		// 重新读取配置文件内容并展开环境变量
		configContent, err := os.ReadFile(e.Name)
		if err != nil {
			zap.S().Errorf("读取配置文件失败: %v", err)
			return
		}
		expandedContent, err := expandEnv(string(configContent))
		if err != nil {
			zap.S().Errorf("配置环境变量展开失败: %v", err)
			return
		}

		// 重新解析展开后的配置
		if err := v.ReadConfig(strings.NewReader(expandedContent)); err != nil {
			zap.S().Errorf("解析配置失败: %v", err)
			return
		}

		var newCfg AppConfig
		if err := v.Unmarshal(&newCfg); err != nil {
			zap.S().Errorf("重新加载配置失败: %v", err)
			return
		}
		normalizePaths(&newCfg)

		if err := validate.Struct(newCfg); err != nil {
			zap.S().Errorf("新配置校验失败: %v", err)
			return
		}

		setConfig(&newCfg)

		if err := GlobalSystemEvents.TriggerReload(); err != nil {
			zap.S().Errorf("重载系统失败: %v", err)
		}
	})
	return &cfg, nil
}
