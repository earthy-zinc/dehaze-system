package config

import (
	"fmt"
	"os"
	"strings"

	"github.com/fsnotify/fsnotify"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
	"github.com/joho/godotenv"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

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
	if err := godotenv.Load(); err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("加载.env失败: %w", err)
	}

	v := viper.New()
	v.AutomaticEnv()
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.SetConfigName(getConfigName())
	v.SetConfigType("yaml")
	v.AddConfigPath(".")
	v.AddConfigPath("./config")

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
	expandedContent := os.ExpandEnv(string(configContent))

	// 重新读取展开后的配置
	v.SetConfigFile(configFile)
	if err := v.ReadConfig(strings.NewReader(expandedContent)); err != nil {
		return nil, fmt.Errorf("解析配置失败: %w", err)
	}

	var cfg AppConfig
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("反序列化配置时发生了错误: %w", err)
	}

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
		expandedContent := os.ExpandEnv(string(configContent))

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
