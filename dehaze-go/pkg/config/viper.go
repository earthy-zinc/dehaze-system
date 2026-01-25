package config

import (
	"fmt"

	"github.com/fsnotify/fsnotify"
	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
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
	v := viper.New()
	v.SetConfigName(getConfigName())
	v.SetConfigType("yaml")
	v.AddConfigPath(".")
	v.AddConfigPath("./config")

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("viper 读取配置失败: %w", err)
	}

	var cfg AppConfig
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("反序列化配置时发生了错误: %w", err)
	}

	validate := validator.New()
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("配置校验失败: %w", err)
	}

	v.WatchConfig()
	v.OnConfigChange(func(e fsnotify.Event) {
		zap.S().Infof("配置文件已变化: %s", e.Name)

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

	setConfig(&cfg)
	return &cfg, nil
}
