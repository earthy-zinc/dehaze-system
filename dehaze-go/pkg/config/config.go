package config

import (
	"sync"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
)

type AppConfig struct {
	JWT      options.JWT      `mapstructure:"jwt" json:"jwt" yaml:"jwt"`
	Zap      options.Zap      `mapstructure:"zap" json:"zap" yaml:"zap"`
	Cache    options.Cache    `mapstructure:"cache" json:"cache" yaml:"cache"`
	System   options.System   `mapstructure:"system" json:"system" yaml:"system"`
	Captcha  options.Captcha  `mapstructure:"captcha" json:"captcha" yaml:"captcha"`
	DB       options.DB       `mapstructure:"db" json:"db" yaml:"db"`
	RabbitMQ options.RabbitMQ `mapstructure:"rabbitmq" json:"rabbitmq" yaml:"rabbitmq"`
	Kafka    options.Kafka    `mapstructure:"kafka" json:"kafka" yaml:"kafka"`
	Cors     options.CORS     `mapstructure:"cors" json:"cors" yaml:"cors"`
}

var (
	Config     *AppConfig
	configLock sync.RWMutex
)

func GetConfig() *AppConfig {
	configLock.RLock()
	defer configLock.RUnlock()
	if Config == nil {
		panic("global config not loaded, call config.Viper() first")
	}
	return Config
}

func setConfig(cfg *AppConfig) {
	configLock.Lock()
	defer configLock.Unlock()
	Config = cfg
}
