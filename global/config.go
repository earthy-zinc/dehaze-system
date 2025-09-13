package global

import (
	"github.com/earthyzinc/dehaze-go/config"
)

type Server struct {
	JWT   config.JWT   `mapstructure:"jwt" json:"jwt" yaml:"jwt"`
	Zap   config.Zap   `mapstructure:"zap" json:"zap" yaml:"zap"`
	Redis config.Redis `mapstructure:"redis" json:"redis" yaml:"redis"`
	// Mongo     Mongo   `mapstructure:"mongo" json:"mongo" yaml:"mongo"`
	// Email     Email   `mapstructure:"email" json:"email" yaml:"email"`
	System config.System `mapstructure:"system" json:"system" yaml:"system"`
	// Captcha   Captcha `mapstructure:"captcha" json:"captcha" yaml:"captcha"`

	// gorm
	Db config.DB `mapstructure:"db" json:"db" yaml:"db"`

	// oss
	// Local        Local        `mapstructure:"local" json:"local" yaml:"local"`
	// Minio        Minio        `mapstructure:"minio" json:"minio" yaml:"minio"`

	// Excel Excel `mapstructure:"excel" json:"excel" yaml:"excel"`

	// 跨域配置
	Cors config.CORS `mapstructure:"cors" json:"cors" yaml:"cors"`
}
