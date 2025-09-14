package config

type JWT struct {
	Key string `mapstructure:"key" json:"key" yaml:"key" validate:"required"`        // jwt签名
	TTL int64  `mapstructure:"ttl" json:"ttl" yaml:"ttl" validate:"required,min=60"` // 过期时间
}
