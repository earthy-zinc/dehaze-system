package config

type JWT struct {
	Key string `mapstructure:"key" json:"key" yaml:"key"` // jwt签名
	TTL int64  `mapstructure:"ttl" json:"ttl" yaml:"ttl"` // 过期时间
}
