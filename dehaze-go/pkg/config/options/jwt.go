package options

type JWT struct {
	Key             string `mapstructure:"key" json:"key" yaml:"key" validate:"required"`                       // jwt签名
	TTL             int64  `mapstructure:"ttl" json:"ttl" yaml:"ttl" validate:"required,min=60"`                // AccessToken过期时间(秒)
	RefreshTokenTTL int64  `mapstructure:"refresh-token-ttl" json:"refresh-token-ttl" yaml:"refresh-token-ttl"` // RefreshToken过期时间(秒)，默认7天
}
