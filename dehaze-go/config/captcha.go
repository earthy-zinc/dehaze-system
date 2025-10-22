package config

type Captcha struct {
	Length     int `mapstructure:"length" json:"length" yaml:"length" validate:"required,min=1,max=10"` // 验证码长度
	Width      int `mapstructure:"width" json:"width" yaml:"width" validate:"required"`                 // 验证码宽度
	Height     int `mapstructure:"height" json:"height" yaml:"height" validate:"required"`              // 验证码高度
	RetryCount int `mapstructure:"retry-count" json:"retry-count" yaml:"retry-count"`                   // 防爆破验证码开启此数，0代表每次登录都需要验证码，其他数字代表错误密码次数，如3代表错误三次后出现验证码
	TimeOut    int `mapstructure:"timeout" json:"timeout" yaml:"timeout"`                               // 防爆破验证码超时时间，单位：s(秒)
}
