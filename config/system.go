package config

type System struct {
	Host          string `mapstructure:"host" json:"host" yaml:"host"`
	Port          int    `mapstructure:"port" json:"port" yaml:"port"` // 端口值
	LimitCountIP  int    `mapstructure:"ip-limit-count" json:"ip-limit-count" yaml:"ip-limit-count"`
	LimitTimeIP   int    `mapstructure:"ip-limit-time" json:"ip-limit-time" yaml:"ip-limit-time"`
	RouterPrefix  string `mapstructure:"router-prefix" json:"router-prefix" yaml:"router-prefix"`
	UseMultiPoint bool   `mapstructure:"use-multi-point" json:"use-multi-point" yaml:"use-multi-point"` // 多点登录拦截
}
