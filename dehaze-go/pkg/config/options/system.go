package options

type System struct {
	Env           string `mapstructure:"env" json:"env" yaml:"env"`
	Host          string `mapstructure:"host" json:"host" yaml:"host"`
	Port          int    `mapstructure:"port" json:"port" yaml:"port"` // 端口值
	LimitCountIP  int    `mapstructure:"ip-limit-count" json:"ip-limit-count" yaml:"ip-limit-count"`
	LimitTimeIP   int    `mapstructure:"ip-limit-time" json:"ip-limit-time" yaml:"ip-limit-time"`
	RouterPrefix  string `mapstructure:"router-prefix" json:"router-prefix" yaml:"router-prefix"`
	UseMultiPoint bool   `mapstructure:"use-multi-point" json:"use-multi-point" yaml:"use-multi-point"` // 多点登录拦截
	TlsHost       string `mapstructure:"tls-host" json:"tls-host" yaml:"tls-host"`                       // HTTPS 重定向目标主机，留空则不限制

	// 登录安全配置
	LoginFailLimit    int `mapstructure:"login-fail-limit" json:"login-fail-limit" yaml:"login-fail-limit"`             // 登录失败次数限制，默认5次
	LoginFailLockTime int `mapstructure:"login-fail-lock-time" json:"login-fail-lock-time" yaml:"login-fail-lock-time"` // 登录失败锁定时间(秒)，默认300秒(5分钟)
}
