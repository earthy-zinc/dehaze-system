package options

// CORS 跨域配置。白名单只保留来源列表，方法/请求头/响应头/凭证为全局统一值
// （原先每个来源各写一遍，16 项里除 allow-origin 外完全相同）。
type CORS struct {
	Mode string `mapstructure:"mode" json:"mode" yaml:"mode"`
	// AllowedOrigins 逗号分隔的跨域来源白名单，由根 .env 的 CORS_ALLOWED_ORIGINS 注入
	// （三端同名：Java 读同一变量、Python 同名配置项）
	AllowedOrigins   string `mapstructure:"allowed-origins" json:"allowed-origins" yaml:"allowed-origins"`
	AllowMethods     string `mapstructure:"allow-methods" json:"allow-methods" yaml:"allow-methods"`
	AllowHeaders     string `mapstructure:"allow-headers" json:"allow-headers" yaml:"allow-headers"`
	ExposeHeaders    string `mapstructure:"expose-headers" json:"expose-headers" yaml:"expose-headers"`
	AllowCredentials bool   `mapstructure:"allow-credentials" json:"allow-credentials" yaml:"allow-credentials"`
}
