package options

// ES Elasticsearch 配置（与 python/java 共用根 .env 的 ES_URL/ES_USERNAME/ES_PASSWORD）
type ES struct {
	URL      string `mapstructure:"url" json:"url" yaml:"url"`                // ES 地址
	Username string `mapstructure:"username" json:"username" yaml:"username"` // 用户名（docker-compose 启用 xpack security）
	Password string `mapstructure:"password" json:"password" yaml:"password"` // 密码
}
