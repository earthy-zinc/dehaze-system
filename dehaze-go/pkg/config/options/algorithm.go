package options

// Algorithm 算法服务配置
type Algorithm struct {
	ServiceURL string `mapstructure:"serviceUrl" json:"serviceUrl" yaml:"serviceUrl"` // Python 算法服务地址
	Timeout    int    `mapstructure:"timeout" json:"timeout" yaml:"timeout"`          // 请求超时（秒），默认 60
}
