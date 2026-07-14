package options

// Algorithm 算法服务配置
type Algorithm struct {
	ServiceURL     string `mapstructure:"serviceUrl" json:"serviceUrl" yaml:"serviceUrl"`           // Python 算法服务地址
	Timeout        int    `mapstructure:"timeout" json:"timeout" yaml:"timeout"`                    // 请求超时（秒），默认 60
	ConnectTimeout int    `mapstructure:"connectTimeout" json:"connectTimeout" yaml:"connectTimeout"` // 连接超时（秒），默认 5
	// 重试配置
	MaxRetry       int    `mapstructure:"maxRetry" json:"maxRetry" yaml:"maxRetry"`                   // 最大重试次数（0 表示不重试），默认 3
	RetryBackoffMs int    `mapstructure:"retryBackoffMs" json:"retryBackoffMs" yaml:"retryBackoffMs"` // 重试初始退避（毫秒），指数增长，默认 1000
	// 熔断器配置（复用 cache.go 中的 CircuitBreakerConfig 类型）
	CircuitBreaker CircuitBreakerConfig `mapstructure:"circuitBreaker" json:"circuitBreaker" yaml:"circuitBreaker"`
}
