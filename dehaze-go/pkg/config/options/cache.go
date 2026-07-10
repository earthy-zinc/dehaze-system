package options

type Cache struct {
	Type       string       `mapstructure:"type" json:"type" yaml:"type"`                   // 缓存类型：redis/local/multi
	Redis      Redis        `mapstructure:"redis" json:"redis" yaml:"redis"`                // Redis缓存配置
	Local      Local        `mapstructure:"local" json:"local" yaml:"local"`                // 本地缓存配置
	Fallback   Fallback     `mapstructure:"fallback" json:"fallback" yaml:"fallback"`       // 降级配置
	MultiLevel MultiLevel   `mapstructure:"multiLevel" json:"multiLevel" yaml:"multiLevel"` // 多级缓存配置
	Protection Protection   `mapstructure:"protection" json:"protection" yaml:"protection"` // 缓存防护配置
	PubSub     PubSubConfig `mapstructure:"pubsub" json:"pubsub" yaml:"pubsub"`             // Pub/Sub 配置
}

type Redis struct {
	Enabled      bool     `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                // 是否启用Redis缓存
	Addr         string   `mapstructure:"addr" json:"addr" yaml:"addr"`                         // Redis服务器地址
	ClusterAddrs []string `mapstructure:"clusterAddrs" json:"clusterAddrs" yaml:"clusterAddrs"` // Redis集群地址
	Password     string   `mapstructure:"password" json:"password" yaml:"password"`             // Redis密码
	DB           int      `mapstructure:"db" json:"db" yaml:"db"`                               // Redis数据库编号
	UseCluster   bool     `mapstructure:"useCluster" json:"useCluster" yaml:"useCluster"`       // 是否使用集群模式
	Timeout      int      `mapstructure:"timeout" json:"timeout" yaml:"timeout"`                // 连接超时时间（秒）
}

type Local struct {
	DefaultExpire int  `mapstructure:"defaultExpire" json:"defaultExpire" yaml:"defaultExpire"` // 默认过期时间（秒）
	MaxSize       int  `mapstructure:"maxSize" json:"maxSize" yaml:"maxSize"`                   // 最大缓存数量
	Enabled       bool `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                   // 是否启用本地缓存
}

type Fallback struct {
	Enabled    bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`          // 是否启用降级
	Type       string `mapstructure:"type" json:"type" yaml:"type"`                   // 降级类型：local/none
	MaxRetries int    `mapstructure:"maxRetries" json:"maxRetries" yaml:"maxRetries"` // 最大重试次数
	RetryDelay int    `mapstructure:"retryDelay" json:"retryDelay" yaml:"retryDelay"` // 重试延迟（毫秒）
}

// MultiLevel 多级缓存配置
type MultiLevel struct {
	Enabled           bool `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                               // 是否启用多级缓存
	L1ExpireSeconds   int  `mapstructure:"l1ExpireSeconds" json:"l1ExpireSeconds" yaml:"l1ExpireSeconds"`       // L1（本地）缓存默认过期秒数
	L2ExpireSeconds   int  `mapstructure:"l2ExpireSeconds" json:"l2ExpireSeconds" yaml:"l2ExpireSeconds"`       // L2（Redis）缓存默认过期秒数
	AsyncWriteBack    bool `mapstructure:"asyncWriteBack" json:"asyncWriteBack" yaml:"asyncWriteBack"`          // 是否异步回写L1
	L1MaxSize         int  `mapstructure:"l1MaxSize" json:"l1MaxSize" yaml:"l1MaxSize"`                         // L1最大缓存条目数
	RandomExpireRange int  `mapstructure:"randomExpireRange" json:"randomExpireRange" yaml:"randomExpireRange"` // 过期时间随机范围（秒），防雪崩
}

// Protection 缓存防护配置
type Protection struct {
	// 布隆过滤器配置（防穿透）
	BloomFilter BloomFilterConfig `mapstructure:"bloomFilter" json:"bloomFilter" yaml:"bloomFilter"`
	// 单飞配置（防击穿）
	SingleFlight SingleFlightConfig `mapstructure:"singleFlight" json:"singleFlight" yaml:"singleFlight"`
	// 熔断器配置（防雪崩）
	CircuitBreaker CircuitBreakerConfig `mapstructure:"circuitBreaker" json:"circuitBreaker" yaml:"circuitBreaker"`
	// 空值缓存配置（防穿透）
	NullCache NullCacheConfig `mapstructure:"nullCache" json:"nullCache" yaml:"nullCache"`
}

// BloomFilterConfig 布隆过滤器配置
type BloomFilterConfig struct {
	Enabled           bool    `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                               // 是否启用
	ExpectedItems     uint    `mapstructure:"expectedItems" json:"expectedItems" yaml:"expectedItems"`             // 预期元素数量
	FalsePositiveRate float64 `mapstructure:"falsePositiveRate" json:"falsePositiveRate" yaml:"falsePositiveRate"` // 误判率
}

// SingleFlightConfig 单飞配置
type SingleFlightConfig struct {
	Enabled bool `mapstructure:"enabled" json:"enabled" yaml:"enabled"` // 是否启用
}

// CircuitBreakerConfig 熔断器配置
type CircuitBreakerConfig struct {
	Enabled          bool `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                            // 是否启用
	MaxRequests      uint `mapstructure:"maxRequests" json:"maxRequests" yaml:"maxRequests"`                // 半开状态下允许的最大请求数
	Interval         int  `mapstructure:"interval" json:"interval" yaml:"interval"`                         // 统计周期（秒）
	Timeout          int  `mapstructure:"timeout" json:"timeout" yaml:"timeout"`                            // 熔断超时时间（秒）
	FailureThreshold int  `mapstructure:"failureThreshold" json:"failureThreshold" yaml:"failureThreshold"` // 失败次数阈值
}

// NullCacheConfig 空值缓存配置
type NullCacheConfig struct {
	Enabled       bool `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                   // 是否启用空值缓存
	ExpireSeconds int  `mapstructure:"expireSeconds" json:"expireSeconds" yaml:"expireSeconds"` // 空值缓存过期时间（秒）
}

// PubSubConfig Redis Pub/Sub 配置（用于缓存失效广播）
type PubSubConfig struct {
	Enabled        bool   `mapstructure:"enabled" json:"enabled" yaml:"enabled"`                      // 是否启用 Pub/Sub 缓存失效广播
	Channel        string `mapstructure:"channel" json:"channel" yaml:"channel"`                      // 订阅频道名称
	SenderID       string `mapstructure:"senderId" json:"senderId" yaml:"senderId"`                   // 实例标识（可选，默认使用 hostname）
	MaxConcurrency int    `mapstructure:"maxConcurrency" json:"maxConcurrency" yaml:"maxConcurrency"` // handler最大并发数（默认16）
}
