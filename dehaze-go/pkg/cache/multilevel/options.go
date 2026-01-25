package multilevel

import (
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/protection"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
)

// Options 多级缓存配置
type Options struct {
	L1Cache types.ICache // L1 本地缓存
	L2Cache types.ICache // L2 远程缓存（Redis）

	L1DefaultExpire time.Duration // L1 默认过期时间
	L2DefaultExpire time.Duration // L2 默认过期时间

	// 过期时间随机范围，防雪崩
	RandomExpireRange time.Duration

	// 异步回写L1
	AsyncWriteBack bool

	// 防护组件
	BloomFilter  protection.BloomFilterer
	SingleFlight protection.SingleFlighter
	Breaker      protection.CircuitBreaker
	NullCache    protection.NullCacher
}

// Option 配置函数
type Option func(*Options)

// WithL1Cache 设置L1缓存
func WithL1Cache(cache types.ICache) Option {
	return func(o *Options) {
		o.L1Cache = cache
	}
}

// WithL2Cache 设置L2缓存
func WithL2Cache(cache types.ICache) Option {
	return func(o *Options) {
		o.L2Cache = cache
	}
}

// WithL1DefaultExpire 设置L1默认过期时间
func WithL1DefaultExpire(d time.Duration) Option {
	return func(o *Options) {
		o.L1DefaultExpire = d
	}
}

// WithL2DefaultExpire 设置L2默认过期时间
func WithL2DefaultExpire(d time.Duration) Option {
	return func(o *Options) {
		o.L2DefaultExpire = d
	}
}

// WithRandomExpireRange 设置随机过期范围（防雪崩）
func WithRandomExpireRange(d time.Duration) Option {
	return func(o *Options) {
		o.RandomExpireRange = d
	}
}

// WithAsyncWriteBack 设置异步回写
func WithAsyncWriteBack(async bool) Option {
	return func(o *Options) {
		o.AsyncWriteBack = async
	}
}

// WithBloomFilter 设置布隆过滤器
func WithBloomFilter(bf protection.BloomFilterer) Option {
	return func(o *Options) {
		o.BloomFilter = bf
	}
}

// WithSingleFlight 设置单飞
func WithSingleFlight(sf protection.SingleFlighter) Option {
	return func(o *Options) {
		o.SingleFlight = sf
	}
}

// WithBreaker 设置熔断器
func WithBreaker(b protection.CircuitBreaker) Option {
	return func(o *Options) {
		o.Breaker = b
	}
}

// WithNullCache 设置空值缓存
func WithNullCache(nc protection.NullCacher) Option {
	return func(o *Options) {
		o.NullCache = nc
	}
}

// defaultOptions 默认配置
func defaultOptions() *Options {
	return &Options{
		L1DefaultExpire:   5 * time.Minute,
		L2DefaultExpire:   30 * time.Minute,
		RandomExpireRange: 60 * time.Second,
		AsyncWriteBack:    true,
	}
}
