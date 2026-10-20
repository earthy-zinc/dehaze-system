package protection

import (
	"context"
	"time"
)

// BloomFilterer 布隆过滤器接口（防穿透）
type BloomFilterer interface {
	// Add 添加元素到布隆过滤器
	Add(key string)
	// MayExist 判断元素是否可能存在（存在误判可能）
	MayExist(key string) bool
	// Reset 重置布隆过滤器
	Reset()
}

// SingleFlighter 单飞接口（防击穿）
type SingleFlighter interface {
	// Do 执行带去重的操作，相同key的并发请求只执行一次
	Do(ctx context.Context, key string, fn func() (any, error)) (any, error)
	// Forget 删除key，允许后续请求重新执行
	Forget(key string)
}

// CircuitBreaker 熔断器接口（防雪崩）
type CircuitBreaker interface {
	// Execute 通过熔断器执行操作
	Execute(fn func() error) error
	// State 获取当前熔断器状态
	State() CircuitState
	// Reset 手动重置熔断器
	Reset()
}

// CircuitState 熔断器状态
type CircuitState int

const (
	StateClosed   CircuitState = iota // 关闭状态，正常工作
	StateOpen                         // 打开状态，快速失败
	StateHalfOpen                     // 半开状态，尝试恢复
)

// NullCacher 空值缓存接口（防穿透）
type NullCacher interface {
	// IsNullValue 检查值是否为空值标记
	IsNullValue(value string) bool
	// SetNull 设置空值缓存
	SetNull(ctx context.Context, key string, expiration time.Duration) error
	// GetNullValue 获取空值标记字符串
	GetNullValue() string
}

// DataLoader 数据加载器接口，用于从数据源加载数据
type DataLoader func(ctx context.Context, key string) (string, error)
